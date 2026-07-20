"""
Genesis FastAPI server.

Endpoints:
  POST  /synthesize           — start a run, returns {run_id}
  POST  /synthesize/compare   — start 3 concurrent runs (Random/Evolutionary/Genesis), returns {run_ids}
  WS    /ws/runs/{run_id}     — stream GenerationEvent JSON frames
  GET   /runs/{run_id}/result — final SynthesisResult (polling fallback)
  GET   /problems             — Tier-1 problem library (no test_cases in response)
  GET   /problems/{slug}      — single problem with test_cases
  GET   /health               — {"status": "ok"}

Run with:
  uvicorn demo.server:app --reload --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure src is importable when running from repo root
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthesis.engine import SynthesisEngine, PureEvolutionaryEngine, RandomSearchEngine, SynthesisResult
from src.synthesis.events import GenerationEvent
from src.symbolic.evaluator import TestCase
from demo.problems import get_problem, list_problems

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

app = FastAPI(
    title="Genesis — Neuro-Symbolic Program Synthesis",
    description="Real-time program synthesis engine with WebSocket streaming.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tightened in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve built frontend if it exists
FRONTEND_BUILD = Path(__file__).parent / "frontend" / "dist"
if FRONTEND_BUILD.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_BUILD), html=True), name="frontend")

# Thread pool for running blocking engine generators
_executor = ThreadPoolExecutor(max_workers=12)

# ------------------------------------------------------------------
# In-memory run state (TTL-evicted)
# ------------------------------------------------------------------

RUN_TTL_SECONDS = 3600  # 1 hour

@dataclass
class RunState:
    run_id: str
    engine_type: Literal['genesis', 'pure_evolutionary', 'random']
    queue: asyncio.Queue
    result: Optional[Dict[str, Any]] = None
    done: bool = False
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)

_runs: Dict[str, RunState] = {}


def _evict_stale_runs():
    now = time.time()
    stale = [rid for rid, rs in _runs.items() if now - rs.created_at > RUN_TTL_SECONDS]
    for rid in stale:
        del _runs[rid]


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class ExamplePair(BaseModel):
    inputs: Dict[str, Any]   # e.g. {"x": 3}
    output: Any               # e.g. 9

class SynthesizeRequest(BaseModel):
    examples: Optional[List[ExamplePair]] = None
    problem_slug: Optional[str] = None    # if set, use preset problem
    max_generations: int = 100
    pop_size: int = 80
    use_neural_scorer: bool = True
    engine_type: Literal['genesis', 'pure_evolutionary', 'random'] = 'genesis'
    seed: Optional[int] = None

class CompareRequest(BaseModel):
    examples: Optional[List[ExamplePair]] = None
    problem_slug: Optional[str] = None
    max_generations: int = 100
    pop_size: int = 60         # smaller pop for faster 3-way race
    seed: Optional[int] = None

class SynthesizeResponse(BaseModel):
    run_id: str
    engine_type: str

class CompareResponse(BaseModel):
    run_ids: Dict[Literal['genesis', 'pure_evolutionary', 'random'], str]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_test_cases(request_examples: Optional[List[ExamplePair]]) -> List[TestCase]:
    if not request_examples:
        return []
    return [({'**kwargs': None, **ex.inputs}, ex.output) for ex in request_examples]

def _build_test_cases_from_problem(slug: str) -> tuple[List[TestCase], List[str]]:
    problem = get_problem(slug)
    return problem['test_cases'], problem['variables']

def _resolve_test_cases(
    examples: Optional[List[ExamplePair]],
    problem_slug: Optional[str],
) -> tuple[List[TestCase], List[str]]:
    if problem_slug:
        return _build_test_cases_from_problem(problem_slug)
    if examples:
        test_cases = [(ex.inputs, ex.output) for ex in examples]
        variables = list(examples[0].inputs.keys()) if examples else ['x']
        return test_cases, variables
    raise ValueError("Either 'examples' or 'problem_slug' must be provided.")


def _run_engine_thread(
    engine,
    test_cases: List[TestCase],
    run_id: str,
    loop: asyncio.AbstractEventLoop,
    state: RunState,
):
    """Run a blocking engine generator in a thread, pushing events into the async queue."""
    try:
        last_event = None
        for event in engine.run_stream(test_cases, run_id):
            last_event = event
            future = asyncio.run_coroutine_threadsafe(
                state.queue.put(event.to_dict()), loop
            )
            future.result()  # block until queued (backpressure)

        # Signal completion
        if last_event:
            state.result = {
                'run_id': run_id,
                'engine_type': last_event.engine_type,
                'program': last_event.best_individual.expr,
                'generations_taken': last_event.generation + 1,
                'best_fitness': last_event.best_individual.fitness,
                'fitness_curve': last_event.fitness_curve,
                'solved': last_event.event_type == 'solved',
            }
    except Exception as e:
        state.error = traceback.format_exc()
    finally:
        state.done = True
        asyncio.run_coroutine_threadsafe(state.queue.put(None), loop)  # sentinel


def _make_engine(engine_type: str, pop_size: int, max_gen: int, variables: List[str], seed: Optional[int]):
    if seed is not None:
        import random
        random.seed(seed)
    if engine_type == 'genesis':
        return SynthesisEngine(pop_size=pop_size, max_gen=max_gen, variables=variables, use_neural_scorer=True)
    elif engine_type == 'pure_evolutionary':
        return PureEvolutionaryEngine(pop_size=pop_size, max_gen=max_gen, variables=variables)
    else:
        return RandomSearchEngine(pop_size=pop_size, max_gen=max_gen, variables=variables)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/problems")
async def get_problems():
    return list_problems()


@app.get("/problems/{slug}")
async def get_problem_detail(slug: str):
    try:
        p = get_problem(slug)
        # Serialize test_cases for JSON
        serialized = {
            **{k: v for k, v in p.items() if k != 'test_cases'},
            'test_cases': [
                {'inputs': tc[0], 'output': tc[1]}
                for tc in p['test_cases']
            ]
        }
        return serialized
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    _evict_stale_runs()

    try:
        test_cases, variables = _resolve_test_cases(request.examples, request.problem_slug)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    run_id = uuid.uuid4().hex[:12]
    loop = asyncio.get_event_loop()
    state = RunState(run_id=run_id, engine_type=request.engine_type, queue=asyncio.Queue(maxsize=50))
    _runs[run_id] = state

    engine = _make_engine(request.engine_type, request.pop_size, request.max_generations, variables, request.seed)
    _executor.submit(_run_engine_thread, engine, test_cases, run_id, loop, state)

    return SynthesizeResponse(run_id=run_id, engine_type=request.engine_type)


@app.post("/synthesize/compare", response_model=CompareResponse)
async def synthesize_compare(request: CompareRequest):
    _evict_stale_runs()

    try:
        test_cases, variables = _resolve_test_cases(request.examples, request.problem_slug)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    loop = asyncio.get_event_loop()
    run_ids = {}

    for engine_type in ('random', 'pure_evolutionary', 'genesis'):
        run_id = uuid.uuid4().hex[:12]
        state = RunState(run_id=run_id, engine_type=engine_type, queue=asyncio.Queue(maxsize=50))
        _runs[run_id] = state
        run_ids[engine_type] = run_id

        # Offset seeds so engines explore different trees (same seed = same init = boring)
        seed_offset = {'random': 0, 'pure_evolutionary': 1000, 'genesis': 2000}
        seed = (request.seed + seed_offset[engine_type]) if request.seed is not None else None
        engine = _make_engine(engine_type, request.pop_size, request.max_generations, variables, seed)
        _executor.submit(_run_engine_thread, engine, test_cases, run_id, loop, state)

    return CompareResponse(run_ids=run_ids)


@app.get("/runs/{run_id}/result")
async def get_result(run_id: str):
    state = _runs.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found (may have expired).")
    if not state.done:
        return {"status": "running", "run_id": run_id}
    if state.error:
        return {"status": "error", "run_id": run_id, "error": state.error}
    return {"status": "complete", **state.result}


@app.websocket("/ws/runs/{run_id}")
async def ws_stream(websocket: WebSocket, run_id: str):
    await websocket.accept()

    state = _runs.get(run_id)
    if state is None:
        await websocket.send_json({"error": f"Run {run_id!r} not found."})
        await websocket.close()
        return

    try:
        while True:
            # Wait for the next event (or sentinel None = done)
            try:
                event = await asyncio.wait_for(state.queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                # Keep-alive ping
                await websocket.send_json({"type": "ping"})
                continue

            if event is None:
                # Run finished
                if state.error:
                    await websocket.send_json({"type": "error", "message": state.error})
                else:
                    await websocket.send_json({"type": "done", "result": state.result})
                break

            await websocket.send_json(event)

    except WebSocketDisconnect:
        pass  # Client disconnected — engine thread keeps running, result stored in state
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
