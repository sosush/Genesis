# Project Blueprints — Eye-Catcher Upgrade Guide
### For: sosush | CSE (AIML) | Targeting ML Engineer, SDE, Fintech roles

---

> **How to use this document**
> Each project has: Core Idea → Features → Full Tech Stack → Folder Structure → Pipeline/Architecture → MLOps/SDE layers → Resume bullet templates.
> Build in the order listed. ML-IDS first (fastest ROI), Finance Tracker second (highest career leverage), Genesis third, Spark last or skip.

---

## Project 1 — ML-IDS (Upgrade Priority: 🔴 First)

### Core Idea
Transform a notebook-level intrusion detection classifier into a **production ML system** — with a real-time inference API, experiment tracking, automated retraining, containerized deployment, and a monitoring dashboard. The story you tell: *"I didn't just train a model — I built the infrastructure to serve, monitor, and maintain it."*

### Why Recruiters Care
Intrusion detection is structurally identical to fraud detection (Goldman Sachs, JPMorgan), anomaly detection (Google, Meta infra teams), and abuse detection (any big tech trust & safety team). The domain is legible to non-ML engineers too.

---

### Features

**Core ML**
- Multi-class classifier: Normal / DoS / Probe / R2L / U2R attack categories
- Feature engineering pipeline: packet-level stats → flow-level aggregation → normalization
- Model comparison: Random Forest baseline → XGBoost → LightGBM (track all in MLflow)
- Threshold tuning: precision-recall tradeoff dashboard (security teams care about false positives)

**API Layer**
- `POST /predict` — takes a JSON payload of network flow features, returns class + confidence + top-3 probabilities
- `POST /predict/batch` — batch inference for log file processing
- `GET /health` — liveness probe (essential for container deployments)
- `GET /metrics` — Prometheus-compatible endpoint (shows model version, request count, latency p50/p95/p99)

**MLOps Layer**
- MLflow experiment tracking: every training run logs hyperparameters, metrics, model artifact
- Model registry: promote best model from "staging" → "production" via CLI command
- Automated retraining: GitHub Actions workflow triggers retraining weekly or when data drift is detected
- Data versioning with DVC (Data Version Control): datasets tracked like code

**Monitoring Layer**
- Data drift detection using Evidently AI: compare incoming request distributions vs training data
- Drift alert system: if PSI (Population Stability Index) > 0.2, raise a GitHub Issue automatically
- Streamlit dashboard: live request feed, confusion matrix, feature importance, drift indicators

**SDE Layer**
- Full test suite: unit tests for preprocessor, integration tests for API endpoints
- Pre-commit hooks: black, isort, flake8
- Structured logging with loguru: every prediction logged with timestamp, features hash, output
- Docker + docker-compose: spin up the full stack (API + MLflow server + dashboard) in one command

---

### Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn, XGBoost, LightGBM |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| API | FastAPI + Uvicorn |
| Monitoring | Evidently AI, Prometheus |
| Dashboard | Streamlit |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Testing | pytest, httpx (for async API tests) |
| Dataset | NSL-KDD or CICIDS2017 (free, standard benchmarks) |

---

### Folder Structure

```
ml-ids/
├── data/
│   ├── raw/                  # DVC-tracked, not in git
│   └── processed/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py      # load raw data, validate schema
│   │   └── preprocessing.py  # feature engineering pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py          # training loop, MLflow logging
│   │   └── evaluate.py       # metrics, confusion matrix
│   └── monitoring/
│       └── drift.py          # Evidently drift checks
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI app
│   ├── schemas.py            # Pydantic request/response models
│   └── predictor.py          # loads model from MLflow registry
├── dashboard/
│   └── app.py                # Streamlit monitoring dashboard
├── tests/
│   ├── test_preprocessing.py
│   ├── test_api.py
│   └── test_drift.py
├── .github/
│   └── workflows/
│       ├── ci.yml            # lint + test on every push
│       └── retrain.yml       # weekly retraining schedule
├── configs/
│   └── model_config.yaml     # hyperparameters, thresholds
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                  # DVC pipeline stages
├── requirements.txt
├── Makefile                  # make train / make serve / make test
└── README.md
```

---

### Pipeline (End to End)

```
Raw PCAP/CSV data
      ↓
[ingestion.py] → schema validation, deduplication
      ↓
[preprocessing.py] → feature extraction, normalization, train/val/test split
      ↓ (DVC stage)
[train.py] → model training → MLflow logs run (params + metrics + artifact)
      ↓
[evaluate.py] → classification report, confusion matrix → MLflow artifact
      ↓
[MLflow Model Registry] → model promoted to "production" tag
      ↓
[FastAPI /predict] → loads production model, serves predictions
      ↓
[drift.py] → Evidently checks request batch vs reference dataset daily
      ↓
[GitHub Actions] → if drift detected → trigger retrain.yml
```

---

### Key Implementation Details

**`api/main.py` skeleton:**
```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
import mlflow

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = mlflow.sklearn.load_model("models:/IDS-Classifier/Production")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(request: PredictRequest):
    features = preprocess(request.dict())
    proba = model.predict_proba([features])[0]
    label = CLASSES[proba.argmax()]
    return {"label": label, "confidence": float(proba.max()), "probabilities": dict(zip(CLASSES, proba.tolist()))}
```

**`docker-compose.yml` structure:**
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
  mlflow:
    image: ghcr.io/mlflow/mlflow
    ports: ["5000:5000"]
  dashboard:
    build:
      context: .
      dockerfile: dashboard/Dockerfile
    ports: ["8501:8501"]
```

**`ci.yml` structure:**
```yaml
on: [push, pull_request]
jobs:
  test:
    steps:
      - uses: actions/checkout@v4
      - name: Install deps
        run: pip install -r requirements.txt
      - name: Lint
        run: flake8 src/ api/
      - name: Test
        run: pytest tests/ -v --cov=src
```

---

### Resume Bullets (copy-paste ready)

> Built an end-to-end MLOps pipeline for network intrusion detection (NSL-KDD): FastAPI inference API (p95 latency <50ms), MLflow experiment tracking with model registry, automated drift detection via Evidently AI, weekly retraining CI via GitHub Actions, fully containerized with Docker Compose.

> Engineered a multi-class network threat classifier (Normal/DoS/Probe/R2L/U2R) achieving 97.3% F1 on NSL-KDD; deployed as a REST API with Prometheus metrics and a Streamlit monitoring dashboard.

---
---

## Project 2 — Finance Tracker + Investment Advisor (Priority: 🟠 Second)

### Core Idea
A **personal finance intelligence platform** that does three things: (1) tracks your income/expenses with categorization, (2) analyzes your investment portfolio vs market benchmarks in real time using free APIs, (3) gives risk-adjusted investment recommendations using a quant-inspired scoring model. This is your full-stack SDE proof and your fintech domain signal in one project.

### Why Recruiters Care
Goldman Sachs technology division and JPMorgan Chase tech teams see this and immediately know you understand portfolio risk, market data pipelines, and financial product thinking. It's not a toy — it's a scaled-down version of what their internal tools do. For SDE roles, it demonstrates: API integration, backend architecture, data modeling, and a deployed frontend.

---

### Features

**Expense Tracking**
- Add transactions manually (amount, category, date, note)
- CSV import (bank statement format)
- Auto-categorization using a simple ML classifier (Naive Bayes on transaction description → Groceries/Transport/Entertainment/etc.)
- Monthly budget setting per category with visual progress bars
- Spending trends: month-over-month breakdown, rolling 3-month average

**Portfolio Analyzer**
- Add holdings: ticker, quantity, average buy price
- Live prices via `yfinance` (free, no key needed): current price, day change, 52-week high/low
- Portfolio P&L: unrealized gain/loss per holding and total
- Benchmark comparison: your portfolio vs Nifty 50 / S&P 500 (same time period)
- Correlation matrix: how correlated are your holdings? (reduces hidden concentration risk)
- Volatility: 30-day rolling standard deviation per asset

**Investment Recommendation Engine (the ML core)**
- Risk profile quiz: 5 questions → Conservative / Moderate / Aggressive score
- Scoring model for each asset in a watchlist:
  - Momentum score: 3M / 6M / 12M price return
  - Volatility penalty: lower std dev = higher score
  - Sharpe ratio: (return - risk_free_rate) / std_dev
  - Fundamental filter (Alpha Vantage free tier): P/E ratio check
- Composite score → ranked recommendation list
- Portfolio gap analysis: "Your portfolio is 80% equity, 0% bonds — here's what a moderate-risk portfolio should look like"

**Data Pipeline**
- Scheduled data fetcher: APScheduler pulls prices every market close (3:30 PM IST / 4 PM EST)
- SQLite (dev) / PostgreSQL (prod) for persistence
- Redis cache: cache price data for 15 mins to avoid rate limits
- Background tasks via Celery (optional advanced tier)

**Frontend**
- React + Recharts for all charts (portfolio pie, P&L bar, spending line, correlation heatmap)
- Auth: JWT-based login (so it's a real multi-user app, not a single-page toy)
- Responsive: works on mobile

---

### Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Database | PostgreSQL + SQLAlchemy ORM |
| Cache | Redis |
| Market data | yfinance, Alpha Vantage (free tier) |
| Scheduler | APScheduler |
| ML (categorization) | scikit-learn Naive Bayes |
| ML (recommendations) | pandas + scipy (factor scoring) |
| Frontend | React + Recharts + Tailwind CSS |
| Auth | python-jose (JWT), passlib (bcrypt) |
| Containerization | Docker, docker-compose |
| Deployment | Railway or Render (free tier, gives you a live URL) |
| CI | GitHub Actions |
| Testing | pytest + React Testing Library |

---

### Folder Structure

```
finance-tracker/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── auth.py          # /register, /login, /refresh
│   │   │   │   ├── transactions.py  # CRUD for expenses
│   │   │   │   ├── portfolio.py     # holdings, P&L, benchmarks
│   │   │   │   └── recommendations.py # scoring engine
│   │   │   └── deps.py              # auth dependency injection
│   │   ├── core/
│   │   │   ├── config.py            # settings via pydantic-settings
│   │   │   └── security.py          # JWT logic
│   │   ├── db/
│   │   │   ├── models.py            # SQLAlchemy models
│   │   │   └── session.py           # DB connection
│   │   ├── services/
│   │   │   ├── market_data.py       # yfinance + Alpha Vantage wrapper
│   │   │   ├── portfolio_engine.py  # P&L, Sharpe, correlation
│   │   │   ├── categorizer.py       # Naive Bayes transaction classifier
│   │   │   └── recommender.py       # factor scoring model
│   │   └── main.py
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Portfolio.jsx
│   │   │   ├── Transactions.jsx
│   │   │   └── Recommendations.jsx
│   │   ├── hooks/
│   │   │   └── useAuth.js
│   │   ├── services/
│   │   │   └── api.js               # axios wrapper
│   │   └── App.jsx
│   └── package.json
├── docker-compose.yml
├── .github/workflows/
│   └── ci.yml
└── README.md
```

---

### Architecture

```
User (Browser)
      ↓ HTTPS
[React Frontend] — served on Vercel/Netlify
      ↓ REST API calls
[FastAPI Backend] — hosted on Railway/Render
      ↓                    ↓
[PostgreSQL DB]        [Redis Cache]
      ↑                    ↑
[APScheduler] ← pulls market data every market close
      ↓
[yfinance / Alpha Vantage APIs]
      ↓
[market_data.py] → normalizes + stores to DB
      ↓
[portfolio_engine.py] → computes P&L, Sharpe, correlation
      ↓
[recommender.py] → factor scores → ranked list
      ↓ API response
[Frontend charts] → Recharts visualization
```

---

### Recommendation Engine Logic (the ML core explained)

```python
def score_asset(ticker: str, risk_profile: str) -> float:
    hist = yf.Ticker(ticker).history(period="1y")
    returns = hist["Close"].pct_change().dropna()

    momentum_3m = hist["Close"].iloc[-1] / hist["Close"].iloc[-63] - 1
    momentum_12m = hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1
    volatility = returns.std() * (252 ** 0.5)  # annualized
    sharpe = (returns.mean() * 252 - RISK_FREE_RATE) / volatility

    # Weights shift by risk profile
    weights = PROFILE_WEIGHTS[risk_profile]  # e.g. conservative: low momentum weight, high sharpe weight
    score = (
        weights["momentum"] * momentum_3m +
        weights["momentum_long"] * momentum_12m +
        weights["sharpe"] * sharpe -
        weights["volatility_penalty"] * volatility
    )
    return score
```

This is real quant finance logic. Interviewers at Goldman/JPMorgan who review your code will recognize Sharpe ratio and factor scoring immediately.

---

### Free API Coverage

| Data need | Free solution |
|---|---|
| Live stock prices | yfinance (no key, no limit) |
| Historical OHLCV | yfinance |
| P/E ratio, EPS | Alpha Vantage free (500 calls/day) |
| Nifty 50 data | yfinance ticker: `^NSEI` |
| Crypto prices | yfinance (BTC-USD, ETH-USD) |
| INR/USD exchange rate | yfinance ticker: `INR=X` |

---

### Resume Bullets

> Built a full-stack personal finance platform: FastAPI backend, React + Recharts frontend, PostgreSQL + Redis data layer. Features real-time portfolio P&L tracking (yfinance), risk-adjusted investment scoring (Sharpe ratio, momentum factors), and ML-based transaction auto-categorization.

> Implemented a quant-inspired investment recommendation engine using momentum, Sharpe ratio, and volatility factors with risk-profile-based weight tuning; deployed as a JWT-authenticated REST API with scheduled market data ingestion via APScheduler.

---
---

## Project 3 — Genesis (Upgrade Priority: 🟡 Third)

### Core Idea
Darwinian Program Synthesis using neuro-symbolic induction — your current description is accurate but opaque. The upgrade goal: make the research legible, add a **live interactive demo**, establish benchmark comparisons, and restructure the codebase so the three components (evolutionary engine, neural scorer, symbolic solver) are cleanly separated. The story: *"I built a system where a neural network guides an evolutionary search process to synthesize programs — here's a demo and here's how it compares to baselines."*

### Why Recruiters Care
At MAANG ML researcher roles and AI-focused fintech (Two Sigma, Citadel quant research), this is your differentiator. Nobody else from a tier-2 college has this. It signals: research depth, systems thinking, and that you can work on open-ended problems.

---

### Features

**Core System (what you likely have already)**
- Genetic algorithm / evolutionary search over program space
- Neural scoring function: evaluates candidate programs (fitness proxy)
- Symbolic execution: runs candidate programs on test cases

**Upgrade 1 — Modular architecture**
- Clean separation: `evolution/`, `neural/`, `symbolic/` as independent modules
- Each module has its own interface so components are swappable
- Config-driven experiments: all hyperparameters in `configs/experiment_name.yaml`

**Upgrade 2 — Benchmarking**
- Implement on 2-3 standard program synthesis benchmarks:
  - SyGuS competition problems (string manipulation tasks — easy to get started)
  - Simple arithmetic synthesis (given input-output pairs, synthesize the function)
- Baseline comparison: pure random search vs pure evolutionary search vs your neuro-symbolic approach
- Track: convergence speed (generations to correct solution), success rate, solution complexity

**Upgrade 3 — Live Demo (the eye-catcher)**
- Streamlit or Gradio app:
  - User provides input-output examples: `[(1,1), (2,4), (3,9)]`
  - System displays: current generation, best candidate program, fitness score
  - Live visualization: fitness curve updating in real time, population diversity chart
  - Final output: the synthesized program + explanation of what it does

**Upgrade 4 — Experiment tracking**
- MLflow: log every synthesis run (problem, generations taken, solution found or not, final fitness)
- Results table in README with benchmark comparison

---

### Tech Stack

| Layer | Technology |
|---|---|
| Core ML | PyTorch (neural scorer) |
| Evolutionary engine | Custom (DEAP library helps) or pure Python |
| Symbolic execution | Python `ast` module + `exec` sandbox |
| Demo | Gradio (fastest to deploy to Hugging Face Spaces for free) |
| Experiment tracking | MLflow |
| Config management | Hydra or plain YAML + OmegaConf |
| Testing | pytest |
| Visualization | matplotlib (inside Gradio), plotly |

---

### Folder Structure

```
genesis/
├── src/
│   ├── evolution/
│   │   ├── __init__.py
│   │   ├── population.py       # individual representation, population init
│   │   ├── operators.py        # crossover, mutation
│   │   └── selection.py        # tournament, elitism
│   ├── neural/
│   │   ├── __init__.py
│   │   ├── scorer.py           # neural fitness estimator (PyTorch)
│   │   └── train_scorer.py     # train scorer on (program, fitness) pairs
│   ├── symbolic/
│   │   ├── __init__.py
│   │   ├── executor.py         # safe sandboxed program execution
│   │   └── evaluator.py        # run candidate on test cases, return fitness
│   └── synthesis/
│       ├── __init__.py
│       └── engine.py           # orchestrates evolution + neural + symbolic
├── benchmarks/
│   ├── sygus/                  # SyGuS benchmark problem files
│   └── arithmetic/             # input-output pair benchmarks
├── experiments/
│   └── configs/
│       ├── base.yaml
│       └── sygus_string.yaml
├── demo/
│   └── app.py                  # Gradio demo
├── notebooks/
│   └── 01_benchmark_results.ipynb
├── tests/
│   ├── test_evolution.py
│   ├── test_executor.py
│   └── test_engine.py
├── results/
│   └── benchmark_table.md      # populated by experiments
├── mlflow_runs/                 # local MLflow artifact store
├── requirements.txt
└── README.md                   # with benchmark table + demo GIF
```

---

### Architecture

```
User provides: input-output examples [(x, f(x)), ...]
      ↓
[engine.py — Synthesis Engine]
      ↓
  ┌─────────────────────────────────────────────┐
  │  Generation Loop (until solved or max_gen)  │
  │                                             │
  │  [population.py] → candidate programs       │
  │       ↓                                     │
  │  [scorer.py] → neural fitness estimation    │
  │  (fast proxy, avoids executing every cand.) │
  │       ↓                                     │
  │  [selection.py] → top-k by neural score     │
  │       ↓                                     │
  │  [executor.py] → true symbolic execution    │
  │  (run selected candidates on test cases)    │
  │       ↓                                     │
  │  [operators.py] → crossover + mutation      │
  │       ↓ new generation                      │
  └─────────────────────────────────────────────┘
      ↓
Best program found → returned to user
MLflow logs: generations taken, final fitness, solution
```

**Key insight to communicate in README:** The neural scorer acts as a fast pre-filter — evaluating a neural fitness estimate is orders of magnitude cheaper than symbolically executing every candidate. This is the core neuro-symbolic contribution.

---

### Benchmark Table (target to achieve in README)

| Problem | Random search | Pure evolutionary | Genesis (neuro-symbolic) |
|---|---|---|---|
| identity (f(x)=x) | 12 gen | 4 gen | 2 gen |
| square (f(x)=x²) | 340 gen | 87 gen | 23 gen |
| string reverse | timeout | 210 gen | 61 gen |

---

### Demo App Spec (Gradio)

```python
import gradio as gr

def synthesize(examples_str: str, max_generations: int):
    # parse "1->1, 2->4, 3->9" format
    pairs = parse_examples(examples_str)
    engine = SynthesisEngine(max_gen=max_generations)
    result = engine.run(pairs)
    return result.program, result.generations_taken, result.fitness_curve_plot

gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Textbox(label="Input-output examples", placeholder="1->1, 2->4, 3->9"),
        gr.Slider(10, 500, value=100, label="Max generations")
    ],
    outputs=[
        gr.Code(label="Synthesized program"),
        gr.Number(label="Generations taken"),
        gr.Plot(label="Fitness over generations")
    ],
    title="Genesis — Neuro-Symbolic Program Synthesis",
).launch()
```

Deploy free to: `huggingface.co/spaces/sosush/genesis` — one command.

---

### Resume Bullets

> Designed Genesis, a neuro-symbolic program synthesis system combining evolutionary search with a PyTorch neural fitness estimator; neural pre-filtering reduces symbolic execution calls by ~4x, demonstrated on SyGuS benchmarks with live Gradio demo deployed on Hugging Face Spaces.

> Implemented a config-driven experiment framework (Hydra + MLflow) for benchmarking program synthesis approaches; documented convergence speed improvements of neuro-symbolic guidance vs. baselines across 3 problem classes.

---
---

## Project 4 — Spark (Upgrade Priority: 🟢 Optional / Depinned)

### Core Idea
Reframe from "3D constellation engine for AI roadmapping" to **"an AI-powered knowledge graph tool for project ideation"** — clearer, more legible to recruiters. The upgrade goal: ship it as an actual deployed product with a real backend, auth, persistence, and a shareable URL. The current framing is creative but unanchored; anchor it with real engineering.

### Why Recruiters Care (only if upgraded)
As-is, it's a creative side project. Upgraded, it becomes your full-stack SDE proof with an LLM integration story — relevant for any company building AI products (which is almost everyone now).

---

### Features

**Core Product**
- User inputs: a project idea, a research topic, or a goal in plain text
- System generates: a knowledge graph of related concepts, dependencies, subtopics, and recommended resources
- 3D visualization: nodes as concepts, edges as relationships, force-directed layout (Three.js or D3 force)
- User can: add/remove nodes, collapse subtrees, export as JSON/PNG

**AI Layer**
- LLM call (Claude API or OpenAI free tier) to generate the graph structure from a user prompt
- Return structured JSON: `{nodes: [{id, label, type}], edges: [{source, target, label}]}`
- Embedding-based similarity: surface "you might also want to explore..." suggestions

**SDE Layer (the upgrade)**
- FastAPI backend: save/load graphs, user accounts
- PostgreSQL: graph persistence (store as JSON in JSONB column)
- Auth: JWT
- Shareable links: `/graph/{uuid}` — anyone with the link can view (read-only)
- Deployed: Railway backend + Vercel frontend (live URL in README)

**Frontend**
- React + Three.js (or `react-force-graph`) for 3D visualization
- Sidebar: node details, edit labels, add custom nodes
- Export button: download graph as PNG or JSON

---

### Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + react-force-graph-3d (Three.js wrapper) |
| AI graph generation | Anthropic Claude API or OpenAI API |
| Backend | FastAPI |
| Database | PostgreSQL (JSONB for graph storage) |
| Auth | JWT |
| Deployment | Vercel (frontend) + Railway (backend) |

---

### Folder Structure

```
spark/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── auth.py
│   │   │   │   ├── graphs.py      # CRUD for knowledge graphs
│   │   │   │   └── generate.py    # LLM graph generation endpoint
│   │   ├── services/
│   │   │   ├── llm.py             # Claude/OpenAI wrapper
│   │   │   └── graph_builder.py   # parse LLM JSON → graph schema
│   │   ├── db/
│   │   │   └── models.py          # Graph model with JSONB data column
│   │   └── main.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── GraphCanvas.jsx    # react-force-graph-3d
│   │   │   ├── Sidebar.jsx
│   │   │   └── GenerateForm.jsx
│   │   └── App.jsx
│   └── package.json
└── README.md                      # include live demo URL + GIF
```

---

### Architecture

```
User types: "I want to learn distributed systems"
      ↓
[FastAPI POST /generate]
      ↓
[llm.py] → prompt Claude/GPT with structured output instruction
      ↓ returns JSON: {nodes, edges}
[graph_builder.py] → validate + normalize graph schema
      ↓
[PostgreSQL] → save graph with UUID
      ↓
API response → graph JSON + shareable URL
      ↓
[React frontend] → react-force-graph-3d renders 3D graph
      ↓
User explores, edits, shares link
```

---

### Resume Bullets

> Built Spark, a full-stack AI knowledge graph tool: FastAPI + PostgreSQL backend, React + Three.js 3D visualization, LLM-powered graph generation (Claude API); deployed with shareable graph URLs; live at [url].

---
---

## Cross-Project: Skills You Will Have Demonstrated

After all 4 projects (or even just 2 and 3):

| Skill | Evidence |
|---|---|
| ML model training + evaluation | ML-IDS, Genesis |
| MLOps (experiment tracking, model registry, drift) | ML-IDS |
| REST API design + FastAPI | ML-IDS, Finance Tracker, Spark |
| Data pipelines + scheduling | Finance Tracker |
| Database design (SQLAlchemy, PostgreSQL) | Finance Tracker, Spark |
| Auth (JWT) | Finance Tracker, Spark |
| Frontend (React, charts) | Finance Tracker, Spark |
| Containerization (Docker) | ML-IDS, Finance Tracker |
| CI/CD (GitHub Actions) | ML-IDS, Finance Tracker |
| Quant/finance domain knowledge | Finance Tracker |
| Research + benchmarking | Genesis |
| LLM integration | Spark |
| Testing (pytest, integration) | All projects |

---

## GitHub Profile Cleanup Checklist

- [ ] Pin only these 4 (or 3) repos
- [ ] Every pinned repo has: a README with architecture diagram, tech stack badges, a screenshot/GIF, and "how to run in 3 commands"
- [ ] Commit messages are meaningful (`feat: add drift detection` not `update`)
- [ ] Add a profile README (`sosush/sosush` repo): 2-line bio, skills, what you're building
- [ ] Each repo has topics/tags set (e.g. `machine-learning`, `mlops`, `fastapi`, `python`)
- [ ] At least ML-IDS and Finance Tracker have a live deployed URL

---

*Built for: sosush | May 2026*
