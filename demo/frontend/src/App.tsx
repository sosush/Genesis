import React, { useState, useEffect } from 'react';
import { useRunStore } from './store/runStore';
import { useGuidanceStore } from './store/guidanceStore';
import { useWebSocket } from './hooks/useWebSocket';
import { EmptyState } from './components/EmptyState';
import { SynthesisForm, RunParams } from './components/SynthesisForm';
import { TreeView3D } from './components/TreeView3D';
import { PopulationChart } from './components/PopulationChart';
import { ComparisonPane } from './components/ComparisonPane';
import { ResultSummary } from './components/ResultSummary';
import { GuidanceTour } from './components/GuidanceTour';
import { GlossaryPanel } from './components/GlossaryPanel';
import { WhatAmILooking } from './components/WhatAmILooking';
import { Problem, EngineType } from './types';

export default function App() {
  const [selectedProblem, setSelectedProblem] = useState<Problem | null>(null);
  const [activeParams, setActiveParams] = useState<RunParams | null>(null);
  const [wsTrigger, setWsTrigger] = useState(0);

  const {
    activeRunId,
    runs,
    compareMode,
    compareRunIds,
    setCompareMode,
    reset,
  } = useRunStore();

  const {
    setGlossary,
    setWhatAmI,
    glossaryOpen,
    whatAmIOpen,
    startTour,
  } = useGuidanceStore();

  // Active run details
  const activeRun = activeRunId ? runs[activeRunId] : null;
  const isRunning = activeRun?.status === 'running' ||
    (compareMode && compareRunIds && Object.values(compareRunIds).some(rid => runs[rid]?.status === 'running'));

  // Trigger WebSocket hooks based on active params
  // Standard WebSocket connection
  useWebSocket(
    !compareMode && activeParams ? activeRunId : null,
    activeParams?.engineType ?? 'genesis'
  );

  // Parallel WebSockets for Comparison Mode
  useWebSocket(compareMode && compareRunIds ? compareRunIds.genesis : null, 'genesis');
  useWebSocket(compareMode && compareRunIds ? compareRunIds.pure_evolutionary : null, 'pure_evolutionary');
  useWebSocket(compareMode && compareRunIds ? compareRunIds.random : null, 'random');

  const handleRun = async (params: RunParams) => {
    reset();
    setCompareMode(false);

    const body: Record<string, any> = {
      max_generations: params.maxGenerations,
      pop_size: params.popSize,
      engine_type: params.engineType,
    };
    if (params.problemSlug) {
      body.problem_slug = params.problemSlug;
    } else if (params.customExamples) {
      // Parse custom examples into structured API schema
      const pairs = params.customExamples.split(',').map(s => s.trim()).filter(Boolean);
      body.examples = pairs.map(p => {
        const [inp, out] = p.split('->').map(s => s.trim());
        return { inputs: { x: Number(inp) }, output: Number(out) };
      });
    }

    try {
      const res = await fetch('/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setActiveParams(params);
      useRunStore.getState().startRun(data.run_id, params.engineType);
      setWsTrigger(t => t + 1);
    } catch (err) {
      console.error(err);
    }
  };

  const handleCompare = async (params: RunParams) => {
    reset();
    setCompareMode(true);

    const body: Record<string, any> = {
      max_generations: params.maxGenerations,
      pop_size: params.popSize,
    };
    if (params.problemSlug) {
      body.problem_slug = params.problemSlug;
    } else if (params.customExamples) {
      const pairs = params.customExamples.split(',').map(s => s.trim()).filter(Boolean);
      body.examples = pairs.map(p => {
        const [inp, out] = p.split('->').map(s => s.trim());
        return { inputs: { x: Number(inp) }, output: Number(out) };
      });
    }

    try {
      const res = await fetch('/synthesize/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setActiveParams(params);
      setCompareMode(true, data.run_ids);

      // Start all three runs in store
      Object.entries(data.run_ids).forEach(([engine, rid]) => {
        useRunStore.getState().startRun(rid as string, engine as EngineType);
      });
      setWsTrigger(t => t + 1);
    } catch (err) {
      console.error(err);
    }
  };

  const handleReset = () => {
    reset();
    setSelectedProblem(null);
    setActiveParams(null);
  };

  // Determine current active display state
  const hasStarted = activeRunId || (compareMode && compareRunIds) || selectedProblem;

  return (
    <div className="app-shell">
      {/* Top Navbar */}
      <nav className="top-nav">
        <div className="nav-logo" style={{ cursor: 'pointer' }} onClick={handleReset}>GENESIS</div>
        <div className="nav-actions">
          <button className="btn btn-ghost btn-sm" onClick={() => { setWhatAmI(true); setGlossary(false); }}>
            💡 What is this?
          </button>
          <button className="btn btn-ghost btn-sm" onClick={() => { setGlossary(true); setWhatAmI(false); }}>
            📚 Glossary
          </button>
          {hasStarted && (
            <button className="btn btn-secondary btn-sm" onClick={handleReset}>
              ✕ Reset
            </button>
          )}
        </div>
      </nav>

      {/* Main Workspace content */}
      <main style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {!selectedProblem && !compareMode ? (
          <EmptyState onSelectProblem={setSelectedProblem} selectedProblem={selectedProblem} />
        ) : compareMode && compareRunIds ? (
          /* Comparison Mode Layout */
          <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '16px', gap: '16px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', flex: 1 }}>
              {(['genesis', 'pure_evolutionary', 'random'] as const).map(engine => {
                const rid = compareRunIds[engine];
                const rState = runs[rid];
                return (
                  <ComparisonPane
                    key={engine}
                    engineType={engine}
                    events={rState?.events ?? []}
                    latestEvent={rState?.latestEvent ?? null}
                    status={rState?.status ?? 'idle'}
                  />
                );
              })}
            </div>
            {/* Show summary when all done or any solved */}
            {Object.values(compareRunIds).some(rid => runs[rid]?.result) && (
              <div style={{ maxWidth: 800, width: '100%', margin: '0 auto' }}>
                {Object.values(compareRunIds)
                  .map(rid => runs[rid]?.result)
                  .filter(Boolean)
                  .sort((a, b) => (b?.best_fitness ?? 0) - (a?.best_fitness ?? 0))
                  .slice(0, 1)
                  .map(res => (
                    <ResultSummary key={res!.run_id} result={res!} generations={res!.generations_taken} />
                  ))}
              </div>
            )}
          </div>
        ) : (
          /* Single Run Layout */
          <div className="workspace-running">
            <div className="sidebar">
              <SynthesisForm
                selectedProblem={selectedProblem}
                onRun={handleRun}
                onCompare={handleCompare}
                running={isRunning ?? false}
              />
              {activeRun?.result && (
                <ResultSummary result={activeRun.result} generations={activeRun.events.length} />
              )}
            </div>

            {/* Main Viz Panes */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 400px', gap: '16px', height: '100%', overflow: 'hidden' }}>
              <TreeView3D bestIndividual={activeRun?.latestEvent?.best_individual ?? null} />
              <PopulationChart
                events={activeRun?.events ?? []}
                latestEvent={activeRun?.latestEvent ?? null}
              />
            </div>
          </div>
        )}
      </main>

      {/* Floating Guidance/Glossary overlays */}
      <GuidanceTour />
      <GlossaryPanel />
      <WhatAmILooking />
    </div>
  );
}
