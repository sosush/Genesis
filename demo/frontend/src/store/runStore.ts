import { create } from 'zustand';
import { GenerationEvent, RunInfo, RunResult, EngineType } from '../types';

interface RunStore {
  // Active single run
  activeRunId: string | null;
  runs: Record<string, RunInfo>;

  // Comparison mode
  compareMode: boolean;
  compareRunIds: Record<EngineType, string> | null;

  // Selected individual (for node-click side panel)
  selectedIndividualId: string | null;

  // Actions
  startRun: (runId: string, engineType: EngineType) => void;
  pushEvent: (runId: string, event: GenerationEvent) => void;
  finishRun: (runId: string, result: RunResult) => void;
  errorRun: (runId: string, error: string) => void;
  setCompareMode: (mode: boolean, runIds?: Record<EngineType, string>) => void;
  selectIndividual: (id: string | null) => void;
  reset: () => void;
}

const EMPTY_RUN = (runId: string, engineType: EngineType): RunInfo => ({
  runId,
  engineType,
  events: [],
  latestEvent: null,
  result: null,
  status: 'running',
  error: null,
});

export const useRunStore = create<RunStore>((set, get) => ({
  activeRunId: null,
  runs: {},
  compareMode: false,
  compareRunIds: null,
  selectedIndividualId: null,

  startRun: (runId, engineType) => {
    set(state => ({
      activeRunId: state.compareMode ? state.activeRunId : runId,
      runs: { ...state.runs, [runId]: EMPTY_RUN(runId, engineType) },
    }));
  },

  pushEvent: (runId, event) => {
    set(state => {
      const run = state.runs[runId];
      if (!run) return state;
      // Keep last 200 events to avoid unbounded growth
      const events = [...run.events, event].slice(-200);
      return {
        runs: {
          ...state.runs,
          [runId]: { ...run, events, latestEvent: event, status: 'running' },
        },
      };
    });
  },

  finishRun: (runId, result) => {
    set(state => {
      const run = state.runs[runId];
      if (!run) return state;
      return {
        runs: { ...state.runs, [runId]: { ...run, result, status: 'done' } },
      };
    });
  },

  errorRun: (runId, error) => {
    set(state => {
      const run = state.runs[runId];
      if (!run) return state;
      return {
        runs: { ...state.runs, [runId]: { ...run, error, status: 'error' } },
      };
    });
  },

  setCompareMode: (mode, runIds) => {
    set({ compareMode: mode, compareRunIds: runIds ?? null });
  },

  selectIndividual: (id) => set({ selectedIndividualId: id }),

  reset: () => set({ activeRunId: null, runs: {}, compareMode: false, compareRunIds: null, selectedIndividualId: null }),
}));
