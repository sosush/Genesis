import { create } from 'zustand';
import { GenerationEvent } from '../types';

export type TourStep =
  | 'gen1_start'
  | 'scorer_active'
  | 'pruned'
  | 'solved'
  | null;

interface GuidanceStore {
  hasSeenTour: boolean;
  tourActive: boolean;
  currentStep: TourStep;
  glossaryOpen: boolean;
  whatAmIOpen: boolean;

  // Actions
  startTour: () => void;
  endTour: () => void;
  advanceStep: (step: TourStep) => void;
  setGlossary: (open: boolean) => void;
  setWhatAmI: (open: boolean) => void;
  processEvent: (event: GenerationEvent) => void;
}

function getStoredTour(): boolean {
  try { return localStorage.getItem('genesis_has_seen_tour') === 'true'; }
  catch { return false; }
}

function storeTourSeen() {
  try { localStorage.setItem('genesis_has_seen_tour', 'true'); }
  catch { /* private browsing */ }
}

export const useGuidanceStore = create<GuidanceStore>((set, get) => ({
  hasSeenTour: getStoredTour(),
  tourActive: false,
  currentStep: null,
  glossaryOpen: false,
  whatAmIOpen: false,

  startTour: () => set({ tourActive: true, currentStep: 'gen1_start' }),

  endTour: () => {
    storeTourSeen();
    set({ tourActive: false, currentStep: null, hasSeenTour: true });
  },

  advanceStep: (step) => set({ currentStep: step }),

  setGlossary: (open) => set({ glossaryOpen: open }),
  setWhatAmI:  (open) => set({ whatAmIOpen: open }),

  processEvent: (event) => {
    const { tourActive, currentStep } = get();
    if (!tourActive) return;

    if (event.generation === 0 && currentStep === 'gen1_start') {
      // Already showing gen1 callout; advance when scorer activates
    }
    if (event.neural_scorer_active && currentStep === 'gen1_start') {
      set({ currentStep: 'scorer_active' });
    }
    if (event.pruned_count > 0 && currentStep === 'scorer_active') {
      set({ currentStep: 'pruned' });
    }
    if (event.event_type === 'solved' && currentStep !== 'solved') {
      set({ currentStep: 'solved' });
    }
  },
}));
