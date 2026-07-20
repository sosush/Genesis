import React from 'react';
import { useGuidanceStore } from '../store/guidanceStore';

export function GuidanceTour() {
  const { tourActive, currentStep, endTour } = useGuidanceStore();

  if (!tourActive || !currentStep) return null;

  const content: Record<string, { title: string; text: string; next?: string }> = {
    gen1_start: {
      title: "Generation 1: Seeding Life",
      text: "The engine initialized a population of 100 completely random candidate programs. None of them fit your examples yet, but selection will begin shortly.",
    },
    scorer_active: {
      title: "Neural Scorer Online",
      text: "Genesis has gathered enough data! A PyTorch Multi-Layer Perceptron is now predicting which programs look promising before running them.",
    },
    pruned: {
      title: "Pruning the Unfit",
      text: "Watch the gray dots. The neural scorer has pruned the bottom 50% of candidates. This shortcut speeds up evolution by about 4x.",
    },
    solved: {
      title: "Fitness 1.0 reached — Solved!",
      text: "Darwinian evolution has successfully synthesized a program that perfectly matches all input/output examples. Click the result card to see the final expression.",
    },
  };

  const step = content[currentStep];
  if (!step) return null;

  return (
    <div className="glass-hi animate-pop" style={{
      position: 'fixed',
      bottom: 24,
      right: 24,
      zIndex: 500,
      width: 320,
      padding: '16px 20px',
      border: '1px solid rgba(0,245,212,0.3)',
      boxShadow: '0 10px 30px rgba(0,245,212,0.15)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span className="callout-dot" />
        <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>{step.title}</h3>
      </div>
      <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.5, margin: '0 0 12px' }}>
        {step.text}
      </p>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
        <button className="btn btn-primary btn-sm" onClick={endTour}>
          {currentStep === 'solved' ? 'Finish' : 'Got it'}
        </button>
      </div>
    </div>
  );
}
