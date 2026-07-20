import React from 'react';
import { ProblemGallery } from './ProblemGallery';
import { Problem } from '../types';

interface EmptyStateProps {
  onSelectProblem: (problem: Problem) => void;
  selectedProblem: Problem | null;
}

export function EmptyState({ onSelectProblem, selectedProblem }: EmptyStateProps) {
  return (
    <div className="workspace-idle animate-in">
      <div style={{ textAlign: 'center', maxWidth: 650, margin: '0 auto' }}>
        <h1 className="gradient-text" style={{ marginBottom: 12 }}>GENESIS</h1>
        <h3 style={{ fontWeight: 400, color: 'var(--text-secondary)', lineHeight: 1.5, margin: '0 0 20px' }}>
          Autonomous Neuro-Symbolic Program Synthesis
        </h3>
        <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
          Genesis breeds, mutates, and evolves functional Python code from examples using Darwinian selection.
          An integrated PyTorch Neural Scorer predicts program viability, speeding up evolutionary convergence.
        </p>
      </div>

      <div className="glass" style={{ width: '100%', padding: '24px 30px' }}>
        <h3 style={{ marginBottom: 12, borderBottom: 'var(--border-dim)', paddingBottom: 8, fontSize: '0.92rem', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--text-secondary)' }}>
          Select a problem to begin
        </h3>
        <ProblemGallery onSelect={onSelectProblem} />
      </div>
    </div>
  );
}
