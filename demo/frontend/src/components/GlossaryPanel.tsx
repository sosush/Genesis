import React from 'react';
import { useGuidanceStore } from '../store/guidanceStore';

export function GlossaryPanel() {
  const { glossaryOpen, setGlossary } = useGuidanceStore();

  if (!glossaryOpen) return null;

  return (
    <div className="glass-hi animate-in" style={{
      position: 'fixed',
      top: 64,
      right: 24,
      width: 340,
      maxHeight: 'calc(100vh - 100px)',
      overflowY: 'auto',
      zIndex: 400,
      padding: '20px 24px',
      border: 'var(--border-subtle)',
      boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h3 style={{ color: 'var(--accent-cyan)' }}>Research Glossary</h3>
        <button onClick={() => setGlossary(false)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.2rem' }}>✕</button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, fontSize: '0.8rem', lineHeight: 1.5 }}>
        <div>
          <h4 style={{ color: 'var(--text-primary)', marginBottom: 4 }}>Program Synthesis</h4>
          <p style={{ margin: 0 }}>Automatically generating computer code that satisfies a set of constraints (e.g. input-output examples) without human intervention.</p>
        </div>

        <div>
          <h4 style={{ color: 'var(--text-primary)', marginBottom: 4 }}>Symbolic Evolution</h4>
          <p style={{ margin: 0 }}>Exploring the infinite space of Abstract Syntax Trees (ASTs) using evolutionary algorithms like crossover and mutation to breed functional programs.</p>
        </div>

        <div>
          <h4 style={{ color: 'var(--text-primary)', marginBottom: 4 }}>Neural Pre-filtering</h4>
          <p style={{ margin: 0 }}>Using a PyTorch MLP to predict if a program will pass test cases before running it. Prunes the worst candidates to speed up convergence by 4x.</p>
        </div>

        <div>
          <h4 style={{ color: 'var(--text-primary)', marginBottom: 4 }}>Fitness Score</h4>
          <p style={{ margin: 0 }}>A value in [0.0, 1.0] representing correctness. 1.0 means all test cases passed perfectly. Incorporates parsimony pressure to penalize bloated programs.</p>
        </div>

        <div>
          <h4 style={{ color: 'var(--text-primary)', marginBottom: 4 }}>Crossover / Mutation</h4>
          <p style={{ margin: 0 }}>Crossover swaps logic subtrees between parents. Mutation perturbs operators, variables, or subtree components randomly to seek novel programs.</p>
        </div>
      </div>
    </div>
  );
}
