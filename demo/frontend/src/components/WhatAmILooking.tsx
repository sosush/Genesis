import React from 'react';
import { useGuidanceStore } from '../store/guidanceStore';

export function WhatAmILooking() {
  const { whatAmIOpen, setWhatAmI } = useGuidanceStore();

  if (!whatAmIOpen) return null;

  return (
    <div className="glass-hi animate-in" style={{
      position: 'fixed',
      top: 64,
      right: 24,
      width: 380,
      maxHeight: 'calc(100vh - 100px)',
      overflowY: 'auto',
      zIndex: 400,
      padding: '24px',
      border: 'var(--border-subtle)',
      boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h3 style={{ color: 'var(--accent-cyan)' }}>What Am I Looking At?</h3>
        <button onClick={() => setWhatAmI(false)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.2rem' }}>✕</button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, fontSize: '0.82rem', lineHeight: 1.5 }}>
        <p style={{ margin: 0 }}>
          This dashboard visualizes a <strong>Neuro-Symbolic program synthesis engine</strong> working in real time.
        </p>

        <div>
          <h4 style={{ color: 'var(--text-primary)', marginBottom: 4 }}>1. The 3D Tree (Centerpiece)</h4>
          <p style={{ margin: 0 }}>
            Visualizes the <strong>Abstract Syntax Tree (AST)</strong> of the current best program. Nodes represent variables (yellow), constants (purple), operators (blue), and conditions (green). Clicking nodes exposes their formula slice.
          </p>
        </div>

        <div>
          <h4 style={{ color: 'var(--text-primary)', marginBottom: 4 }}>2. The Population overview</h4>
          <p style={{ margin: 0 }}>
            Every dot represents an individual program in this generation. Green ones survived evaluation; gray ones were safely skipped because the neural pre-filter predicted they would fail.
          </p>
        </div>

        <div>
          <h4 style={{ color: 'var(--text-primary)', marginBottom: 4 }}>3. The Comparison mode</h4>
          <p style={{ margin: 0 }}>
            Launches 3 parallel runs with the same test configuration to prove how much faster Genesis (neural pre-filtering) converges compared to pure evolution or random search.
          </p>
        </div>
      </div>
    </div>
  );
}
