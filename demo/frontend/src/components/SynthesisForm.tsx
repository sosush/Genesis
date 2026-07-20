import React, { useState } from 'react';
import { Problem, EngineType } from '../types';

interface SynthesisFormProps {
  selectedProblem: Problem | null;
  onRun: (params: RunParams) => void;
  onCompare: (params: RunParams) => void;
  running: boolean;
}

export interface RunParams {
  problemSlug: string | null;
  customExamples: string | null;
  maxGenerations: number;
  popSize: number;
  engineType: EngineType;
  seed: number | null;
}

function parseExamples(raw: string): { inputs: Record<string, number>; output: number }[] | null {
  try {
    const pairs = raw.split(',').map(s => s.trim()).filter(Boolean);
    return pairs.map(p => {
      const [inp, out] = p.split('->').map(s => s.trim());
      const num = Number(inp);
      if (isNaN(num) || isNaN(Number(out))) throw new Error('invalid');
      return { inputs: { x: num }, output: Number(out) };
    });
  } catch {
    return null;
  }
}

export function SynthesisForm({ selectedProblem, onRun, onCompare, running }: SynthesisFormProps) {
  const [mode, setMode]           = useState<'preset' | 'custom'>('preset');
  const [customExamples, setCustom] = useState('1->1, 2->4, 3->9, 4->16');
  const [maxGen, setMaxGen]       = useState(100);
  const [popSize, setPopSize]     = useState(80);
  const [engineType, setEngine]   = useState<EngineType>('genesis');
  const [useSeed, setUseSeed]     = useState(false);
  const [seed, setSeed]           = useState(42);
  const [parseError, setParseError] = useState(false);

  const validate = (): RunParams | null => {
    if (mode === 'custom') {
      const parsed = parseExamples(customExamples);
      if (!parsed) { setParseError(true); return null; }
      setParseError(false);
      return {
        problemSlug: null,
        customExamples: customExamples,
        maxGenerations: maxGen,
        popSize,
        engineType,
        seed: useSeed ? seed : null,
      };
    }
    if (!selectedProblem) return null;
    return {
      problemSlug: selectedProblem.slug,
      customExamples: null,
      maxGenerations: maxGen,
      popSize,
      engineType,
      seed: useSeed ? seed : null,
    };
  };

  const handleRun = () => {
    const params = validate();
    if (params) onRun(params);
  };

  const handleCompare = () => {
    const params = validate();
    if (params) onCompare({ ...params, popSize: Math.min(popSize, 60) });
  };

  return (
    <div className="glass" style={{ padding: 'var(--space-5)', display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <h4>Configuration</h4>

      {/* Mode toggle */}
      <div style={{ display: 'flex', gap: 6, background: 'var(--bg-surface)', borderRadius: 8, padding: 4 }}>
        {(['preset', 'custom'] as const).map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            style={{
              flex: 1, padding: '6px 12px', borderRadius: 6, border: 'none',
              background: mode === m ? 'var(--bg-elevated)' : 'transparent',
              color: mode === m ? 'var(--text-primary)' : 'var(--text-muted)',
              cursor: 'pointer', fontFamily: 'var(--font-ui)', fontSize: '0.82rem',
              fontWeight: mode === m ? 600 : 400,
              transition: 'all 150ms ease',
            }}
          >
            {m === 'preset' ? 'Use Preset' : 'Custom I/O'}
          </button>
        ))}
      </div>

      {/* Preset / custom input */}
      {mode === 'preset' ? (
        <div>
          {selectedProblem ? (
            <div style={{ padding: '10px 12px', background: 'rgba(0,245,212,0.05)', border: '1px solid rgba(0,245,212,0.2)', borderRadius: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>{selectedProblem.name}</span>
                <span className={`badge badge-${selectedProblem.difficulty}`}>{selectedProblem.difficulty}</span>
              </div>
              <p style={{ fontSize: '0.76rem', marginTop: 5, color: 'var(--text-muted)', margin: '5px 0 0' }}>{selectedProblem.description}</p>
              <code style={{ display: 'block', marginTop: 6, fontSize: '0.72rem' }}>hint: {selectedProblem.hint}</code>
            </div>
          ) : (
            <div style={{ padding: '10px 12px', background: 'var(--bg-surface)', borderRadius: 8, fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              ← Select a problem from the gallery above
            </div>
          )}
        </div>
      ) : (
        <div>
          <label style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'block', marginBottom: 5 }}>
            Input → Output pairs (variable x, format: x_val→output, ...)
          </label>
          <textarea
            value={customExamples}
            onChange={e => { setCustom(e.target.value); setParseError(false); }}
            rows={3}
            placeholder="1->1, 2->4, 3->9, 4->16"
            style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.85rem', borderColor: parseError ? 'rgba(249,65,68,0.5)' : undefined }}
          />
          {parseError && <div style={{ fontSize: '0.72rem', color: 'var(--accent-coral)', marginTop: 4 }}>⚠ Invalid format. Use: 1-&gt;1, 2-&gt;4, 3-&gt;9</div>}
        </div>
      )}

      {/* Params */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <div>
          <label style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>Max Generations</label>
          <input type="number" min={10} max={500} step={10} value={maxGen} onChange={e => setMaxGen(+e.target.value)} />
        </div>
        <div>
          <label style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>Population Size</label>
          <input type="number" min={20} max={200} step={10} value={popSize} onChange={e => setPopSize(+e.target.value)} />
        </div>
      </div>

      {/* Engine selector */}
      <div>
        <label style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>Engine (single run)</label>
        <select value={engineType} onChange={e => setEngine(e.target.value as EngineType)}>
          <option value="genesis">Genesis (Neuro-Symbolic)</option>
          <option value="pure_evolutionary">Pure Evolutionary</option>
          <option value="random">Random Search</option>
        </select>
      </div>

      {/* Seed */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <input type="checkbox" id="use-seed" checked={useSeed} onChange={e => setUseSeed(e.target.checked)} style={{ width: 'auto', cursor: 'pointer' }} />
        <label htmlFor="use-seed" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>Fix random seed</label>
        {useSeed && (
          <input type="number" value={seed} onChange={e => setSeed(+e.target.value)} style={{ width: 70 }} />
        )}
      </div>

      {/* Action buttons */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 4 }}>
        <button
          id="btn-synthesize"
          className="btn btn-primary"
          onClick={handleRun}
          disabled={running || (mode === 'preset' && !selectedProblem)}
          style={{ opacity: (running || (mode === 'preset' && !selectedProblem)) ? 0.5 : 1 }}
        >
          {running ? '⟳ Evolving…' : '🚀 Synthesize Program'}
        </button>
        <button
          id="btn-compare"
          className="btn btn-secondary"
          onClick={handleCompare}
          disabled={running || (mode === 'preset' && !selectedProblem)}
          style={{ opacity: (running || (mode === 'preset' && !selectedProblem)) ? 0.5 : 1, fontSize: '0.82rem' }}
        >
          ⚡ Compare All Three Engines
        </button>
      </div>
    </div>
  );
}
