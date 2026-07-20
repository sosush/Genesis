import React, { useEffect, useRef, useCallback } from 'react';
import { GenerationEvent, EngineType } from '../types';

interface ComparisonPaneProps {
  engineType: EngineType;
  events: GenerationEvent[];
  latestEvent: GenerationEvent | null;
  status: 'idle' | 'running' | 'done' | 'error';
}

const ENGINE_LABELS: Record<EngineType, string> = {
  genesis:           'Genesis',
  pure_evolutionary: 'Pure Evolutionary',
  random:            'Random Search',
};

const ENGINE_COLORS: Record<EngineType, string> = {
  genesis:           '#00f5d4',
  pure_evolutionary: '#f9c74f',
  random:            '#9b5de5',
};

const ENGINE_DESCRIPTIONS: Record<EngineType, string> = {
  genesis:           'Genetic programming + neural pre-filter predicts which programs are worth testing.',
  pure_evolutionary: 'Genetic programming only — no neural shortcut. Evolves but evaluates everything.',
  random:            'No evolution. Fresh random population every generation — baseline noise floor.',
};

const W = 320;
const H = 110;
const PAD = { top: 10, right: 10, bottom: 20, left: 30 };

export function ComparisonPane({ engineType, events, latestEvent, status }: ComparisonPaneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const color = ENGINE_COLORS[engineType];

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const curve = events[events.length - 1]?.fitness_curve ?? [];
    if (curve.length < 2) {
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.font = '11px Space Grotesk, sans-serif';
      ctx.fillText('Waiting…', W / 2 - 28, H / 2);
      return;
    }

    const totalGens = curve.length;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (const f of [0, 0.5, 1]) {
      const y = PAD.top + (1 - f) * (H - PAD.top - PAD.bottom);
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.18)';
      ctx.font = '8px JetBrains Mono, monospace';
      ctx.fillText(f.toFixed(1), 2, y + 3);
    }

    // Gradient fill
    const grad = ctx.createLinearGradient(0, PAD.top, 0, H - PAD.bottom);
    grad.addColorStop(0, color + '44');
    grad.addColorStop(1, color + '00');
    ctx.beginPath();
    const xScale = (W - PAD.left - PAD.right) / Math.max(1, totalGens - 1);
    ctx.moveTo(PAD.left, H - PAD.bottom);
    curve.forEach((f, g) => {
      const x = PAD.left + g * xScale;
      const y = PAD.top + (1 - f) * (H - PAD.top - PAD.bottom);
      ctx.lineTo(x, y);
    });
    ctx.lineTo(PAD.left + (totalGens - 1) * xScale, H - PAD.bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    curve.forEach((f, g) => {
      const x = PAD.left + g * xScale;
      const y = PAD.top + (1 - f) * (H - PAD.top - PAD.bottom);
      g === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Current point
    if (curve.length > 0) {
      const lastF = curve[curve.length - 1];
      const lx = PAD.left + (curve.length - 1) * xScale;
      const ly = PAD.top + (1 - lastF) * (H - PAD.top - PAD.bottom);
      ctx.beginPath();
      ctx.arc(lx, ly, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }, [events, color]);

  useEffect(() => { draw(); }, [draw]);

  const bestFitness = latestEvent?.best_individual.fitness ?? 0;
  const genCount    = latestEvent ? latestEvent.generation + 1 : 0;
  const isSolved    = latestEvent?.event_type === 'solved';

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: 'var(--bg-deep)', border: `1px solid ${color}22`,
      borderRadius: 'var(--radius-lg)', overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{ padding: '12px 16px 8px', borderBottom: `1px solid ${color}18`, display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 700, color }}>{ENGINE_LABELS[engineType]}</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            {status === 'running' && (
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: color, animation: 'pulse 1.5s infinite', display: 'inline-block' }} />
            )}
            {isSolved && <span style={{ fontSize: '0.72rem', color: '#43aa8b', fontWeight: 600 }}>✓ SOLVED</span>}
            {status === 'done' && !isSolved && <span style={{ fontSize: '0.72rem', color: '#4a5568' }}>TIMEOUT</span>}
          </span>
        </div>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
          {ENGINE_DESCRIPTIONS[engineType]}
        </span>
      </div>

      {/* Stats row */}
      <div style={{ display: 'flex', gap: 16, padding: '8px 16px', borderBottom: `1px solid ${color}10` }}>
        <div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Generation</div>
          <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>{genCount}</div>
        </div>
        <div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Best Fitness</div>
          <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '1.1rem', fontWeight: 700, color }}>{bestFitness.toFixed(3)}</div>
        </div>
        {latestEvent?.neural_scorer_active && (
          <div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Pruned</div>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '1.1rem', fontWeight: 700, color: '#4a5568' }}>{latestEvent.pruned_count}</div>
          </div>
        )}
      </div>

      {/* Fitness curve */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '8px' }}>
        <canvas ref={canvasRef} style={{ display: 'block', borderRadius: 6 }} />
      </div>

      {/* Best program */}
      {latestEvent && (
        <div style={{ padding: '6px 12px 10px', borderTop: `1px solid ${color}10` }}>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 3 }}>Best program</div>
          <pre style={{
            fontFamily: 'JetBrains Mono, monospace', fontSize: '0.72rem',
            color, background: 'rgba(0,0,0,0.25)', borderRadius: 5,
            padding: '5px 8px', overflow: 'auto', maxHeight: 60,
            whiteSpace: 'pre-wrap', wordBreak: 'break-all',
          }}>{latestEvent.best_individual.expr}</pre>
        </div>
      )}

      {/* Caption */}
      <div className="panel-caption" style={{ borderColor: `${color}15` }}>
        Same problem, three strategies — watch which one wins.
      </div>
    </div>
  );
}
