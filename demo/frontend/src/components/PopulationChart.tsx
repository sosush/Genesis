import React, { useEffect, useRef, useCallback } from 'react';
import { GenerationEvent } from '../types';

interface PopulationChartProps {
  events: GenerationEvent[];
  latestEvent: GenerationEvent | null;
  engineType?: string;
}

const COLORS = {
  survived: '#00bbf9',
  pruned:   '#334155',
  elite:    '#f9c74f',
  best:     '#00f5d4',
  curve:    '#00f5d4',
};

const W = 380;
const H_SCATTER = 160;
const H_CURVE   = 80;
const PAD = { top: 12, right: 12, bottom: 24, left: 36 };

function fitnessToY(f: number, height: number): number {
  return PAD.top + (1 - f) * (height - PAD.top - PAD.bottom);
}

function genToX(gen: number, totalGens: number, width: number): number {
  if (totalGens <= 1) return PAD.left + (width - PAD.left - PAD.right) / 2;
  return PAD.left + (gen / (totalGens - 1)) * (width - PAD.left - PAD.right);
}

export function PopulationChart({ events, latestEvent }: PopulationChartProps) {
  const scatterRef = useRef<HTMLCanvasElement>(null);
  const curveRef   = useRef<HTMLCanvasElement>(null);

  const drawScatter = useCallback(() => {
    const canvas = scatterRef.current;
    if (!canvas || !latestEvent) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width  = W * dpr;
    canvas.height = H_SCATTER * dpr;
    canvas.style.width  = `${W}px`;
    canvas.style.height = `${H_SCATTER}px`;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H_SCATTER);

    // Background grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let f = 0; f <= 1; f += 0.25) {
      const y = fitnessToY(f, H_SCATTER);
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.2)';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.fillText(f.toFixed(1), 2, y + 3);
    }

    // Axis label
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.font = '9px Space Grotesk, sans-serif';
    ctx.fillText('fitness', PAD.left, PAD.top - 2);

    // Histogram bins — use as column bar chart background
    const hist = latestEvent.fitness_histogram;
    const maxCount = Math.max(...hist, 1);
    const binW = (W - PAD.left - PAD.right) / hist.length;
    hist.forEach((count, i) => {
      const barH = ((count / maxCount) * (H_SCATTER - PAD.top - PAD.bottom)) * 0.9;
      const x = PAD.left + i * binW;
      const y = H_SCATTER - PAD.bottom - barH;
      ctx.fillStyle = 'rgba(0,187,249,0.08)';
      ctx.fillRect(x, y, binW - 1, barH);
    });

    // Individual dots — scatter by rank within fitness band
    const pops = latestEvent.population_snapshot;
    const totalPop = pops.length;

    pops.forEach((ind, rankIdx) => {
      const x = PAD.left + (W - PAD.left - PAD.right) * ind.fitness +
                (Math.sin(rankIdx * 7.3) * 6); // x-jitter within fitness column
      const y = fitnessToY(ind.fitness, H_SCATTER) + (Math.cos(rankIdx * 5.1) * 4);

      const r = ind.rank === 0 ? 5 : 2.5;
      let color = ind.is_pruned_by_scorer ? COLORS.pruned : COLORS.survived;
      if (ind.rank === 0) color = COLORS.best;
      else if (ind.rank <= 2) color = COLORS.elite;

      ctx.beginPath();
      ctx.arc(Math.max(PAD.left + 4, Math.min(W - PAD.right - 4, x)), Math.max(PAD.top + 4, Math.min(H_SCATTER - PAD.bottom - 4, y)), r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = ind.is_pruned_by_scorer ? 0.3 : 0.85;
      ctx.fill();
      ctx.globalAlpha = 1;
    });

    // Best individual highlight ring
    const best = latestEvent.best_individual;
    const bx = PAD.left + (W - PAD.left - PAD.right) * best.fitness;
    const by = fitnessToY(best.fitness, H_SCATTER);
    ctx.beginPath();
    ctx.arc(Math.min(bx, W - PAD.right - 6), by, 7, 0, Math.PI * 2);
    ctx.strokeStyle = COLORS.best;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Pruned count label
    if (latestEvent.pruned_count > 0) {
      ctx.fillStyle = '#4a5568';
      ctx.font = '9px Space Grotesk, sans-serif';
      ctx.fillText(`${latestEvent.pruned_count} cut by neural scorer`, PAD.left, H_SCATTER - 6);
    }
  }, [latestEvent]);

  const drawCurve = useCallback(() => {
    const canvas = curveRef.current;
    if (!canvas || events.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width  = W * dpr;
    canvas.height = H_CURVE * dpr;
    canvas.style.width  = `${W}px`;
    canvas.style.height = `${H_CURVE}px`;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H_CURVE);

    const curve = events[events.length - 1]?.fitness_curve ?? [];
    if (curve.length < 2) return;

    const totalGens = curve.length;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let f of [0, 0.5, 1]) {
      const y = fitnessToY(f, H_CURVE);
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
    }

    // Fill gradient under curve
    const grad = ctx.createLinearGradient(0, PAD.top, 0, H_CURVE - PAD.bottom);
    grad.addColorStop(0, 'rgba(0,245,212,0.25)');
    grad.addColorStop(1, 'rgba(0,245,212,0)');
    ctx.beginPath();
    ctx.moveTo(genToX(0, totalGens, W), H_CURVE - PAD.bottom);
    curve.forEach((f, g) => {
      ctx.lineTo(genToX(g, totalGens, W), fitnessToY(f, H_CURVE));
    });
    ctx.lineTo(genToX(totalGens - 1, totalGens, W), H_CURVE - PAD.bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    curve.forEach((f, g) => {
      const x = genToX(g, totalGens, W);
      const y = fitnessToY(f, H_CURVE);
      g === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = COLORS.curve;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Current point
    const lastF = curve[curve.length - 1];
    ctx.beginPath();
    ctx.arc(genToX(curve.length - 1, totalGens, W), fitnessToY(lastF, H_CURVE), 4, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.curve;
    ctx.fill();

    // Gen label
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = '9px Space Grotesk, sans-serif';
    ctx.fillText(`gen ${curve.length - 1}`, W - PAD.right - 28, H_CURVE - 6);

  }, [events]);

  useEffect(() => { drawScatter(); }, [drawScatter]);
  useEffect(() => { drawCurve(); }, [drawCurve]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0, height: '100%', background: 'var(--bg-deep)', borderRadius: 'var(--radius-lg)', overflow: 'hidden', border: 'var(--border-dim)' }}>
      {/* Header */}
      <div style={{ padding: '10px 16px 6px', borderBottom: 'var(--border-dim)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>Population</span>
        {latestEvent && (
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
            gen {latestEvent.generation} · best {latestEvent.best_individual.fitness.toFixed(3)}
          </span>
        )}
      </div>

      {/* Scatter */}
      <div style={{ padding: '8px 8px 0', flexShrink: 0 }}>
        <canvas ref={scatterRef} style={{ display: 'block', borderRadius: 6 }} />
      </div>

      {/* Legend row */}
      <div style={{ padding: '4px 16px', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {[
          { color: COLORS.best,    label: 'best' },
          { color: COLORS.elite,   label: 'elite' },
          { color: COLORS.survived,label: 'survived' },
          { color: COLORS.pruned,  label: 'cut by scorer' },
        ].map(({ color, label }) => (
          <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: color, display: 'inline-block' }} />
            {label}
          </span>
        ))}
      </div>

      {/* Divider */}
      <div style={{ height: 1, background: 'rgba(255,255,255,0.05)', margin: '4px 0' }} />

      {/* Fitness curve */}
      <div style={{ padding: '2px 8px 6px', flexShrink: 0 }}>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', padding: '0 8px 2px', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
          Best fitness over time
        </div>
        <canvas ref={curveRef} style={{ display: 'block', borderRadius: 6 }} />
      </div>

      {/* Panel caption */}
      <div className="panel-caption">
        Every candidate this generation — green survived, gray was cut by the neural scorer.
      </div>
    </div>
  );
}
