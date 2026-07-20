import React from 'react';
import { RunResult } from '../types';

interface ResultSummaryProps {
  result: RunResult;
  generations: number;
}

export function ResultSummary({ result, generations }: ResultSummaryProps) {
  const isGenesis = result.engine_type === 'genesis';

  // Construct a clear "so what" summary statement
  let explanation = '';
  if (result.solved) {
    if (isGenesis) {
      explanation = `Genesis successfully evolved the solution \`${result.program}\` in only ${result.generations_taken} generations. By pre-filtering candidates with a neural network, it avoided executing hundreds of broken syntax trees.`;
    } else if (result.engine_type === 'pure_evolutionary') {
      explanation = `The Pure Evolutionary engine successfully evolved \`${result.program}\` in ${result.generations_taken} generations. It succeeded, but had to symbolically execute every candidate program in a sandbox, taking more overall computational effort.`;
    } else {
      explanation = `Random search found a working candidate \`${result.program}\` by sheer chance in generation ${result.generations_taken}.`;
    }
  } else {
    explanation = `The run completed without finding a perfect program matching all test cases. The best attempt reached a fitness score of ${(result.best_fitness).toFixed(3)}. Try increasing the maximum generations or simplifying the target rules.`;
  }

  return (
    <div className="result-card animate-pop" style={{ width: '100%', border: 'var(--border-accent)', background: 'rgba(0, 245, 212, 0.03)' }}>
      <div className="headline" style={{ color: result.solved ? 'var(--accent-cyan)' : 'var(--accent-amber)' }}>
        {result.solved ? '🎉 Synthesis Successful!' : '⌛ Run Completed'}
      </div>
      <p className="subline" style={{ fontSize: '0.85rem', margin: '0 0 14px', lineHeight: 1.5 }}>
        {explanation}
      </p>
      {result.solved && (
        <div style={{ display: 'inline-flex', flexDirection: 'column', gap: 6, alignItems: 'center' }}>
          <span style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-secondary)' }}>Synthesized Algorithm</span>
          <code style={{ fontSize: '1rem', padding: '8px 16px', background: 'var(--bg-surface)', border: 'var(--border-subtle)', borderRadius: 8 }}>
            {result.program}
          </code>
        </div>
      )}
    </div>
  );
}
