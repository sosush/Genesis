import React, { useState, useEffect } from 'react';
import { Problem } from '../types';

interface ProblemGalleryProps {
  onSelect: (problem: Problem) => void;
}

const CATEGORY_ORDER = ['benchmark', 'classic', 'interview', 'math'];

function DifficultyBadge({ diff }: { diff: string }) {
  return (
    <span className={`badge badge-${diff}`}>{diff}</span>
  );
}

function CategoryLabel({ cat }: { cat: string }) {
  const labels: Record<string, string> = {
    benchmark: '📊 Benchmark',
    classic:   '⚡ Classic',
    interview: '🎯 Interview',
    math:      '∑ Math',
  };
  return (
    <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
      {labels[cat] ?? cat}
    </span>
  );
}

export function ProblemGallery({ onSelect }: ProblemGalleryProps) {
  const [problems, setProblems] = useState<Problem[]>([]);
  const [loading, setLoading]   = useState(true);
  const [hoverId, setHoverId]   = useState<string | null>(null);

  useEffect(() => {
    fetch('/problems')
      .then(r => r.json())
      .then((data: Problem[]) => { setProblems(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const grouped = CATEGORY_ORDER.reduce((acc, cat) => {
    acc[cat] = problems.filter(p => p.category === cat);
    return acc;
  }, {} as Record<string, Problem[]>);

  if (loading) {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="shimmer" style={{ height: 90, borderRadius: 10 }} />
        ))}
      </div>
    );
  }

  return (
    <div style={{ width: '100%' }}>
      {CATEGORY_ORDER.map(cat => {
        const items = grouped[cat] ?? [];
        if (!items.length) return null;
        return (
          <div key={cat} style={{ marginBottom: 24 }}>
            <div style={{ marginBottom: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
              <CategoryLabel cat={cat} />
              <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.05)' }} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: 10 }}>
              {items.map(problem => (
                <button
                  key={problem.slug}
                  id={`problem-${problem.slug}`}
                  onClick={() => onSelect(problem)}
                  onMouseEnter={() => setHoverId(problem.slug)}
                  onMouseLeave={() => setHoverId(null)}
                  style={{
                    background: hoverId === problem.slug
                      ? 'rgba(20,28,55,0.9)'
                      : 'rgba(13,18,36,0.6)',
                    border: hoverId === problem.slug
                      ? '1px solid rgba(0,245,212,0.25)'
                      : '1px solid rgba(255,255,255,0.07)',
                    borderRadius: 10,
                    padding: '14px 16px',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                    transition: 'all 200ms cubic-bezier(0.4,0,0.2,1)',
                    transform: hoverId === problem.slug ? 'translateY(-2px)' : 'none',
                    boxShadow: hoverId === problem.slug ? '0 8px 24px rgba(0,0,0,0.3)' : 'none',
                    textAlign: 'left',
                    width: '100%',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <span style={{ fontFamily: 'Space Grotesk, sans-serif', fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.3 }}>
                      {problem.name}
                    </span>
                    <DifficultyBadge diff={problem.difficulty} />
                  </div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: 0 }}>
                    {problem.description.split(' — ')[0]}
                  </p>
                  <div style={{ marginTop: 2 }}>
                    <code style={{ fontSize: '0.72rem' }}>{problem.hint}</code>
                  </div>
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
