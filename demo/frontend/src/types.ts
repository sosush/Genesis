// Shared TypeScript types for Genesis frontend

export type EngineType = 'genesis' | 'pure_evolutionary' | 'random';
export type EventType = 'evaluation' | 'scorer_training' | 'solved' | 'timeout';
export type Difficulty = 'easy' | 'medium' | 'hard';

export interface NodeSnapshot {
  node_id: string;
  ntype: 'op' | 'var' | 'const' | 'cmp' | 'ternary';
  value: string | number;
  children: NodeSnapshot[];
}

export interface IndividualSnapshot {
  individual_id: string;
  expr: string;
  fitness: number;
  parent_ids: string[];
  rank: number;
  is_pruned_by_scorer: boolean;
  tree: NodeSnapshot | null;
}

export interface GenerationEvent {
  run_id: string;
  generation: number;
  population_snapshot: IndividualSnapshot[];
  best_individual: IndividualSnapshot;
  fitness_curve: number[];
  fitness_histogram: number[];  // 10 bins
  neural_scorer_active: boolean;
  pruned_count: number;
  scorer_training_loss: number | null;
  event_type: EventType;
  engine_type: EngineType;
}

export interface Problem {
  name: string;
  slug: string;
  description: string;
  variables: string[];
  hint: string;
  difficulty: Difficulty;
  category: string;
}

export interface RunResult {
  run_id: string;
  engine_type: EngineType;
  program: string;
  generations_taken: number;
  best_fitness: number;
  fitness_curve: number[];
  solved: boolean;
}

export interface RunInfo {
  runId: string;
  engineType: EngineType;
  events: GenerationEvent[];
  latestEvent: GenerationEvent | null;
  result: RunResult | null;
  status: 'idle' | 'running' | 'done' | 'error';
  error: string | null;
}

// 3D layout types
export interface TreeNode3D {
  node: NodeSnapshot;
  x: number;
  y: number;
  z: number;
  subtreeSize: number;
  depth: number;
  children: TreeNode3D[];
}
