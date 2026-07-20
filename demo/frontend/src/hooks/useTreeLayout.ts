import { useMemo } from 'react';
import { NodeSnapshot, TreeNode3D } from '../types';

/**
 * Arc-fan 3D tree layout (Reingold-Tilford inspired, deterministic).
 *
 * Root at y=2, each depth level drops by 1.8 units.
 * Children fan out on an arc at each level, z-jitter adds depth perception.
 * Subtree widths drive the horizontal spread so dense subtrees don't collide.
 */

const DEPTH_STEP  = 1.8;   // vertical distance per depth level
const SPREAD_BASE = 2.5;   // horizontal spread multiplier
const Z_JITTER    = 0.3;   // max z-offset for depth perception

function subtreeSize(node: NodeSnapshot): number {
  if (!node.children.length) return 1;
  return 1 + node.children.reduce((s, c) => s + subtreeSize(c), 0);
}

function layoutNode(
  node: NodeSnapshot,
  x: number,
  y: number,
  z: number,
  spreadWidth: number,
  depth: number,
): TreeNode3D {
  const size = subtreeSize(node);
  const childCount = node.children.length;

  let layoutChildren: TreeNode3D[] = [];

  if (childCount > 0) {
    const childY = y - DEPTH_STEP;
    const childSizes = node.children.map(c => Math.max(1, subtreeSize(c)));
    const totalSize  = childSizes.reduce((a, b) => a + b, 0);

    // Fan children evenly within spreadWidth, weighted by subtree size
    const startX = x - spreadWidth / 2;
    let cursor = startX;

    layoutChildren = node.children.map((child, i) => {
      const fraction = childSizes[i] / totalSize;
      const childW   = spreadWidth * fraction;
      const childX   = cursor + childW / 2;
      const childZ   = z + (i % 2 === 0 ? Z_JITTER : -Z_JITTER) * (depth + 1) * 0.3;
      cursor += childW;
      return layoutNode(child, childX, childY, childZ, childW * SPREAD_BASE * 0.6, depth + 1);
    });
  }

  return {
    node,
    x,
    y,
    z,
    subtreeSize: size,
    depth,
    children: layoutChildren,
  };
}

export function useTreeLayout(root: NodeSnapshot | null): TreeNode3D | null {
  return useMemo(() => {
    if (!root) return null;
    return layoutNode(root, 0, 0, 0, SPREAD_BASE * Math.max(2, subtreeSize(root) * 0.4), 0);
  }, [root?.node_id, JSON.stringify(root)]);
}

/** Flatten a TreeNode3D tree into arrays for Three.js rendering. */
export interface FlatNode3D extends TreeNode3D {
  parentPos: [number, number, number] | null;
}

export function flattenTree(node: TreeNode3D, parentPos: [number, number, number] | null = null): FlatNode3D[] {
  const flat: FlatNode3D[] = [{ ...node, parentPos }];
  for (const child of node.children) {
    flat.push(...flattenTree(child, [node.x, node.y, node.z]));
  }
  return flat;
}
