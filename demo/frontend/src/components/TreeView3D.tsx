import React, { useRef, useState, useCallback, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Billboard } from '@react-three/drei';
import { useSpring, animated } from '@react-spring/three';
import * as THREE from 'three';

import { NodeSnapshot, IndividualSnapshot } from '../types';
import { useTreeLayout, flattenTree, FlatNode3D } from '../hooks/useTreeLayout';
import { useRunStore } from '../store/runStore';

// ---- Color mapping per ntype ----
const NODE_COLORS: Record<string, string> = {
  op:      '#00bbf9',
  var:     '#f9c74f',
  const:   '#9b5de5',
  cmp:     '#f94144',
  ternary: '#43aa8b',
};

function nodeColor(ntype: string): string {
  return NODE_COLORS[ntype] ?? '#888';
}

function nodeRadius(subtreeSize: number): number {
  return Math.max(0.12, Math.min(0.28, 0.1 + subtreeSize * 0.018));
}

// ---- Animated 3D node sphere ----
interface NodeSphereProps {
  flatNode: FlatNode3D;
  onClick: (node: FlatNode3D) => void;
  selected: boolean;
}

function NodeSphere({ flatNode, onClick, selected }: NodeSphereProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const color = nodeColor(flatNode.node.ntype);
  const radius = nodeRadius(flatNode.subtreeSize);

  const { position, scale } = useSpring({
    position: [flatNode.x, flatNode.y, flatNode.z] as [number, number, number],
    scale: selected ? 1.5 : hovered ? 1.25 : 1,
    config: { tension: 200, friction: 22 },
  });

  // Subtle breathing animation
  useFrame((_, delta) => {
    if (meshRef.current && (selected || hovered)) {
      meshRef.current.rotation.y += delta * 0.8;
    }
  });

  return (
    <animated.mesh
      ref={meshRef}
      position={position}
      scale={scale}
      onClick={(e) => { e.stopPropagation(); onClick(flatNode); }}
      onPointerEnter={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = 'pointer'; }}
      onPointerLeave={(e) => { e.stopPropagation(); setHovered(false); document.body.style.cursor = 'auto'; }}
    >
      <sphereGeometry args={[radius, 20, 20]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={selected ? 0.8 : hovered ? 0.5 : 0.25}
        roughness={0.2}
        metalness={0.6}
        transparent
        opacity={flatNode.node.ntype === 'ternary' ? 0.9 : 1}
      />
    </animated.mesh>
  );
}

// ---- Edge (cylinder between parent and child) ----
interface EdgeProps {
  from: [number, number, number];
  to:   [number, number, number];
  opacity?: number;
}

function Edge({ from, to, opacity = 0.35 }: EdgeProps) {
  const mid = new THREE.Vector3(
    (from[0] + to[0]) / 2,
    (from[1] + to[1]) / 2,
    (from[2] + to[2]) / 2,
  );
  const dir = new THREE.Vector3(to[0] - from[0], to[1] - from[1], to[2] - from[2]);
  const length = dir.length();
  const quat = new THREE.Quaternion();
  quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());

  return (
    <mesh position={[mid.x, mid.y, mid.z]} quaternion={quat}>
      <cylinderGeometry args={[0.015, 0.015, length, 6]} />
      <meshStandardMaterial color="#334155" transparent opacity={opacity} roughness={0.9} />
    </mesh>
  );
}

// ---- Label billboard ----
function NodeLabel({ flatNode }: { flatNode: FlatNode3D }) {
  const { node } = flatNode;
  const label = node.ntype === 'ternary' ? '?' : String(node.value);
  return (
    <Billboard position={[flatNode.x, flatNode.y + 0.35, flatNode.z]}>
      <Text fontSize={0.18} color="#e8eaf6" anchorX="center" anchorY="middle" outlineWidth={0.02} outlineColor="#03050d">
        {label}
      </Text>
    </Billboard>
  );
}

// ---- Scene ----
function TreeScene({
  bestIndividual,
  onNodeClick,
}: {
  bestIndividual: IndividualSnapshot | null;
  onNodeClick: (node: FlatNode3D) => void;
}) {
  const selectedId = useRunStore(s => s.selectedIndividualId);
  const treeLayout = useTreeLayout(bestIndividual?.tree ?? null);
  const flatNodes  = treeLayout ? flattenTree(treeLayout) : [];

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[5, 8, 5]}  intensity={1.2} color="#00f5d4" />
      <pointLight position={[-5, -4, -3]} intensity={0.6} color="#9b5de5" />
      <directionalLight position={[0, 10, 5]} intensity={0.5} />

      {/* Edges */}
      {flatNodes.map(fn =>
        fn.parentPos ? (
          <Edge
            key={`edge-${fn.node.node_id}`}
            from={fn.parentPos}
            to={[fn.x, fn.y, fn.z]}
          />
        ) : null
      )}

      {/* Nodes */}
      {flatNodes.map(fn => (
        <NodeSphere
          key={fn.node.node_id}
          flatNode={fn}
          onClick={onNodeClick}
          selected={selectedId === fn.node.node_id}
        />
      ))}

      {/* Labels (only for top 2 depth levels to avoid clutter) */}
      {flatNodes.filter(fn => fn.depth <= 2).map(fn => (
        <NodeLabel key={`lbl-${fn.node.node_id}`} flatNode={fn} />
      ))}
    </>
  );
}

// ---- Node detail side panel ----
interface NodePanelProps {
  flatNode: FlatNode3D;
  onClose: () => void;
}

function nodeSubtreeExpr(node: NodeSnapshot): string {
  if (node.ntype === 'var' || node.ntype === 'const') return String(node.value);
  if ((node.ntype === 'op' || node.ntype === 'cmp') && node.children.length === 2) {
    return `(${nodeSubtreeExpr(node.children[0])} ${node.value} ${nodeSubtreeExpr(node.children[1])})`;
  }
  if (node.ntype === 'ternary' && node.children.length === 3) {
    return `(${nodeSubtreeExpr(node.children[1])} if (${nodeSubtreeExpr(node.children[0])}) else ${nodeSubtreeExpr(node.children[2])})`;
  }
  return String(node.value);
}

function NodePanel({ flatNode, onClose }: NodePanelProps) {
  const color = nodeColor(flatNode.node.ntype);
  const expr  = nodeSubtreeExpr(flatNode.node);
  return (
    <div style={{
      position: 'absolute', top: 12, right: 12, zIndex: 10,
      background: 'rgba(13,18,36,0.92)', border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 12, padding: '16px 20px', minWidth: 220, maxWidth: 300,
      backdropFilter: 'blur(16px)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#8892b0' }}>
          Node Detail
        </span>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#4a5568', cursor: 'pointer', fontSize: '1rem' }}>✕</button>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: color, boxShadow: `0 0 8px ${color}` }} />
        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.8rem', color }}>{flatNode.node.ntype}</span>
        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.9rem', color: '#e8eaf6', fontWeight: 600 }}>
          {String(flatNode.node.value)}
        </span>
      </div>
      <div style={{ fontSize: '0.72rem', color: '#8892b0', marginBottom: 4 }}>Subtree expression:</div>
      <pre style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: '0.76rem',
        color: '#00f5d4', background: 'rgba(0,0,0,0.3)',
        borderRadius: 6, padding: '8px 10px', overflow: 'auto', maxHeight: 120,
        whiteSpace: 'pre-wrap', wordBreak: 'break-all',
      }}>{expr}</pre>
      <div style={{ marginTop: 10, fontSize: '0.72rem', color: '#4a5568' }}>
        Subtree size: {flatNode.subtreeSize} node{flatNode.subtreeSize !== 1 ? 's' : ''} · Depth: {flatNode.depth}
      </div>
    </div>
  );
}

// ---- Main export ----
interface TreeView3DProps {
  bestIndividual: IndividualSnapshot | null;
}

export function TreeView3D({ bestIndividual }: TreeView3DProps) {
  const [selectedNode, setSelectedNode] = useState<FlatNode3D | null>(null);
  const { selectIndividual } = useRunStore();

  const handleNodeClick = useCallback((fn: FlatNode3D) => {
    setSelectedNode(fn);
    selectIndividual(fn.node.node_id);
  }, [selectIndividual]);

  const handleClose = useCallback(() => {
    setSelectedNode(null);
    selectIndividual(null);
  }, [selectIndividual]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: 'var(--bg-deep)', borderRadius: 'var(--radius-lg)', overflow: 'hidden', border: 'var(--border-dim)' }}>
      {!bestIndividual && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 2, flexDirection: 'column', gap: 8 }}>
          <div style={{ fontSize: '2rem', opacity: 0.3 }}>🌱</div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Waiting for first generation…</div>
        </div>
      )}

      <Canvas
        camera={{ position: [0, 0, 8], fov: 55 }}
        style={{ background: 'transparent' }}
        gl={{ antialias: true, alpha: true }}
      >
        <Suspense fallback={null}>
          <TreeScene bestIndividual={bestIndividual} onNodeClick={handleNodeClick} />
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={2}
            maxDistance={20}
            dampingFactor={0.08}
            enableDamping
          />
        </Suspense>
      </Canvas>

      {selectedNode && (
        <NodePanel flatNode={selectedNode} onClose={handleClose} />
      )}

      {/* Legend */}
      <div style={{ position: 'absolute', bottom: 12, left: 12, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        {Object.entries(NODE_COLORS).map(([ntype, color]) => (
          <span key={ntype} className="node-legend">
            <span className="node-dot" style={{ background: color }} />
            {ntype}
          </span>
        ))}
      </div>

      {/* Panel caption */}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0 }}>
        <div className="panel-caption">
          Best program found so far, shown as a tree — trace it top to bottom to read the formula. Click any node for details.
        </div>
      </div>
    </div>
  );
}
