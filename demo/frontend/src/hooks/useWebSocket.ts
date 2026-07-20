import { useEffect, useRef, useCallback } from 'react';
import { GenerationEvent, RunResult, EngineType } from '../types';
import { useRunStore } from '../store/runStore';
import { useGuidanceStore } from '../store/guidanceStore';

const WS_BASE = `ws://${window.location.hostname}:8000`;

export function useWebSocket(runId: string | null, engineType: EngineType = 'genesis') {
  const wsRef = useRef<WebSocket | null>(null);
  const { startRun, pushEvent, finishRun, errorRun } = useRunStore();
  const { processEvent, tourActive, startTour, hasSeenTour } = useGuidanceStore();

  const connect = useCallback(() => {
    if (!runId) return;

    startRun(runId, engineType);

    // Start tour on very first run
    if (!hasSeenTour && !tourActive) {
      startTour();
    }

    const ws = new WebSocket(`${WS_BASE}/ws/runs/${runId}`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);

      if (data.type === 'ping') return;

      if (data.type === 'done') {
        const result = data.result as RunResult;
        finishRun(runId, result);
        return;
      }

      if (data.type === 'error') {
        errorRun(runId, data.message);
        return;
      }

      // It's a GenerationEvent
      const event = data as GenerationEvent;
      pushEvent(runId, event);
      processEvent(event);
    };

    ws.onerror = () => {
      errorRun(runId, 'WebSocket connection failed.');
    };

    ws.onclose = () => {
      wsRef.current = null;
    };
  }, [runId]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  return { reconnect: connect };
}
