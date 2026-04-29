import { useEffect, useRef, useState, useCallback } from "react";

const WS_BASE = import.meta.env.VITE_WS_URL || "ws://localhost:8000";
const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECT_ATTEMPTS = 10;

/**
 * useStream
 * ---------
 * Manages a WebSocket connection to /ws/stream/{cameraId}.
 *
 * Returns:
 *   lastEvent   – most recent DetectionResult payload (or null)
 *   connected   – boolean connection state
 *   error       – error string (or null)
 *   disconnect  – imperative disconnect function
 */
export function useStream(cameraId) {
  const wsRef = useRef(null);
  const attemptsRef = useRef(0);
  const timerRef = useRef(null);

  const [lastEvent, setLastEvent] = useState(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const url = `${WS_BASE}/ws/stream/${cameraId}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
      attemptsRef.current = 0;
      console.debug(`[useStream] connected to ${url}`);
    };

    ws.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data);
        setLastEvent(payload);
      } catch {
        console.warn("[useStream] failed to parse message", evt.data);
      }
    };

    ws.onerror = () => {
      setError(`WebSocket error for ${cameraId}`);
    };

    ws.onclose = (evt) => {
      setConnected(false);
      if (attemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        attemptsRef.current += 1;
        console.debug(
          `[useStream] reconnecting (attempt ${attemptsRef.current}) in ${RECONNECT_DELAY_MS}ms`
        );
        timerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
      } else {
        setError(`Max reconnect attempts reached for ${cameraId}`);
      }
    };
  }, [cameraId]);

  const disconnect = useCallback(() => {
    clearTimeout(timerRef.current);
    wsRef.current?.close();
    setConnected(false);
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(timerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { lastEvent, connected, error, disconnect };
}
