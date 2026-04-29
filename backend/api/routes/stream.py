"""
visp.api.routes.stream
~~~~~~~~~~~~~~~~~~~~~~
WebSocket endpoint for live video stream ingestion.

Protocol
--------
Client → Server : raw JPEG bytes (one frame per message)
Server → Client : JSON-encoded DetectionResult on alert events

Example client (Python)
-----------------------
    async with websockets.connect("ws://localhost:8000/ws/stream/cam-01") as ws:
        while cap.isOpened():
            _, frame = cap.read()
            _, buf = cv2.imencode(".jpg", frame)
            await ws.send(buf.tobytes())
            if msg := await asyncio.wait_for(ws.recv(), timeout=0.01):
                print(json.loads(msg))
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.dependencies import get_inference_service, get_event_queue
from backend.models.base import DetectionResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["stream"])


def _decode_frame(data: bytes) -> np.ndarray | None:
    """Decode JPEG bytes → (H, W, C) uint8 BGR array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame  # None if decode fails


def _result_to_message(result: DetectionResult, camera_id: str) -> str:
    return json.dumps(
        {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "camera_id": camera_id,
            **result.to_dict(),
        }
    )


@router.websocket("/stream/{camera_id}")
async def video_stream(websocket: WebSocket, camera_id: str) -> None:
    """
    Accepts a continuous stream of JPEG frames from a camera client.
    Fires JSON alert messages back whenever a safety event is detected.
    """
    inference_svc = get_inference_service()
    event_queue = get_event_queue()

    await websocket.accept()
    logger.info("Stream connected: camera_id=%s", camera_id)
    frame_id = 0

    try:
        while True:
            data: bytes = await websocket.receive_bytes()
            frame = _decode_frame(data)

            if frame is None:
                logger.warning("camera=%s: failed to decode frame %d", camera_id, frame_id)
                frame_id += 1
                continue

            async for result in inference_svc.process_frame(camera_id, frame, frame_id):
                message = _result_to_message(result, camera_id)
                # Push to Redis event queue
                await event_queue.publish(camera_id, json.loads(message))
                # Echo back to the connected client
                await websocket.send_text(message)

            frame_id += 1

    except WebSocketDisconnect:
        logger.info("Stream disconnected: camera_id=%s after %d frames", camera_id, frame_id)
        inference_svc.remove_camera(camera_id)
    except Exception as exc:
        logger.exception("Unhandled error in stream for camera_id=%s: %s", camera_id, exc)
        await websocket.close(code=1011, reason="Internal server error")
        inference_svc.remove_camera(camera_id)
