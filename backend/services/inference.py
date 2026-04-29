"""
visp.services.inference
~~~~~~~~~~~~~~~~~~~~~~~
Central inference service: buffers incoming frames into clips,
dispatches them to the active detector, and publishes results.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncGenerator, Deque

import numpy as np

from backend.core.config import ModelBackend, Settings
from backend.models.base import BaseDetector, DetectionResult, EventType

logger = logging.getLogger(__name__)


@dataclass
class StreamBuffer:
    """Per-camera frame ring buffer."""

    camera_id: str
    clip_length: int
    frame_skip: int
    _frames: Deque[np.ndarray] = field(init=False, default_factory=deque)
    _frame_counter: int = field(init=False, default=0)

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Add a frame. Returns a clip (T, H, W, C) when the buffer is full,
        else None.
        """
        self._frame_counter += 1
        if self._frame_counter % self.frame_skip != 0:
            return None

        self._frames.append(frame)
        if len(self._frames) > self.clip_length:
            self._frames.popleft()

        if len(self._frames) == self.clip_length:
            return np.stack(list(self._frames), axis=0)
        return None


def _build_detector(settings: Settings) -> BaseDetector:
    """Instantiate the correct detector from settings."""
    common = {
        "device": settings.device,
        "confidence_threshold": settings.confidence_threshold,
    }

    match settings.model_backend:
        case ModelBackend.MVIT:
            from backend.models.mvit import MViTDetector
            return MViTDetector(clip_length=settings.clip_length, **common)

        case ModelBackend.VIVIT:
            from backend.models.vivit import ViViTDetector
            return ViViTDetector(clip_length=settings.clip_length, **common)

        case ModelBackend.R2PLUS1D:
            from backend.models.r2plus1d import R2Plus1DDetector
            return R2Plus1DDetector(clip_length=settings.clip_length, **common)

        case ModelBackend.ONNX:
            from backend.models.onnx_detector import OnnxDetector
            return OnnxDetector(model_path=settings.onnx_model_path, **common)

        case _:
            raise ValueError(f"Unknown model backend: {settings.model_backend}")


class InferenceService:
    """
    Manages per-camera stream buffers and runs inference.

    Usage
    -----
    svc = InferenceService(settings)
    async for result in svc.process_frame(camera_id, frame, frame_id):
        ...  # only yields on non-normal events above threshold
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._detector: BaseDetector | None = None
        self._buffers: dict[str, StreamBuffer] = {}
        self._lock = asyncio.Lock()

    async def get_detector(self) -> BaseDetector:
        if self._detector is None:
            async with self._lock:
                if self._detector is None:  # double-checked locking
                    self._detector = _build_detector(self._settings)
                    logger.info("Detector initialised: %s", self._detector)
        return self._detector

    def _get_buffer(self, camera_id: str) -> StreamBuffer:
        if camera_id not in self._buffers:
            self._buffers[camera_id] = StreamBuffer(
                camera_id=camera_id,
                clip_length=self._settings.clip_length,
                frame_skip=self._settings.frame_skip,
            )
        return self._buffers[camera_id]

    async def process_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        frame_id: int,
    ) -> AsyncGenerator[DetectionResult, None]:
        """
        Push a frame into the camera's buffer. If a full clip is ready,
        run inference and yield the result (only if actionable).
        """
        buf = self._get_buffer(camera_id)
        clip = buf.push(frame)

        if clip is None:
            return

        detector = await self.get_detector()
        # Offload blocking inference to a thread pool
        result: DetectionResult = await asyncio.get_event_loop().run_in_executor(
            None, detector, clip, frame_id
        )

        logger.debug(
            "camera=%s frame=%d event=%s confidence=%.3f",
            camera_id, frame_id, result.event_type.value, result.confidence,
        )

        if result.is_alert:
            yield result

    def remove_camera(self, camera_id: str) -> None:
        self._buffers.pop(camera_id, None)
        logger.info("Removed stream buffer for camera=%s", camera_id)
