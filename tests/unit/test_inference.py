"""
tests/unit/test_inference.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the InferenceService and StreamBuffer.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.models.base import DetectionResult, EventType
from backend.services.inference import InferenceService, StreamBuffer


# ── StreamBuffer ──────────────────────────────────────────────────────────────

class TestStreamBuffer:
    def test_returns_none_until_full(self):
        buf = StreamBuffer(camera_id="cam-01", clip_length=4, frame_skip=1)
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(3):
            assert buf.push(frame) is None

    def test_returns_clip_when_full(self):
        buf = StreamBuffer(camera_id="cam-01", clip_length=4, frame_skip=1)
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        clip = None
        for _ in range(4):
            clip = buf.push(frame)
        assert clip is not None
        assert clip.shape == (4, 224, 224, 3)

    def test_frame_skip_filters_frames(self):
        buf = StreamBuffer(camera_id="cam-01", clip_length=4, frame_skip=2)
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        # Only frames 2, 4, 6, 8 (every 2nd) should count
        results = [buf.push(frame) for _ in range(8)]
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 1  # clip filled on the 8th frame

    def test_sliding_window_after_first_clip(self):
        buf = StreamBuffer(camera_id="cam-01", clip_length=2, frame_skip=1)
        f1 = np.ones((10, 10, 3), dtype=np.uint8)
        f2 = np.ones((10, 10, 3), dtype=np.uint8) * 2
        f3 = np.ones((10, 10, 3), dtype=np.uint8) * 3

        buf.push(f1)
        clip2 = buf.push(f2)
        assert clip2 is not None

        clip3 = buf.push(f3)
        assert clip3 is not None
        assert np.all(clip3[0] == 2)  # f2
        assert np.all(clip3[1] == 3)  # f3


# ── InferenceService ──────────────────────────────────────────────────────────

class TestInferenceService:
    def _make_settings(self, **kwargs):
        settings = MagicMock()
        settings.model_backend.value = "mvit"
        settings.device = "cpu"
        settings.confidence_threshold = 0.75
        settings.clip_length = 4
        settings.frame_skip = 1
        for k, v in kwargs.items():
            setattr(settings, k, v)
        return settings

    @pytest.mark.asyncio
    async def test_no_yield_below_clip_length(self):
        svc = InferenceService(self._make_settings())
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        results = []
        async for r in svc.process_frame("cam-01", frame, 0):
            results.append(r)
        assert results == []  # buffer not full yet

    @pytest.mark.asyncio
    async def test_yields_alert_result(self):
        svc = InferenceService(self._make_settings())

        alert_result = DetectionResult(
            event_type=EventType.VIOLENCE,
            confidence=0.95,
            frame_id=3,
        )
        mock_detector = MagicMock()
        mock_detector.return_value = alert_result
        svc._detector = mock_detector
        svc._loaded = True

        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        results = []
        # push 4 frames to fill the buffer
        for i in range(4):
            async for r in svc.process_frame("cam-01", frame, i):
                results.append(r)

        assert len(results) == 1
        assert results[0].event_type == EventType.VIOLENCE

    @pytest.mark.asyncio
    async def test_normal_events_not_yielded(self):
        svc = InferenceService(self._make_settings())

        normal_result = DetectionResult(
            event_type=EventType.NORMAL,
            confidence=0.99,
            frame_id=3,
        )
        mock_detector = MagicMock(return_value=normal_result)
        svc._detector = mock_detector

        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        results = []
        for i in range(4):
            async for r in svc.process_frame("cam-01", frame, i):
                results.append(r)

        assert results == []

    def test_remove_camera_clears_buffer(self):
        svc = InferenceService(self._make_settings())
        svc._buffers["cam-99"] = StreamBuffer("cam-99", 16, 1)
        svc.remove_camera("cam-99")
        assert "cam-99" not in svc._buffers

    def test_remove_nonexistent_camera_is_safe(self):
        svc = InferenceService(self._make_settings())
        svc.remove_camera("does-not-exist")  # should not raise
