"""
visp.models.base
~~~~~~~~~~~~~~~~
Abstract interface that every detector model must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class EventType(str, Enum):
    VIOLENCE = "violence_detected"
    PPE_VIOLATION = "ppe_violation"
    ZONE_INTRUSION = "zone_intrusion"
    NORMAL = "normal"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """Single inference result for one clip."""

    event_type: EventType
    confidence: float
    frame_id: int
    bounding_boxes: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    raw_logits: np.ndarray | None = None

    @property
    def is_alert(self) -> bool:
        return self.event_type != EventType.NORMAL and self.event_type != EventType.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "confidence": round(self.confidence, 4),
            "frame_id": self.frame_id,
            "bounding_boxes": self.bounding_boxes,
            "labels": self.labels,
        }


class BaseDetector(ABC):
    """
    Abstract detector. Subclass this for every model backend.

    Subclasses must implement:
        - load()        → load weights, move to device
        - preprocess()  → (T, H, W, C) uint8 → model input tensor
        - infer()       → tensor → DetectionResult
    """

    def __init__(self, device: str = "cpu", confidence_threshold: float = 0.75) -> None:
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights and move to target device."""

    @abstractmethod
    def preprocess(self, frames: np.ndarray) -> object:
        """
        Convert a raw clip (T, H, W, C) uint8 array to
        a model-ready tensor / dict.
        """

    @abstractmethod
    def infer(self, preprocessed: object, frame_id: int) -> DetectionResult:
        """Run forward pass and return a DetectionResult."""

    def __call__(self, frames: np.ndarray, frame_id: int = 0) -> DetectionResult:
        if not self._loaded:
            self.load()
            self._loaded = True
        preprocessed = self.preprocess(frames)
        return self.infer(preprocessed, frame_id)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device!r}, threshold={self.confidence_threshold})"
