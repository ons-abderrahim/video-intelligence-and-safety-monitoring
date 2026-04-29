"""
visp.models.mvit
~~~~~~~~~~~~~~~~
MViT-v2 detector wrapper built on top of PyTorch Video.

References
----------
- Fan et al. "Multiscale Vision Transformers" (2021)  arXiv:2104.11227
- PyTorch Video: https://github.com/facebookresearch/pytorchvideo
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import BaseDetector, DetectionResult, EventType

logger = logging.getLogger(__name__)

# Label index → EventType mapping for VISP fine-tuned checkpoints
_LABEL_MAP: dict[int, EventType] = {
    0: EventType.NORMAL,
    1: EventType.VIOLENCE,
    2: EventType.PPE_VIOLATION,
    3: EventType.ZONE_INTRUSION,
}

# Standard Kinetics mean/std for normalisation
_MEAN = (0.45, 0.45, 0.45)
_STD = (0.225, 0.225, 0.225)


class MViTDetector(BaseDetector):
    """
    MViT-v2-S fine-tuned on the VISP safety dataset.

    Parameters
    ----------
    checkpoint_path:
        Path to a fine-tuned .pt checkpoint. If None, loads Kinetics-400
        pretrained weights via PyTorch Video (useful for prototyping).
    clip_length:
        Number of frames in each input clip (default 16).
    spatial_size:
        Spatial resolution fed to the model (default 224).
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        clip_length: int = 16,
        spatial_size: int = 224,
        device: str = "cpu",
        confidence_threshold: float = 0.75,
    ) -> None:
        super().__init__(device=device, confidence_threshold=confidence_threshold)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.clip_length = clip_length
        self.spatial_size = spatial_size
        self.model: torch.nn.Module | None = None

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        try:
            from pytorchvideo.models.hub import mvit_v2_s  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pytorchvideo is required for the MViT backend. "
                "Install it with: pip install pytorchvideo"
            ) from exc

        logger.info("Loading MViT-v2-S …")
        self.model = mvit_v2_s(pretrained=(self.checkpoint_path is None))

        if self.checkpoint_path is not None:
            state = torch.load(self.checkpoint_path, map_location="cpu")
            # Support both raw state_dict and Lightning checkpoints
            state_dict = state.get("state_dict", state)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded checkpoint: %s", self.checkpoint_path)

        self.model.eval()
        self.model.to(self.device)
        logger.info("MViT-v2-S ready on device=%s", self.device)

    def preprocess(self, frames: np.ndarray) -> Tensor:
        """
        Parameters
        ----------
        frames: np.ndarray
            Shape (T, H, W, C), dtype uint8, values 0-255.

        Returns
        -------
        Tensor of shape (1, C, T, H, W), float32, normalised.
        """
        if frames.ndim != 4:
            raise ValueError(f"Expected (T,H,W,C) array, got shape {frames.shape}")

        t = torch.from_numpy(frames).float() / 255.0  # (T, H, W, C)
        t = t.permute(3, 0, 1, 2)  # → (C, T, H, W)

        # Normalise
        mean = torch.tensor(_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        std = torch.tensor(_STD, dtype=torch.float32).view(3, 1, 1, 1)
        t = (t - mean) / std

        # Resize spatial dims
        c, t_len, h, w = t.shape
        t = t.view(c * t_len, h, w).unsqueeze(0)
        t = F.interpolate(t, size=(self.spatial_size, self.spatial_size), mode="bilinear", align_corners=False)
        t = t.squeeze(0).view(c, t_len, self.spatial_size, self.spatial_size)

        return t.unsqueeze(0).to(self.device)  # (1, C, T, H, W)

    def infer(self, preprocessed: Tensor, frame_id: int = 0) -> DetectionResult:
        assert self.model is not None, "Call load() before infer()"

        with torch.no_grad():
            logits: Tensor = self.model(preprocessed)  # (1, num_classes)

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        event_type = (
            _LABEL_MAP.get(class_idx, EventType.UNKNOWN)
            if confidence >= self.confidence_threshold
            else EventType.NORMAL
        )

        return DetectionResult(
            event_type=event_type,
            confidence=confidence,
            frame_id=frame_id,
            raw_logits=logits.cpu().numpy(),
        )
