"""
scripts/export_onnx.py
~~~~~~~~~~~~~~~~~~~~~~
Export a VISP PyTorch model to an optimized ONNX graph.

Usage
-----
    python scripts/export_onnx.py \
        --model mvit \
        --checkpoint checkpoints/mvit_safety_v2.pt \
        --output models/visp_mvit.onnx \
        --clip-length 16 \
        --optimize

After export, verify with:
    python scripts/export_onnx.py --verify --output models/visp_mvit.onnx
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export VISP model to ONNX")
    p.add_argument("--model", choices=["mvit", "vivit", "r2plus1d"], default="mvit")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("models/visp_model.onnx"))
    p.add_argument("--clip-length", type=int, default=16)
    p.add_argument("--spatial-size", type=int, default=224)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--optimize", action="store_true", help="Run ONNX graph optimizations")
    p.add_argument("--verify", action="store_true", help="Verify existing ONNX model")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_torch_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "mvit":
        from backend.models.mvit import MViTDetector
        detector = MViTDetector(
            checkpoint_path=args.checkpoint,
            clip_length=args.clip_length,
            spatial_size=args.spatial_size,
            device=args.device,
        )
    else:
        raise NotImplementedError(f"Export for {args.model!r} not yet implemented")

    detector.load()
    assert detector.model is not None
    return detector.model


def export(args: argparse.Namespace) -> None:
    logger.info("Loading %s model …", args.model.upper())
    model = load_torch_model(args)
    model.eval()

    # Dummy input: (batch=1, C=3, T, H, W)
    dummy = torch.zeros(
        1, 3, args.clip_length, args.spatial_size, args.spatial_size,
        device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting to %s (opset=%d) …", args.output, args.opset)

    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        opset_version=args.opset,
        input_names=["video_clip"],
        output_names=["logits"],
        dynamic_axes={
            "video_clip": {0: "batch_size", 2: "clip_length"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    if args.optimize:
        _optimize_onnx(args.output)

    logger.info("✅ ONNX model saved to %s", args.output)
    _print_model_info(args.output)


def _optimize_onnx(path: Path) -> None:
    try:
        import onnx
        from onnxruntime.transformers import optimizer  # type: ignore[import]
    except ImportError:
        logger.warning("onnxruntime-tools not available; skipping optimization")
        return

    logger.info("Running ONNX graph optimizations …")
    opt_path = path.with_suffix(".opt.onnx")
    optimized = optimizer.optimize_model(str(path), model_type="bert")
    optimized.save_model_to_file(str(opt_path))
    # Overwrite original
    opt_path.rename(path)
    logger.info("Optimization complete")


def _print_model_info(path: Path) -> None:
    import onnx
    model = onnx.load(str(path))
    inputs = [f"{i.name}: {[d.dim_value or '?' for d in i.type.tensor_type.shape.dim]}" for i in model.graph.input]
    outputs = [f"{o.name}: {[d.dim_value or '?' for d in o.type.tensor_type.shape.dim]}" for o in model.graph.output]
    logger.info("Inputs:  %s", inputs)
    logger.info("Outputs: %s", outputs)
    size_mb = path.stat().st_size / 1e6
    logger.info("File size: %.1f MB", size_mb)


def verify(args: argparse.Namespace) -> None:
    import onnxruntime as ort
    logger.info("Verifying %s …", args.output)
    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    dummy = np.zeros(
        (1, 3, args.clip_length, args.spatial_size, args.spatial_size), dtype=np.float32
    )
    outputs = sess.run(None, {"video_clip": dummy})
    logger.info("Output shape: %s  (all zeros OK for dummy input)", outputs[0].shape)
    logger.info("✅ Verification passed")


def main() -> None:
    args = parse_args()
    if args.verify:
        verify(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
