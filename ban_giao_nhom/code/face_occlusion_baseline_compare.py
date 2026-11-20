#!/usr/bin/env python3
"""B0: compare one reference face with query images using cosine similarity.

Only exactly one detected face is accepted. No identity threshold is inferred.
Imports of ML packages are delayed so --help works before installation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "1.0.1"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path, help="One reference image")
    p.add_argument("--queries", required=True, nargs="+", type=Path,
                   help="One or more query images; quote every path with spaces")
    p.add_argument("--output", required=True, type=Path, help="Output JSON; overwritten if present")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--weights-dir", type=Path, help="Optional TORCH_HOME; weights in checkpoints/")
    return p


def error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def version_info() -> dict[str, Any]:
    packages = {}
    for name in ("facenet-pytorch", "torch", "torchvision", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages}


def image_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.expanduser().resolve()), "status": "pending", "sha256": None,
            "original_mode": None, "size_after_exif": None, "face_count": None,
            "boxes_xyxy": [], "detector_scores": [], "embedding_dim": None, "error": None}


def extract_embedding(path: Path, detector: Any, encoder: Any, torch: Any,
                      device: Any) -> tuple[dict[str, Any], Any]:
    """Return an auditable record and normalized 1-D embedding, or None on failure."""
    from PIL import Image, ImageOps

    record = image_record(path)
    file_path = Path(record["path"])
    try:
        # pathlib/Pillow avoid OpenCV's Windows Unicode filename pitfalls.
        with file_path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            record["sha256"] = digest.hexdigest()
        with Image.open(file_path) as source:
            record["original_mode"] = source.mode
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            rgb.load()
        record["size_after_exif"] = [rgb.width, rgb.height]
    except Exception as exc:
        record.update(status="error", error=error("image_read_failed", f"{type(exc).__name__}: {exc}"))
        return record, None

    # MTCNN cannot build a valid pyramid below its configured min_face_size.
    # Preserve these samples as explicit failures; upscaling would change the protocol.
    minimum_size = int(detector.min_face_size)
    if min(rgb.size) < minimum_size:
        record.update(status="error", error=error(
            "image_too_small",
            f"Image size {rgb.width}x{rgb.height} has a side below MTCNN min_face_size={minimum_size}; image was not enlarged."))
        return record, None

    try:
        with torch.inference_mode():
            # detect all boxes once; never silently select the largest of several people.
            boxes, scores = detector.detect(rgb)
            record["face_count"] = 0 if boxes is None else len(boxes)
            if boxes is not None:
                record["boxes_xyxy"] = [[float(x) for x in box] for box in boxes]
                record["detector_scores"] = [float(x) for x in scores]
            if record["face_count"] != 1:
                code = "no_face" if record["face_count"] == 0 else "multiple_faces"
                record.update(status="error", error=error(code, "Exactly one detected face is required."))
                return record, None
            # extract applies the same crop and (x - 127.5) / 128 to both sides.
            crops = detector.extract(rgb, boxes, save_path=None)
            if crops is None or tuple(crops.shape) != (1, 3, 160, 160):
                raise ValueError("Unexpected face crop shape")
            raw = encoder(crops.to(device))[0]
            if raw.numel() != 512 or not bool(torch.isfinite(raw).all()):
                raise ValueError("Embedding must contain 512 finite values")
            norm = torch.linalg.vector_norm(raw)
            if not bool(norm > 1e-12):
                raise ValueError("Embedding norm is zero")
            embedding = (raw / norm).detach().cpu()
            record.update(status="ok", embedding_dim=512)
            return record, embedding
    except Exception as exc:
        record.update(status="error", error=error("inference_failed", f"{type(exc).__name__}: {exc}"))
        return record, None


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomic UTF-8 JSON; do not leave a half-written report after a crash."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def main(argv: list[str] | None = None) -> int:
    # Windows pipes may use cp1252 even with Vietnamese filesystem paths.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parser().parse_args(argv)
    output_path = args.output.expanduser().resolve()
    input_paths = [args.reference, *args.queries]
    if any(output_path == p.expanduser().resolve() for p in input_paths):
        parser().error("--output must not overwrite an input image")
    if args.weights_dir:
        os.environ["TORCH_HOME"] = str(args.weights_dir.expanduser().resolve())
    report: dict[str, Any] = {
        "schema_version": "1.0", "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(), "baseline": "B0",
        "environment": version_info(), "device_requested": args.device, "device_used": None,
        "config": {
            "detector": "MTCNN", "keep_all": True, "image_size": 160, "margin": 0,
            "min_face_size": 20, "detector_stage_thresholds": [0.6, 0.7, 0.7],
            "tiny_image_policy": "Reject image if min(width,height) < min_face_size; no automatic upscaling",
            "factor": 0.709, "post_process": True,
            "standardization": "(RGB_uint8_float - 127.5) / 128.0",
            "crop": "MTCNN box crop and resize; no landmark similarity alignment",
            "encoder": "InceptionResnetV1", "pretrained": "vggface2", "classify": False,
            "embedding_dim": 512, "embedding_normalization": "L2",
            "score": "dot product of L2-normalized embeddings (cosine), range [-1, 1]",
            "identity_threshold": None, "torch_home": os.getenv("TORCH_HOME"),
        },
        "reference": image_record(args.reference),
        "queries": [{**image_record(p), "cosine_similarity": None,
                     "comparison_error": None} for p in args.queries],
        "summary": {}, "fatal_error": None,
        "notes": ["Cosine similarity is not an identity probability.",
                  "Detector scores describe face detection, not identity confidence.",
                  "This compares pairs; it does not measure benchmark accuracy or implement open-set 1:N.",
                  "Pretraining may overlap benchmark identities; this script cannot audit pretraining data."],
    }
    exit_code = 0
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu or auto")
        device = torch.device("cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()) else "cpu")
        report["device_used"] = str(device)
        report["environment"]["cuda_runtime"] = torch.version.cuda
        report["environment"]["gpu_name"] = torch.cuda.get_device_name(0) if device.type == "cuda" else None
        detector = MTCNN(image_size=160, margin=0, min_face_size=20,
                         thresholds=[0.6, 0.7, 0.7], factor=0.709,
                         post_process=True, keep_all=True, device=device).eval()
        encoder = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
        report["reference"], reference_embedding = extract_embedding(
            args.reference, detector, encoder, torch, device)
        for index, path in enumerate(args.queries):
            record, query_embedding = extract_embedding(path, detector, encoder, torch, device)
            record.update(cosine_similarity=None, comparison_error=None)
            if reference_embedding is None:
                record["comparison_error"] = error("reference_invalid", "Reference embedding is unavailable.")
            elif query_embedding is not None:
                score = float(torch.dot(reference_embedding, query_embedding).clamp(-1.0, 1.0))
                record["cosine_similarity"] = score
            report["queries"][index] = record
        if reference_embedding is None:
            exit_code = 2
        elif any(q["status"] != "ok" for q in report["queries"]):
            exit_code = 1
    except Exception as exc:
        report["fatal_error"] = error("environment_or_model_failed", f"{type(exc).__name__}: {exc}")
        # Every query still has an explicit failure record when setup/download fails.
        for record in [report["reference"], *report["queries"]]:
            if record["status"] == "pending":
                record.update(status="not_processed", error=error(
                    "setup_failed", "See fatal_error; this image was not processed."))
        exit_code = 2
    report["summary"] = {
        "query_total": len(report["queries"]),
        "query_embedding_ok": sum(q["status"] == "ok" for q in report["queries"]),
        "query_scored": sum(q["cosine_similarity"] is not None for q in report["queries"]),
        "query_failed_or_not_processed": sum(q["status"] != "ok" for q in report["queries"]),
        "exit_code": exit_code,
    }
    try:
        write_report(output_path, report)
    except Exception as exc:
        print(f"Could not save JSON: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Saved JSON: {output_path}")
    print(json.dumps(report["summary"], ensure_ascii=False))
    if report["fatal_error"]:
        print(report["fatal_error"]["message"], file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
