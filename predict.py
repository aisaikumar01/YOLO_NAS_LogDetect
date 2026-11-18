"""
Detect logs in a folder of photos with a YOLO-NAS model and save annotated copies.

Loads the trained YOLO-NAS-S detector (single class: "logs"), runs it over every
image in a source folder, and writes each image back out with a green circle
fitted to each log end. No labels, no text — just the circles.

    python predict.py                       # all of test_images -> predicted/
    python predict.py --conf 0.5            # stricter detections
    python predict.py --keep                # do not clear predicted/ first
    python predict.py single.jpg            # one image only

The weights file `weights_logs_nas_clean.pt` is a clean checkpoint: only the
raw network state dict, no training metadata of any kind.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent
WEIGHTS = PROJECT / "weights_logs_nas_clean.pt"
DEFAULT_SOURCE = PROJECT / "test_images"
DEFAULT_OUTPUT = PROJECT / "predicted"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

CONF = 0.25   # confidence threshold
IOU = 0.65    # NMS IoU threshold (the value the model was tuned with)
IMGSZ = 640   # inference resolution; the model was trained at 640

CLASS_NAMES = ["logs"]

# BGR green
COLOR = (0, 255, 0)
LINE_THICKNESS = 2


def log(msg: str) -> None:
    """Print to the real stdout (super_gradients rebinds sys.stdout on import)."""
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()


def clear_outputs(out_dir: Path) -> int:
    removed = 0
    if not out_dir.is_dir():
        return 0
    for p in out_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES + (".json",):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def build_model(weights: Path, imgsz: int, iou: float):
    """Build the YOLO-NAS-S graph and load the clean weights into it.

    The clean checkpoint holds a bare state dict, so the architecture comes
    from the super_gradients recipe and the weights load strictly. The
    preprocessing mirrors training: BGR->RGB, longest-side rescale to imgsz,
    bottom-right padding with 114, HWC->CHW, scale to 0..1.
    """
    os.environ.setdefault("CRASH_HANDLER", "FALSE")
    import logging
    logging.disable(logging.CRITICAL)

    from super_gradients.training import models
    from super_gradients.training.processing.processing import (
        ComposeProcessing, ReverseImageChannels, DetectionLongestMaxSizeRescale,
        DetectionBottomRightPadding, ImagePermute, StandardizeImage)

    model = models.get("yolo_nas_s", num_classes=len(CLASS_NAMES),
                       pretrained_weights=None)
    model.eval()

    import torch
    sd = torch.load(str(weights), map_location="cpu", weights_only=True)["net"]
    model.load_state_dict(sd, strict=True)

    model.set_dataset_processing_params(
        class_names=CLASS_NAMES,
        image_processor=ComposeProcessing([
            ReverseImageChannels(),
            DetectionLongestMaxSizeRescale(output_shape=(imgsz, imgsz)),
            DetectionBottomRightPadding(output_shape=(imgsz, imgsz), pad_value=114),
            ImagePermute(permutation=(2, 0, 1)),
            StandardizeImage(max_value=255.0),
        ]),
        conf=CONF,
        iou=iou,
    )
    return model


def fit_circle(img: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    """Fit the real log end inside a detection box with a Hough circle.

    Returns (cx, cy, r) in full-image coordinates, or the inscribed circle of
    the box when no plausible circle is found.
    """
    pad = 4
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = x1 + pad, y1 + pad
    roi = img[cy0:cy1, cx0:cx1]
    bw, bh = x1 - x0, y1 - y0
    fallback = ((x0 + x1) // 2, (y0 + y1) // 2, max(1, min(bw, bh) // 2))
    if roi.size == 0:
        return fallback

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    min_r = max(3, int(min(bw, bh) * 0.30))
    max_r = max(min_r + 2, int(max(bw, bh) * 0.75))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(min_r, 10),
        param1=120, param2=28, minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return fallback

    bcx, bcy = bw / 2, bh / 2
    target_r = (bw + bh) / 4
    best, best_score = None, None
    for cx, cy, r in np.round(circles[0]).astype(int):
        score = (abs(cx - bcx) + abs(cy - bcy)) / max(bw, bh) \
            + abs(r - target_r) / max(target_r, 1)
        if best_score is None or score < best_score:
            best, best_score = (cx + cx0, cy + cy0, r), score
    return best or fallback


def detect(model, img_bgr: np.ndarray, conf: float) -> list[dict]:
    """Run one BGR image through the model. Returns a list of box dicts."""
    pred = model.predict(img_bgr, conf=conf)
    p = pred.prediction
    boxes = []
    if p.bboxes_xyxy is None or len(p.bboxes_xyxy) == 0:
        return boxes
    for (x0, y0, x1, y1), c in zip(p.bboxes_xyxy, p.confidence):
        boxes.append({"x0": float(x0), "y0": float(y0),
                      "x1": float(x1), "y1": float(y1), "conf": float(c)})
    return boxes


def draw_circles(img, boxes: list[dict]) -> int:
    """Draw one fitted circle per detection. Returns the number drawn."""
    h, w = img.shape[:2]
    drawn = 0
    for b in boxes:
        x0 = max(0, int(round(b["x0"])))
        y0 = max(0, int(round(b["y0"])))
        x1 = min(w - 1, int(round(b["x1"])))
        y1 = min(h - 1, int(round(b["y1"])))
        if x1 <= x0 or y1 <= y0:
            continue
        cx, cy, r = fit_circle(img, x0, y0, x1, y1)
        cv2.circle(img, (cx, cy), max(1, r), COLOR, LINE_THICKNESS)
        drawn += 1
    return drawn


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Detect logs with YOLO-NAS and save images with circles.")
    ap.add_argument("source", nargs="?", default=str(DEFAULT_SOURCE),
                    help=f"image file or folder (default: {DEFAULT_SOURCE.name}/)")
    ap.add_argument("--out", default=str(DEFAULT_OUTPUT),
                    help=f"output folder (default: {DEFAULT_OUTPUT.name}/)")
    ap.add_argument("--weights", default=str(WEIGHTS),
                    help=f"model weights (default: {WEIGHTS.name})")
    ap.add_argument("--conf", type=float, default=CONF,
                    help=f"confidence threshold (default {CONF})")
    ap.add_argument("--keep", action="store_true",
                    help="keep existing files in the output folder")
    args = ap.parse_args()

    src = Path(args.source)
    if src.is_file():
        images = [src]
    elif src.is_dir():
        images = sorted(p for p in src.iterdir()
                        if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        log(f"error: source not found: {src}")
        return 2
    if not images:
        log(f"error: no images in {src}")
        return 2

    weights = Path(args.weights)
    if not weights.is_file():
        log(f"error: weights not found: {weights}")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"weights : {weights}")
    log(f"source  : {src}  ({len(images)} image(s))")
    log(f"output  : {out_dir}   conf: {args.conf}")
    if not args.keep:
        log(f"cleared : {clear_outputs(out_dir)} old file(s)")
    log("")

    model = build_model(weights, IMGSZ, IOU)
    log("model loaded (YOLO-NAS-S, 1 class: logs)\n")

    records = []
    for i, img_path in enumerate(images, 1):
        rec = {"image": img_path.name, "ok": False, "error": None,
               "n_detections": 0, "best_conf": None}
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("could not read image")
            boxes = detect(model, img, conf=args.conf)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            records.append(rec)
            log(f"[{i}/{len(images)}] {img_path.name:40s} FAILED  {rec['error']}")
            continue

        rec["n_detections"] = len(boxes)
        if boxes:
            rec["best_conf"] = max(b["conf"] for b in boxes)
        draw_circles(img, boxes)
        cv2.imwrite(str(out_dir / img_path.name), img)
        rec["ok"] = True
        records.append(rec)

        best = f"{rec['best_conf']:.2f}" if rec["best_conf"] is not None else "-"
        log(f"[{i}/{len(images)}] {img_path.name:38s} "
            f"{rec['n_detections']:3d} found  best={best}")

    ok = sum(1 for r in records if r["ok"])
    total = sum(r["n_detections"] for r in records)
    summary = {
        "weights": str(weights),
        "model": "yolo_nas_s",
        "conf": args.conf,
        "iou": IOU,
        "imgsz": IMGSZ,
        "images": len(records),
        "images_ok": ok,
        "total_detections": total,
        "records": records,
    }
    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("")
    log(f"{ok}/{len(records)} images processed, {total} detection(s) total")
    log(f"annotated images -> {out_dir}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
