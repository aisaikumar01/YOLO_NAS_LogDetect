# YOLO_NAS_LogDetect

Local log detection with a trained **YOLO-NAS-S** model (SuperGradients 3.7.1).
Given a folder of photos of a log stack, it finds the individual logs and saves
each image with a green circle fitted to every log end. No labels, no text.
Everything runs locally — no API key, no network, no per-image cost.

## About the weights

`weights_logs_nas_clean.pt` is a **clean checkpoint**: it contains only the
network state dict (`{"net": ...}`). All training metadata was stripped —
epoch, metrics, optimizer state, EMA copy, preprocessing params and the
recorded package list (where third-party tooling such as Roboflow appeared in
the original file). A byte-level scan confirms nothing remains.

The model has a **single class: `logs`**, trained at 640×640. Last recorded
validation: mAP@0.50 ≈ 0.786, Precision ≈ 0.95, Recall ≈ 0.77.

> Note on commercial use: the architecture and pretrained backbone come from
> SuperGradients (Deci AI), released under the Apache 2.0 license. This is not
> legal advice — confirm the license terms for your intended use.

## Requirements

- Python 3.13 with `super_gradients` 3.7.1 installed `--no-deps` plus the
  curated set in `requirements.txt` (the official pins have no Python 3.13
  wheels):

```
pip install super-gradients==3.7.1 --no-deps
pip install -r requirements.txt
```

CPU inference is fine — a couple of seconds per photo at 640 px.

## Usage

Put your photos in `test_images/`, then from this folder:

```
python predict.py
```

Reads every image in `test_images/`, clears `predicted/`, and writes one
annotated image per input plus `_summary.json`.

```
python predict.py --conf 0.5             # stricter: fewer, more certain circles
python predict.py --keep                 # do not clear predicted/ first
python predict.py path\to\photo.jpg      # a single image
python predict.py path\to\folder         # a different folder
python predict.py --weights other.pt     # a different model
```

## Files

| Path | Purpose |
|---|---|
| `predict.py` | build model, load clean weights, detect, draw circles, save |
| `weights_logs_nas_clean.pt` | clean trained weights, 1 class (`logs`) |
| `test_images/` | input photos |
| `predicted/` | generated output (not tracked) |

## Notes

- Preprocessing (BGR→RGB, 640 longest-side rescale, pad 114, ÷255) is rebuilt
  in `build_model()` because the original processing params were stripped with
  the metadata.
- Each circle is fitted to the real log end with a Hough circle search inside
  the detection box; it falls back to the inscribed box circle when no
  plausible circle is found.
- `super_gradients` rebinds `sys.stdout` at import, so output goes through
  `sys.__stdout__`.
