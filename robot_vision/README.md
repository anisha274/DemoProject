# robot_vision — environment segmentation (stage 1)

First stage of the indoor service-robot vision system:
**THUD dataset loading → pretrained semantic segmentation → visualisations → evaluation.**

Depth, tracking, passage fitting, gestures and Unity integration are *not* part of this stage.
The teammate's gesture code lives in a separate repository
(https://github.com/anisha274/media_pipe_gesture) and is not used here.

## Files

| File | Purpose |
|---|---|
| `configs/office.json` | Dataset root, capture, sensor, model + revision, class lists and label mappings. Edit this (or pass CLI options) instead of the Python code. |
| `check_environment.py` | Prints Python/PyTorch/package versions, CUDA/MPS availability, `nvidia-smi`; selects a device and runs a small tensor test on it. |
| `thud_data.py` | Reads `captures_*.json` and the semantic palette, matches RGB ↔ semantic files via metadata, checks steps/timestamps, decodes semantic colours. |
| `audit_dataset.py` | Validates a capture and N frames; writes `audit_report.json`. |
| `run_segmentation.py` | SegFormer inference, label grouping, saved masks/visualisations, confusion matrix and evaluation report. |
| `requirements.txt` | Common dependencies (PyTorch is installed separately, per platform). |

## Setup

### Windows (tested: RTX 3060, driver 616.92, Python 3.12.14)

```powershell
cd E:\RobotVisionWorkspace\robot_vision
uv venv --python 3.12 .venv
# PyTorch with CUDA 13.0, command from the official selector at pytorch.org/get-started/locally
uv pip install --python .venv\Scripts\python.exe torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

(Without `uv`: create the venv with `py -3.12 -m venv .venv`, then use `.venv\Scripts\python.exe -m pip install ...` with the same arguments.)

### macOS

The official selector gives the default package for macOS (MPS on Apple silicon, CPU otherwise):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch torchvision
.venv/bin/python -m pip install -r requirements.txt
```

Point the code at the dataset without editing source, e.g.
`export THUD_DATASET_ROOT=/Volumes/Data/RGB-D` or `--dataset-root /Volumes/Data/RGB-D`.
Use `--device mps` (fails if MPS is missing) or `--device auto`.

### Tested versions

torch 2.14.1+cu130, torchvision 0.29.1+cu130, transformers 5.19.0, huggingface_hub 2.2.0,
numpy 2.5.2, pillow 12.3.0, matplotlib 3.11.2. Every run records the versions it used in
`evaluation_report.json`.

## Running (Windows PowerShell)

```powershell
cd E:\RobotVisionWorkspace\robot_vision
.\.venv\Scripts\python.exe check_environment.py --device cuda
.\.venv\Scripts\python.exe audit_dataset.py --config configs\office.json --frames 3
.\.venv\Scripts\python.exe run_segmentation.py --config configs\office.json --frames 3 --device cuda
.\.venv\Scripts\python.exe run_segmentation.py --config configs\office.json --frames 300 --device cuda
```

Useful options: `--start N` (first matched frame index), `--capture <path>` (relative to the dataset root or
absolute), `--dataset-root`, `--output-root`. `--device cuda|mps|cpu` never falls back to another device;
`--device auto` picks CUDA, then MPS, then CPU. Batch size is 1.

The model weights download once into the Hugging Face cache in your user profile
(`%USERPROFILE%\.cache\huggingface`), outside this repository.

## Dataset handling (THUD synthetic, Unity Perception format)

Checked on `Synthetic_Scenes/Office/dynamic/Capture_1`:

- `Label/captures_000.json … captures_019.json` hold 2980 records: 1490 for sensor `camera_0` and 1490 for
  sensor `camera`. Only `camera_0` records carry the `semantic segmentation` annotation, so `camera_0` is the
  RGB stream used (`rgb_sensor_id` in the config).
- Each record gives `sequence_id`, `step`, `timestamp`, the RGB file name and the semantic file name.
  Frames are matched through these metadata fields and ordered by step. Image numbers do **not** equal steps
  (step 0 → `rgb_2.png` / `segmentation_2.png`).
- The dataset copy on disk has flattened Unity's `RGB<uuid>/` and `SemanticSegmentation<uuid>/` folders into
  `RGB/` and `Label/Semantic/`, so files are found by basename. Both sensors name their RGB files
  identically and have identical poses, intrinsics and timestamps; the flattened folder cannot show which
  sensor rendered each file. This is reported as a warning in every audit/run.
- Duplicate steps or basenames, non-finite or non-increasing timestamps, missing files and too few frames
  stop the run. No FPS is assumed; only the recorded timestamps are used.

### Semantic colour decoding

The palette (`annotation_definitions.json`, 29 classes) stores colours as floats 0–1. Colour code =
`round(value × 255)` per RGBA channel. Pixels are matched exactly first; a pixel colour with no exact match
may match within ±1 per channel if exactly one class qualifies. Anything else (unmatched or ambiguous) gets
ignore ID **255**. In practice: `Bag` needs the ±1 tolerance (palette 176.51 → file value 176), and
pure-black `(0,0,0)` pixels match no class and are ignored.

## Classes and mappings

| ID | Project class | Model (ADE20K, resolved by name from `model.config.id2label`) | Dataset (THUD palette name) |
|---|---|---|---|
| 0 | other | every other model class | every other palette class: BookShelf, Bag, Desk, Table, Plants, Window, Curtain, Lamp, TV, laptop, PlantRack, Pot, Folder, Book, Box, Ceramics, cup, Computer, clock, Glass, Airconditioner, Beam, Spotlight |
| 1 | wall | wall (0) | wall |
| 2 | floor | floor (3) | Floor |
| 3 | ceiling | ceiling (5) | Ceiling |
| 4 | person | person (12) | people |
| 5 | chair | chair (19), armchair (30), swivel chair (75) | Chair |
| 6 | sofa | sofa (23) | Sofa |
| 255 | ignore | — | pixels whose colour cannot be decoded |

## Outputs

Each run creates a new folder `outputs/seg_<scene>_<split>_<capture>_<N>f_<date>_<time>/`:

| Path | Content |
|---|---|
| `pred_ade20k/step_XXXXXX.png` | Native model prediction, ADE20K class IDs 0–149 (8-bit, original size) |
| `pred_grouped/` | Project class IDs 0–6 |
| `ref_grouped/` | Reference labels as project IDs 0–6, 255 = ignored |
| `overlays/` | RGB blended with the grouped prediction colours |
| `floor_candidate/` | 255 where the model predicted floor, else 0 |
| `comparisons/` | RGB / reference / prediction side by side (all frames if ≤ 10, else first, middle, last) |
| `legend.png` | Colour legend |
| `confusion_matrix.csv/.png` | Rows = reference class, columns = predicted class, in pixels |
| `evaluation_report.json` | Metrics, settings, model revision, versions, device, timings, mappings |
| `manifest.jsonl` | One line per processed frame (metadata, file paths, timings, decoding stats), written as it goes |
| `run_info.json`, `run_status.json`, `COMPLETE` | Run settings and status. `COMPLETE` exists only after all frames and the evaluation finished |

A **floor-candidate mask** only shows pixels the model called floor. It is *not* a safe lane, verified free
space or a passable route; those need depth, obstacle geometry and robot dimensions.

## Reading the evaluation report

- **Confusion matrix**: for every valid reference pixel, counts (reference class, predicted class).
  The diagonal is correct pixels.
- **IoU** for a class = correct pixels ÷ (pixels that are that class in the reference **or** the prediction).
  1.0 is perfect, 0 means no overlap.
- **reference_pixels** (support): how many reference pixels belong to the class. Small support means a
  noisy IoU.
- **Pixel accuracy**: correct pixels ÷ all valid reference pixels, "other" included.
- **Mean IoU**: average IoU over wall, floor, ceiling, person, chair and sofa. "other" pixels stay in the
  matrix, so predicting e.g. wall on a bookshelf still lowers wall IoU. A class is left out of the mean only
  if its union is zero; the report lists the included classes.
- **Ignored pixels**: reference pixels whose colour could not be decoded; they are not scored.

These numbers come from consecutive frames of one development capture. They are **not** independent
held-out performance.

## Known limitations

- Depth is not used. The encoding of the synthetic 16-bit depth PNGs is not yet validated: do not assume
  millimetres, do not divide by 1000, and do not treat relative depth predictions as metric. No distance,
  speed, passage width or clearance is reported.
- Semantic labels are used only as evaluation references, never as model input.
- Real-scene (`Real_Scenes/store`) captures use a different layout and are not supported yet.
