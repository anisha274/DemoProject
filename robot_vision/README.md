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
| `configs/tracking_office.json` | Stage 2: detector weights, thresholds, image size, tracker YAML, overlay steps. |
| `check_camera_streams.py` | Compares the two camera streams that share RGB basenames; checks RGB/label alignment. |
| `track_objects.py` | Stage 2: YOLO11n + ByteTrack on a segmentation run's frames, with box-level segmentation summaries. |
| `verify_tracking_run.py` | Integrity checks recomputed from a tracking run's files (not an accuracy evaluation). |
| `configs/depth_thud_synthetic.json` | Validated synthetic depth encoding and pixel intrinsics, with the evidence summary. |
| `thud_depth.py` | Depth decoding (raw → metres, saturation → invalid), intrinsics from FOV, back-projection. |
| `validate_depth.py` | Fits and tests the depth/intrinsics against dataset 3D boxes on several captures. |
| `tests/test_thud_depth.py` | Unit tests for `thud_depth.py`, plus one dataset check (`python -m unittest discover -s tests`). |
| `fuse_objects.py` | Step 4: per-tracked-object 3D position (camera and world frame) and floor context from depth + pose + predicted masks + tracks, with geometry checks and evaluation. |
| `configs/fusion_office.json` | Inputs, pixel-selection rules, floor-context rule and evaluation settings for `fuse_objects.py`. |
| `configs/robot.json` | Provisional robot width (0.60 m) and safety margin per side (0.10 m); load dimensions to come from Unity. |
| `tests/test_fuse_objects.py` | Unit tests for the pose convention, pixel selection and point-to-box distance. |
| `report_step4.py` | Step 4 report supplement: calibration used, fitted parameters, all-class table, unmatched-couch classification, timing and step sizes. Reads a fusion run; never modifies it. |
| `configs/data_splits.json` | Office/dynamic split: Capture_1 development, Capture_2 adjustment, Capture_3/4 final evaluation only. |

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
numpy 2.5.2, pillow 12.3.0, matplotlib 3.11.2; stage 2 adds ultralytics 8.4.175, lap 0.5.13, scipy 1.18.1,
opencv-python 5.0.0.93. Every run records the versions it used in
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

## Camera-stream provenance

`check_camera_streams.py` writes `outputs/camera_check_<date>_<time>/camera_stream_report.json` and
`alignment_step_*.png` images (semantic boundaries in red over the RGB frame).

Confirmed for Office/dynamic/Capture_1:
- `camera_0` and `camera` have 1490 common steps with identical sequence ID, timestamp, translation, rotation,
  intrinsics, projection, ego record and RGB basename at every step. The metadata names a different source
  RGB folder for each sensor.
- Only `camera_0` has semantic, instance and 2D box annotations; `camera` has only 3D boxes.
- At steps 0, 150 and 299 the on-disk RGB frames line up with the `camera_0` semantic labels: the edge-based
  shift search peaks at 0 or 1 px, within the method's resolution. `camera_0` instance masks match its
  2D boxes exactly (median IoU 1.0).

Still ambiguous:
- Which sensor rendered each on-disk RGB file cannot be proven: the folder is flattened, the basenames
  are shared, and both sensors have the same pose, so their renders would look the same.
- `check_camera_streams.py` projects 3D boxes with the *recorded* intrinsics, which are not pixel units (see
  the depth section). With the validated 60° intrinsics, `camera`'s 3D boxes project onto `camera_0`'s 2D boxes
  (median IoU about 0.72) and contain `camera_0`'s depth-mask points. That confirms shared geometry, not
  which sensor rendered each RGB file.

## Stage 2: 2D detection, tracking and segmentation association

```powershell
cd E:\RobotVisionWorkspace\robot_vision
.\.venv\Scripts\python.exe check_camera_streams.py --config configs\office.json --steps 0 150 299
.\.venv\Scripts\python.exe track_objects.py --segmentation-run outputs\seg_Office_dynamic_Capture_1_300f_20261010_032646 --frames 3 --device cuda
.\.venv\Scripts\python.exe track_objects.py --segmentation-run outputs\seg_Office_dynamic_Capture_1_300f_20261010_032646 --device cuda
.\.venv\Scripts\python.exe verify_tracking_run.py outputs\track_300f_<date>_<time>
```

- **Detector**: Ultralytics YOLO11n (`yolo11n.pt`, COCO classes), downloaded on first use to
  `robot_vision/models/` (ignored by Git). The SHA-256 is recorded in `tracking_summary.json`.
- **Tracker**: ByteTrack via `tracker="bytetrack.yaml"`, the Ultralytics default file. Its exact text is copied
  to `tracker_config.yaml` in every run. Detector inference runs on the selected device; the ByteTrack update
  runs on the CPU.
- **Baseline settings** (`configs/tracking_office.json`, or `--confidence`, `--nms-iou`, `--image-size`):
  confidence 0.10, NMS IoU 0.70, image size 640, batch size 1, all COCO classes.
- **Frames**: taken from the segmentation run's `manifest.jsonl` in its chronological order. Each frame's
  identifiers, timestamp and RGB file are re-checked against the dataset metadata, and segmentation masks
  are joined on `(sequence_id, sensor_id, step)`. Tracker state persists within a sequence and is reset at
  each new sequence and each new run (track IDs restart).
- Only tracks that ByteTrack reports are recorded. Detections that never become a confirmed track are not
  saved. Track IDs are not contiguous, because IDs are also allocated to tentative tracks.

### Tracking outputs (`outputs/track_<N>f_<date>_<time>/`)

| File | Content |
|---|---|
| `tracks.csv` | One row per tracked observation: sequence, camera, step, timestamp, track ID, COCO class ID/name, confidence, box `x1,y1,x2,y2` in original image pixels, validity/clipping flags, image size |
| `frames.jsonl` | One line per processed frame, including frames with no tracked objects |
| `object_segmentation_associations.csv` | Per observation: detector label and confidence, the pixel region used, and the share of each predicted segmentation class inside the box |
| `tracking_summary.json` | Counts, per-track class history, class-change warnings, integrity checks, settings, versions, weight hash, timings |
| `overlays/step_*.png` | Predicted segmentation, boxes, IDs, labels and timestamp (steps 0, 150, 299; all frames for runs of 10 frames or fewer) |
| `preview.mp4` | The same annotation for every frame (MPEG-4 Part 2, `mp4v`) |
| `tracker_config.yaml`, `run_status.json`, `COMPLETE` | Exact tracker configuration and run status |

**Segmentation proportions are bounding-box summaries.** The box is rounded outward to whole pixels and
clipped to the image. Background and overlapping objects inside the box contaminate the shares, so they
are neither instance masks nor a measure of object extent. The detector label is stored separately and is
not changed by them. Reference masks and reference object IDs are not used.

**Track IDs are not object counts.** One real object can receive several IDs (after occlusion or missed
detections), and one ID can drift between objects. A track's class can change between frames; such tracks
are listed under `class_change_warnings`.

**Video timing**: the playback rate is 1 / median timestamp difference (≈ 60.24 fps for this capture), and each
frame shows its recorded timestamp. The video is for viewing only. Use the timestamps in `frames.jsonl`,
never the video frame rate, for any motion measurement.

`verify_tracking_run.py` checks frame count and order against the manifest, the identifier join,
timestamps, box bounds, image sizes, duplicate `(sequence, camera, step, track ID)` observations, the
association arithmetic on a sample of boxes, and the video frame count. These are integrity checks, not a
detection or identity-accuracy evaluation; that is a separate future step.

## Stage 3: synthetic depth encoding and camera intrinsics (validated)

```powershell
cd E:\RobotVisionWorkspace\robot_vision
.\.venv\Scripts\python.exe validate_depth.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result, recorded in `configs/depth_thud_synthetic.json` and used by `thud_depth.py`:

- **Depth**: `depth_m = raw / 65535 × 10.0`. The value is planar depth along the optical axis (Z), not distance
  along the pixel ray. `raw == 65535` means "10 m or more" and is treated as invalid. **Not millimetres.**
- **Intrinsics**: square pixels, vertical field of view 60°: `fx = fy = 265 / tan(30°) ≈ 459.0`, `cx = 365`,
  `cy = 265` for 730×530. The `camera_intrinsic` values in the metadata (1013.89 / 1520.83) are **not** pixel
  units; don't use them.

How it was established (`validate_depth.py`):
- **Evidence.** In each frame one sensor has instance masks and the other has 3D boxes in metres, with the
  same pose and timestamp. Each object's masked depth pixels are back-projected to 3D and tested against
  its 3D box.
- **Fit, then held-out test.** Far plane and field of view were fitted on Office/dynamic/Capture_1
  (9.97 m, 60.3°). The round values 10 m / 60° fit tighter and were tested unchanged on Office/dynamic/Capture_2,
  Office/static/Capture_1, Gym/dynamic/Capture_1 and Gym/static/Capture_1. The Gym scenes have the sensor
  roles swapped; the script detects the roles from the annotations.
- **Error estimate.** Chairs are 100% inside their boxes at 0 cm tolerance, desks 100% within 0.5 cm, and all
  Gym classes 100% within 2 cm. So combined depth + intrinsics error is within about 2 cm over roughly
  0.5–10 m, judged against Unity's boxes.
- **Rejected alternatives**: millimetres (0% inside), ray distance (25–42%), recorded intrinsics (6–16%).
  Projecting 3D boxes with the 60° intrinsics matches the 2D boxes (median IoU 0.71–0.82) vs about 0 with the
  recorded ones.
- **Known exception**: Office sofa boxes enclose only part of the sofa (about 44% inside). The overhang is
  one-sided and constant across hundreds of viewpoints, which points to an incomplete box annotation, not
  a depth error.

Outputs: `outputs/depth_check_<date>_<time>/depth_validation_report.json` and `depth_vs_3d_boxes.png`.

## Data splits (Office/dynamic)

Recorded in `configs/data_splits.json` and copied into every report.

| Capture | Role | Earlier use |
|---|---|---|
| Capture_1 | development | Stages 1–4; Stage 3 depth/FOV fit |
| Capture_2 | adjustment: the only capture on which a parameter may be adjusted | Stage 3 held-out depth test only |
| Capture_3, Capture_4 | final evaluation: one run with a frozen config | not used |

## Stage 4: tracked-object 3D fusion (reference-depth baseline)

```powershell
.\.venv\Scripts\python.exe fuse_objects.py --check-only   # geometry checks + steps 0, 150, 299
.\.venv\Scripts\python.exe fuse_objects.py                # checks again, then all 300 tracked frames
```

**Depth source.** THUD reference depth (Unity render). Every result here is a *reference-depth fusion
baseline*; it says nothing about the accuracy of an RGB-only depth model.

**Prediction inputs, unchanged.** The 300-frame tracking run (YOLO11n + ByteTrack) and the SegFormer run it
references: boxes, track IDs, steps, timestamps and predicted grouped masks. Reference masks, boxes and object
positions are read only by the geometry checks and the evaluation.

**Geometry checks** (`geometry_checks.json`; the run stops before producing any positions if one fails):
- Tracked frames belong to the configured Office capture. RGB file name, step, timestamp, sequence and capture
  ID match the metadata. The tracked sensor is the labelled RGB sensor (`camera_0`).
- Depth, RGB and predicted mask have the same size. Depth edges coincide with RGB edges at zero shift.
- Depth encoding re-confirmed per frame: reference-instance points fall inside the 3D boxes when read as planar
  Z, but not when read as distance along the ray.
- Pose convention: `world = R(q) p_cam + t`, with `q` stored as `[x, y, z, w]`. Ego pose is identity, and both
  sensors share the pose. Static 3D boxes stay fixed in the world under this convention, and move by metres
  under the transposed one.

**Calibration actually used** (recorded in each run's `run_info.json`):
- Intrinsics: `fx = fy = 458.9935`, `cx = 365.0`, `cy = 265.0` for 730×530 px. Source: square-pixel pinhole from a
  60° vertical field of view, principal point at the image centre: `fy = (H/2) / tan(vfov/2)`, `fx = fy`,
  `cx = W/2`, `cy = H/2` (`configs/depth_thud_synthetic.json`, `thud_depth.pixel_intrinsics`).
- Back-projection of pixel (column `u`, row `v`; pixel centre at index + 0.5) with planar depth `Z`:
  `x = (u + 0.5 − cx)·Z / fx`, `y = −(v + 0.5 − cy)·Z / fy`, `z = Z`.
- World transform: `p_world = R(q)·p_cam + t`, with `q` the sensor rotation stored as `[x, y, z, w]` and `t` the
  sensor translation in metres. Ego pose is identity in this capture.
- Depth: `depth_m = raw / 65535 × 10.0`, planar Z along the optical axis. `raw = 65535` is invalid (at or beyond
  10 m). Evidence: reference-instance points fall inside reference 3D boxes (chairs 100% at 0 cm, desks 100%
  within 0.5 cm, Gym classes 100% within 2 cm). Millimetres put 0% inside, ray distance 25–42%.
- The metadata `camera_intrinsic` (1013.89 / 1520.83) was rejected because it is not in pixel units.
  Back-projecting with it puts only 6–16% of points inside their boxes. Projecting 3D boxes with it gives
  median IoU 0 against the 2D boxes, versus 0.72–0.82 with the 60° model.

**Parameters fitted on reference boxes.** Two were: the depth far plane (fitted 9.966 m, used 10.0) and the
vertical FOV (fitted 60.27°, used 60.0).
- The fit used **Office/dynamic/Capture_1**, the same capture Step 4 is evaluated on.
- The rounded values were then confirmed unchanged on four held-out captures: Office/dynamic/Capture_2,
  Office/static/Capture_1, and Gym dynamic and static. That is the independent evidence for the calibration.
- Step 4's inside-3D-box rates on Capture_1 are therefore **not** independent of the calibration.
- Nothing else was fitted. Pixel-selection rules, thresholds and the association IoU were set before the full
  run and were not tuned on reference data.

**What the checks do and do not show.**
- The static-box check verifies the pose convention: 3D boxes stay fixed to about 1 µm under `R(q)·p + t` and
  move by metres under the transpose. It does **not** show micrometre-level perception accuracy.
- Stored axes: `cam_x/cam_y/cam_z` are camera-frame metres (x right, y up, z forward; Unity, left-handed).
  `world_x/world_y/world_z` are Unity world metres with **y up**, so the ground plane is world x–z.
- With rendered reference depth, the estimate and the reference point come from the same depth map. So
  **visible-point error measures pixel-selection / segmentation error, not depth error.**

**What a position is.** For each tracked box, pixels are chosen from the *predicted* mask:
1. the detector's own class (person, chair, couch→sofa);
2. otherwise box pixels that are not wall, floor or ceiling;
3. otherwise the box is flagged invalid.

The estimate is the per-axis median of those pixels back-projected with depth. That is a point on the visible
object surface, not a 3D box centre. `object_states.csv` gives it in camera coordinates (x right, y up, z
forward) and world coordinates, plus validity flags. `floor_context` says whether predicted floor touches the
box's bottom edge. `bottom_height_above_floor_m` uses the floor height measured from predicted floor pixels.

**Evaluation** (`evaluation_report.json`, `association_evaluation.csv`):
- *Association.* Each frame, tracked boxes are matched to reference 2D boxes with a class-compatible label and
  IoU >= 0.5 (Hungarian).
- *Visible-point error.* Distance to the median of the reference instance mask back-projected with the same
  depth. Because the depth is the same, this isolates pixel-selection error.
- *3D-box test.* Only Chair, Desk and Sofa have 3D boxes in this capture; **people have none**.
- *World spread.* Spread of a static object's track position, compared with the same spread for the
  reference visible point.

Run `fuse_300f_20261011_042733` (300 frames, development capture):

| Summary | Result |
|---|---|
| Tracked observations / valid 3D positions | 1903 / 1793 |
| Associated with a reference object (valid position) | 1380 |
| Visible-point error, all classes: median / p90 | 0.07 m / 0.41 m |
| Static-object track spread in world: estimate vs reference visible point | 0.20 m vs 0.06 m |

Every associated class (rows sum to 1380):

| Reference label | Associated (valid) | Visible-point error n | median | p90 | With 3D box | Inside 3D box (+5 cm) |
|---|---|---|---|---|---|---|
| people | 546 | 546 | 0.023 m | 0.201 m | 0 | — |
| Chair | 379 | 369 | 0.129 m | 0.663 m | 369 | 94% |
| Computer | 328 | 328 | 0.063 m | 0.188 m | 0 | — |
| Sofa | 98 | 98 | 0.398 m | 0.602 m | 98 | 66% |
| Plants | 22 | 22 | 0.113 m | 0.138 m | 0 | — |
| Book | 7 | 7 | 0.053 m | 0.060 m | 0 | — |

Chair has 10 associated observations with no visible-point error, because their reference instance had
fewer than 20 valid-depth pixels.

**The 263 unmatched couch boxes** (10 tracks) are classified by the reference 2D box, of any label, that they
overlap most:

| Category | Boxes |
|---|---|
| Same label (Sofa), IoU < 0.5 (median 0.42) | 185 |
| Another label: Chair (41 of these with IoU ≥ 0.5, i.e. the detector called a chair a couch) | 48 |
| Same label, IoU ≥ 0.5, but that sofa was already matched to another couch track (duplicate box) | 30 |
| No overlap with any reference box | 0 |

**Timing and frame-to-frame world step.**
- Timing: 300 frames, median dt 0.0166 s (60.24 Hz), constant.
- Ground-plane (x–z) step between consecutive valid positions of one track:

| Group | Step pairs | Median | p90 | Max |
|---|---|---|---|---|
| People | 570 | 0.016 m | 0.026 m | 3.25 m |
| Static (Chair / Desk / Sofa) | 768 | 0.025 m | 0.234 m | 5.38 m |

- For static objects the true step is 0, so their step size is pure pixel-selection jitter. It is larger than a
  walking person's step, which is why Step 8 estimates velocity over a time window, not from adjacent frames.
- Groups come from each track's most frequent associated reference label, used for evaluation only.

Failure modes seen:
- The box-minus-background fallback can pick an object behind or in front of the target. This causes the
  multi-metre outliers.
- Predicted sofa pixels cover only part of the sofa.

The selection rules were fixed before the full run and were not tuned on these results.

Supplement (does not modify the fusion run):

```powershell
.\.venv\Scripts\python.exe report_step4.py --fusion-run outputs\fuse_300f_20261011_042733 --depth-validation-run outputs\depth_check_20261011_031941
```

It writes `outputs/step4_supplement_<date>_<time>/`:
- `step4_supplement.json` and `step4_report.md`: calibration, fitted parameters, per-class table, couch
  classification, timing.
- `couch_unmatched.csv`.

## Known limitations

- Depth decoding is validated only for the THUD **synthetic** captures above. Real-scene depth and any
  monocular or relative depth model are not validated. No distance, speed, passage width or clearance is
  computed yet.
- Tracking works in image pixels only. No metric distance, speed, 3D clearance or movement decision is
  computed.
- Semantic labels are used only as evaluation references, never as model input.
- Real-scene (`Real_Scenes/store`) captures use a different layout and are not supported yet.
