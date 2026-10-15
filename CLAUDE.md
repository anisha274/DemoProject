# Warehouse robot: perception and passage decisions

## Project goal
- Target: a warehouse robot running this full pipeline inside a Unity
  emulator (aisles between racks, workers, pallets, a carried load).
- THUD++ (Unity-made) is used for development and ground-truth
  validation of the perception steps, not as the final setting.
- Safety logic must depend only on reliable floor and person
  recognition; all other objects are obstacles by geometry.

## Standing rules (apply to every task)
- Code must run on Windows 11 and macOS. Use pathlib. Read all paths
  from config files. Never hard-code drive letters or C: paths. Run
  outputs store paths relative to configured roots.
- Never copy datasets into the repo. Datasets, model weights, outputs
  and CLAUDE.local.md stay git-ignored.
- Do not modify my friend's gesture module (folder: <FILL IN PATH>).
  Connect to it only through an adapter.
- Do not rerun successful inference. Never overwrite existing output
  folders; each run writes a new timestamped folder under outputs/.
- Reference (ground-truth) data is read only by evaluation code.
  Inference code must never import reference loaders.
- All thresholds live in config and are fixed before evaluation.
  Never tune on evaluation results. Disclose any rule changed after
  seeing results.
- Report every accuracy figure together with its coverage.
- Prefer unknown / insufficient_history / not_evaluated over guesses.
- Label results honestly: "reference-depth baseline",
  "reference-camera-pose baseline", "camera_proxy", "proxy_route".
- Save the exact config and git commit hash in every output folder.
- Commit locally at checkpoints. Never push.
- Finish every task with: exact commands for Windows PowerShell and
  macOS Terminal, output paths, checks completed, remaining
  limitations. Then stop and wait for review.

## Verified facts (see README for evidence)
- THUD++ synthetic images: 730x530, 60.24 Hz; documented camera
  height 1.2 m, pitch 0.
- Intrinsics: fx = fy = 458.99, cx = 365, cy = 265 (vertical FOV
  60.0 deg, Unity default). Recorded camera_intrinsic is not in pixel
  units and is not used.
- Depth: planar (forward-axis); fitted far plane 9.966 m (~10 m);
  saturated pixels are invalid.
- Pose: quaternion [x, y, z, w]; world = R * p_cam + t; y is up,
  x-z is the ground plane.
- Device selection: CUDA, then MPS, then CPU
  (check_environment.select_device).

## Project status
- Done: segmentation baseline, YOLO + ByteTrack, Step 4 fusion,
  Step 8 motion (frozen: configs/motion_frozen.json, hash
  15821ff9...1625).
- Splits (enforced by data_splits.json and the guard): Office/dynamic
  Capture_1 development, Capture_2 adjustment (used), Captures 3-4
  final; Gym/dynamic Captures 2-6 cross-scene final; Gym Capture_1
  excluded (prior use in Stage 3).
- Decided, not yet implemented:
  - Step 9: decisions use only motion_trusted (dynamic_classes:
    person); no trusted motion near the route -> stop; motion
    operating range 0-6 m as a decision-config setting.
  - Step 6: a map cell is free only if labelled floor AND within a
    height tolerance of the fitted floor plane; unseen cells count as
    blocked.
  - Load size comes from config ("configured, not measured").
- Next, in order: Step 9 decision rules; Step 6 passage width; slimmed
  evaluation harness; one final THUD++ evaluation. Unity warehouse
  scene is built by me in parallel.
