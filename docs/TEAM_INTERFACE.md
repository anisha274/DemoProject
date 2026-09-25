# Team Interface

This pipeline supports dropping in saved outputs from teammates.

## Calibration File
JSON file containing camera intrinsic matrix `K` and distortion coefficients.

## Poses File
JSON list of `T_world_from_camera` 4x4 transform matrices for each frame.

## Acquisition Timebase File
Mapping of frame index to actual POSIX timestamp from the physical acquisition clock.

## Saved Outputs
To replace model adapters, simply place the corresponding outputs (`depth/`, `segmentation/`, `tracks.jsonl`) into the sequence output directory. The pipeline will skip running models and use the saved outputs during the `--resume` process if they already exist.
