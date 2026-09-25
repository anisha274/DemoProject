# Project State

## Environment
- Interpreter: Python 3.12 (via uv) in `.venv`
- Package versions: Installed via pip, using `torch` index `cu124`

## Implemented Files
- `pyproject.toml`
- `configs/pilot.yaml`
- `src/traffic_pipeline/*.py` (CLI, config, io_schema, video_prep, fusion, temporal)
- `src/traffic_pipeline/model_adapters/*.py` (detector, depth, segmentation)
- `tests/test_core.py`

## Next Steps
- Run tests and validate small synthetic inputs.
- Run the 8-frame smoke test.
- Run the full pilot clip.
