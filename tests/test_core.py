import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import tempfile
import os

from traffic_pipeline.io_schema import FUSION_SCHEMA, TEMPORAL_SCHEMA, SCHEMA_VERSION
from traffic_pipeline.fusion import fuse_frame, rows_to_table
from traffic_pipeline.temporal import add_temporal_features

def test_schema_roundtrip():
    row = {
        "schema_version": SCHEMA_VERSION,
        "sequence_id": "test_001",
        "frame_id": 0,
        "track_id": 1,
        "class_name": "car",
        "video_time_s": 0.0,
        "timebase_verified": False,
        "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0,
        "center_u": 5.0, "center_v": 5.0,
        "confidence": 0.9,
        "relative_depth_median": None,
        "relative_depth_iqr": None,
        "depth_valid_fraction": None,
        "context": "unknown",
        "context_support": 0.0,
        "quality_flags": "valid",
        "depth_m": None, "camera_x_m": None, "camera_y_m": None, "camera_z_m": None,
        "world_x_m": None, "world_y_m": None, "world_z_m": None,
        "metric_status": "unsupported"
    }
    tb = rows_to_table([row], FUSION_SCHEMA)
    
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        pq.write_table(tb, tmp.name)
        tb_read = pq.read_table(tmp.name)
        
    os.remove(tmp.name)
    assert tb_read.schema == FUSION_SCHEMA
    read_row = tb_read.to_pylist()[0]
    assert read_row["schema_version"] == "1.0"
    assert read_row["depth_m"] is None

def test_empty_table():
    tb = rows_to_table([], FUSION_SCHEMA)
    assert tb.num_rows == 0
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        pq.write_table(tb, tmp.name)
        tb_read = pq.read_table(tmp.name)
    os.remove(tmp.name)
    assert tb_read.num_rows == 0

def test_fusion_synthetic():
    frame_meta = {"frame_id": 0, "video_time_s": 0.0, "timebase_verified": False}
    objs = [{"track_id": 1, "class_name": "car", "confidence": 0.9, "x1": 10, "y1": 10, "x2": 20, "y2": 20}]
    depth_map = np.ones((50, 50), dtype=np.float32)
    context_mask = np.zeros((50, 50), dtype=np.uint8)
    
    rows = fuse_frame(frame_meta, objs, depth_map, context_mask, "seq1")
    assert len(rows) == 1
    assert rows[0]["relative_depth_median"] == 1.0
    assert rows[0]["context"] == "unknown"

def test_temporal_gaps():
    rows = [
        {"track_id": 1, "frame_id": 0, "center_u": 0.0, "center_v": 0.0, "video_time_s": 0.0, "context": "road"},
        {"track_id": 1, "frame_id": 1, "center_u": 10.0, "center_v": 0.0, "video_time_s": 1.0, "context": "road"},
        # Gap > 2 (max_gap_frames = 2)
        {"track_id": 1, "frame_id": 5, "center_u": 50.0, "center_v": 0.0, "video_time_s": 5.0, "context": "road"},
    ]
    # Add dummy required fields
    for r in rows:
        for f in FUSION_SCHEMA:
            if f.name not in r:
                r[f.name] = None
    
    tb = rows_to_table(rows, FUSION_SCHEMA)
    res = add_temporal_features(tb, max_gap_frames=2)
    res_rows = res.to_pylist()
    
    # First frame age=1, speed=None
    assert res_rows[0]["track_age_frames"] == 1
    assert res_rows[0]["du_px_per_frame"] is None
    
    # Second frame age=2, speed=10
    assert res_rows[1]["track_age_frames"] == 2
    assert res_rows[1]["du_px_per_frame"] == 10.0
    
    # Third frame age=1 (reset), speed=None
    assert res_rows[2]["track_age_frames"] == 1
    assert res_rows[2]["du_px_per_frame"] is None

