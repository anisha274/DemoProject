import pyarrow as pa
from collections import defaultdict
import math
from .io_schema import TEMPORAL_SCHEMA

def add_temporal_features(fusion_table: pa.Table, max_gap_frames: int):
    # fusion_table is sorted by frame_id conceptually, but let's group by track_id
    rows = fusion_table.to_pylist()
    
    # Sort by track_id then frame_id
    rows.sort(key=lambda r: (r["track_id"], r["frame_id"]))
    
    track_histories = defaultdict(list)
    result_rows = []
    
    for row in rows:
        tid = row["track_id"]
        fid = row["frame_id"]
        
        hist = track_histories[tid]
        
        # Check gap
        if hist and (fid - hist[-1]["frame_id"] > max_gap_frames):
            hist.clear()
            
        hist.append(row)
        
        # Compute features
        track_age = len(hist)
        
        du_px_frame = None
        dv_px_frame = None
        speed_px_frame = None
        du_px_s = None
        dv_px_s = None
        context_changed = False
        
        if track_age >= 2:
            prev = hist[-2]
            d_frames = fid - prev["frame_id"]
            if d_frames > 0:
                du_px_frame = (row["center_u"] - prev["center_u"]) / d_frames
                dv_px_frame = (row["center_v"] - prev["center_v"]) / d_frames
                speed_px_frame = math.hypot(du_px_frame, dv_px_frame)
                
                dt = None
                if row["video_time_s"] is not None and prev["video_time_s"] is not None:
                    dt = row["video_time_s"] - prev["video_time_s"]
                    if dt > 0:
                        du_px_s = (row["center_u"] - prev["center_u"]) / dt
                        dv_px_s = (row["center_v"] - prev["center_v"]) / dt
                        
            if row["context"] != prev["context"]:
                context_changed = True
                
        new_row = dict(row)
        new_row.update({
            "track_age_frames": track_age,
            "du_px_per_frame": du_px_frame,
            "dv_px_per_frame": dv_px_frame,
            "image_speed_px_per_frame": speed_px_frame,
            "du_px_per_video_s": du_px_s,
            "dv_px_per_video_s": dv_px_s,
            "context_changed": context_changed,
            "vx_mps": None,
            "vy_mps": None,
            "vz_mps": None,
            "speed_mps": None,
            "acceleration_mps2": None,
            "motion_status": "unsupported",
        })
        result_rows.append(new_row)
        
    # restore original order by frame_id then track_id
    result_rows.sort(key=lambda r: (r["frame_id"], r["track_id"]))
    
    return pa.Table.from_pylist(result_rows, schema=TEMPORAL_SCHEMA)
