import numpy as np
import pyarrow as pa
from .io_schema import FUSION_SCHEMA, SCHEMA_VERSION

def fuse_frame(frame_meta, objects, depth_map, context_mask, seq_id):
    rows = []
    
    H, W = depth_map.shape
    
    for obj in objects:
        x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
        # clip to image
        x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
        x2_c, y2_c = min(W, int(x2)), min(H, int(y2))
        
        # calculate depth
        depth_median, depth_iqr, valid_frac = None, None, None
        quality_flags = "valid"
        
        if x1_c < x2_c and y1_c < y2_c:
            roi_depth = depth_map[y1_c:y2_c, x1_c:x2_c]
            valid_pixels = roi_depth[roi_depth > 0] # strictly positive or just all? User: "Zero can be a valid relative value; do not apply a metric positive-depth mask to it."
            # Actually Marigold produces affine invariant depth. Let's use all finite pixels.
            valid_pixels = roi_depth[np.isfinite(roi_depth)]
            
            if valid_pixels.size > 0:
                depth_median = float(np.median(valid_pixels))
                depth_iqr = float(np.percentile(valid_pixels, 75) - np.percentile(valid_pixels, 25))
                valid_frac = float(valid_pixels.size / roi_depth.size)
            else:
                quality_flags = "invalid_depth_roi"
        else:
            quality_flags = "invalid_roi"
            
        # context support: small patch at/just below lower boundary
        support_y1 = min(H - 1, y2_c - 2)
        support_y2 = min(H, y2_c + 5)
        
        context_str = "unknown"
        support_val = 0.0
        
        if support_y1 < support_y2 and x1_c < x2_c:
            patch = context_mask[support_y1:support_y2, x1_c:x2_c]
            road_frac = float(np.mean(patch == 1))
            sidewalk_frac = float(np.mean(patch == 2))
            
            if road_frac >= 0.5:
                context_str = "road"
                support_val = road_frac
            elif sidewalk_frac >= 0.5:
                context_str = "sidewalk"
                support_val = sidewalk_frac
                
        row = {
            "schema_version": SCHEMA_VERSION,
            "sequence_id": seq_id,
            "frame_id": frame_meta["frame_id"],
            "track_id": obj["track_id"],
            "class_name": obj["class_name"],
            "video_time_s": frame_meta["video_time_s"],
            "timebase_verified": frame_meta["timebase_verified"],
            "x1": obj["x1"],
            "y1": obj["y1"],
            "x2": obj["x2"],
            "y2": obj["y2"],
            "center_u": float((obj["x1"] + obj["x2"]) / 2),
            "center_v": float((obj["y1"] + obj["y2"]) / 2),
            "confidence": obj["confidence"],
            "relative_depth_median": depth_median,
            "relative_depth_iqr": depth_iqr,
            "depth_valid_fraction": valid_frac,
            "context": context_str,
            "context_support": support_val,
            "quality_flags": quality_flags,
            "depth_m": None,
            "camera_x_m": None,
            "camera_y_m": None,
            "camera_z_m": None,
            "world_x_m": None,
            "world_y_m": None,
            "world_z_m": None,
            "metric_status": "unsupported",
        }
        rows.append(row)
        
    return rows

def rows_to_table(rows, schema):
    return pa.Table.from_pylist(rows, schema=schema)
