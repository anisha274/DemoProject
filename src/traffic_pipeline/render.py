import json
import os
import cv2
import pyarrow.parquet as pq

def render_results(cfg, video_path, run_id):
    out_dir = os.path.join(cfg.output_dir, run_id)
    seq_id = os.path.splitext(os.path.basename(video_path))[0]
    seq_dir = os.path.join(out_dir, seq_id)
    
    fusion_pq = os.path.join(seq_dir, "object_features.parquet")
    if not os.path.exists(fusion_pq):
        return
    tb = pq.read_table(fusion_pq)
    rows = tb.to_pylist()
    
    # group by frame
    frames = {}
    for r in rows:
        fid = r["frame_id"]
        if fid not in frames:
            frames[fid] = []
        frames[fid].append(r)
        
    frames_dir = os.path.join(seq_dir, "frames")
    manifest_path = os.path.join(seq_dir, "manifest.jsonl")
    manifest = []
    with open(manifest_path, 'r') as f:
        for line in f:
            manifest.append(json.loads(line))
            
    if not manifest: return
    
    first_img_path = os.path.join(frames_dir, manifest[0]["filename"])
    first_img = cv2.imread(first_img_path)
    if first_img is None: return
    
    H, W = first_img.shape[:2]
    
    out_video = os.path.join(seq_dir, "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Original FPS or fallback 10
    fps = 10
    if len(manifest) >= 2 and manifest[1]["video_time_s"] and manifest[0]["video_time_s"]:
        dt = manifest[1]["video_time_s"] - manifest[0]["video_time_s"]
        if dt > 0:
            fps = 1.0 / dt
            
    writer = cv2.VideoWriter(out_video, fourcc, fps, (W, H))
    
    preview_saved = False
    
    for m in manifest:
        fid = m["frame_id"]
        img_path = os.path.join(frames_dir, m["filename"])
        img = cv2.imread(img_path)
        if img is None: continue
        
        objs = frames.get(fid, [])
        for obj in objs:
            x1, y1, x2, y2 = int(obj["x1"]), int(obj["y1"]), int(obj["x2"]), int(obj["y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"ID:{obj['track_id']} {obj['class_name']}"
            if obj["relative_depth_median"] is not None:
                label += f" D:{obj['relative_depth_median']:.2f}"
            if obj["context"] != "unknown":
                label += f" C:{obj['context']}"
                
            cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        writer.write(img)
        
        if not preview_saved:
            cv2.imwrite(os.path.join(seq_dir, "preview.jpg"), img)
            preview_saved = True
            
    writer.release()
    
    # Write summary
    unique_tracks = set(r["track_id"] for r in rows)
    summary = {
        "frames_processed": len(manifest),
        "total_observations": len(rows),
        "unique_tracks": len(unique_tracks),
    }
    with open(os.path.join(seq_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
        
    with open(os.path.join(seq_dir, "report.md"), 'w') as f:
        f.write("# Sequence Report\n\n")
        f.write(f"- Frames processed: {summary['frames_processed']}\n")
        f.write(f"- Unique tracks: {summary['unique_tracks']}\n")
        f.write(f"- Observations: {summary['total_observations']}\n")
