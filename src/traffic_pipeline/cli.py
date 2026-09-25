import argparse
import os
import json
import logging
from PIL import Image
import numpy as np
import pyarrow.parquet as pq
import pyarrow.csv as csv

from .config import Config
from .video_prep import extract_frames
from .fusion import fuse_frame, rows_to_table
from .temporal import add_temporal_features
from .io_schema import FUSION_SCHEMA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_doctor(cfg):
    logger.info("Running doctor...")
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    return 0

def run_inventory(cfg):
    logger.info("Running inventory...")
    video_dir = cfg.video_dir
    if not os.path.exists(video_dir):
        logger.error(f"Video dir not found: {video_dir}")
        return 1
    videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    logger.info(f"Found {len(videos)} videos.")
    return 0

def run_validate(cfg, run_id):
    logger.info(f"Validating run {run_id}...")
    return 0

def run_status(cfg, run_id):
    logger.info(f"Status of run {run_id}...")
    return 0

def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                rows.append(json.loads(line))
    return rows

def do_run(cfg, video_path, run_id, max_frames, resume):
    out_dir = os.path.join(cfg.output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    
    seq_id = os.path.splitext(os.path.basename(video_path))[0]
    seq_dir = os.path.join(out_dir, seq_id)
    os.makedirs(seq_dir, exist_ok=True)
    
    stage_file = os.path.join(seq_dir, "stage_status.json")
    status = {}
    if resume and os.path.exists(stage_file):
        with open(stage_file, 'r') as f:
            status = json.load(f)
            
    manifest_path = os.path.join(seq_dir, "manifest.jsonl")
    manifest = load_jsonl(manifest_path)
    
    # Check if we need to extend the frame range
    if resume and manifest:
        current_frames = len(manifest)
        needs_extension = False
        if max_frames is None:
            import av
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            if current_frames < total_frames and total_frames > 0:
                needs_extension = True
        elif current_frames < max_frames:
            needs_extension = True
            
        if needs_extension:
            logger.info("Extending frame range, resetting stages...")
            status = {} # Reset status to force rerun of sequence preparation and tracking
            
    # Stage 1: prepare
    if not status.get("prepare") or not resume:
        logger.info("Stage: prepare")
        extract_frames(video_path, seq_dir, max_frames)
        status["prepare"] = True
        with open(stage_file, 'w') as f: json.dump(status, f)
        
    manifest = load_jsonl(manifest_path)
    if not manifest:
        return 1
        
    frames_dir = os.path.join(seq_dir, "frames")
    
    # Init models lazily
    def get_detector():
        from .model_adapters import DetectorAdapter
        return DetectorAdapter(cfg)
    def get_depth():
        from .model_adapters import DepthAdapter
        return DepthAdapter(cfg)
    def get_seg():
        from .model_adapters import SegmentationAdapter
        return SegmentationAdapter(cfg)

    # Stage 2: track
    tracks_path = os.path.join(seq_dir, "tracks.jsonl")
    if not status.get("track") or not resume:
        logger.info("Stage: track")
        detector = get_detector()
        with open(tracks_path, 'w') as f:
            for m in manifest:
                img_path = os.path.join(frames_dir, m["filename"])
                import cv2
                bgr = cv2.imread(img_path)
                objs = detector(bgr)
                f.write(json.dumps({"frame_id": m["frame_id"], "objects": objs}) + "\n")
        status["track"] = True
        with open(stage_file, 'w') as f: json.dump(status, f)
        
    # Stage 3: depth
    depth_dir = os.path.join(seq_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    if not status.get("depth") or not resume:
        logger.info("Stage: depth")
        depth_model = get_depth()
        for m in manifest:
            dpath = os.path.join(depth_dir, f"{m['frame_id']:06d}.npy")
            if resume and os.path.exists(dpath): continue
            img_path = os.path.join(frames_dir, m["filename"])
            rgb = Image.open(img_path).convert("RGB")
            dmap = depth_model(rgb)
            np.save(dpath, dmap)
        status["depth"] = True
        with open(stage_file, 'w') as f: json.dump(status, f)

    # Stage 4: segment
    seg_dir = os.path.join(seq_dir, "segmentation")
    os.makedirs(seg_dir, exist_ok=True)
    if not status.get("segment") or not resume:
        logger.info("Stage: segment")
        seg_model = get_seg()
        for m in manifest:
            spath = os.path.join(seg_dir, f"{m['frame_id']:06d}.png")
            if resume and os.path.exists(spath): continue
            img_path = os.path.join(frames_dir, m["filename"])
            rgb = Image.open(img_path).convert("RGB")
            cmask = seg_model(rgb)
            Image.fromarray(cmask).save(spath)
        status["segment"] = True
        with open(stage_file, 'w') as f: json.dump(status, f)

    # Stage 5: fuse
    fusion_pq = os.path.join(seq_dir, "fusion.parquet")
    fusion_csv = os.path.join(seq_dir, "fusion.csv")
    if not status.get("fuse") or not resume:
        logger.info("Stage: fuse")
        tracks = {t["frame_id"]: t["objects"] for t in load_jsonl(tracks_path)}
        all_rows = []
        for m in manifest:
            fid = m["frame_id"]
            dmap = np.load(os.path.join(depth_dir, f"{fid:06d}.npy"))
            cmask = np.array(Image.open(os.path.join(seg_dir, f"{fid:06d}.png")))
            objs = tracks.get(fid, [])
            rows = fuse_frame(m, objs, dmap, cmask, seq_id)
            all_rows.extend(rows)
            
        tb = rows_to_table(all_rows, FUSION_SCHEMA)
        pq.write_table(tb, fusion_pq)
        csv.write_csv(tb, fusion_csv)
        status["fuse"] = True
        with open(stage_file, 'w') as f: json.dump(status, f)

    # Stage 6: temporal
    temp_pq = os.path.join(seq_dir, "object_features.parquet")
    temp_csv = os.path.join(seq_dir, "object_features.csv")
    if not status.get("temporal") or not resume:
        logger.info("Stage: temporal")
        tb = pq.read_table(fusion_pq)
        tb_temp = add_temporal_features(tb, cfg.max_gap_frames)
        pq.write_table(tb_temp, temp_pq)
        csv.write_csv(tb_temp, temp_csv)
        status["temporal"] = True
        with open(stage_file, 'w') as f: json.dump(status, f)
        
    # Stage 7: render
    if not status.get("render") or not resume:
        logger.info("Stage: render")
        from .render import render_results
        render_results(cfg, video_path, run_id)
        status["render"] = True
        with open(stage_file, 'w') as f: json.dump(status, f)
        
    logger.info("Run complete.")
    return 0

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    
    p_doc = subparsers.add_parser("doctor")
    p_doc.add_argument("--config", required=True)
    
    p_inv = subparsers.add_parser("inventory")
    p_inv.add_argument("--config", required=True)
    
    p_val = subparsers.add_parser("validate")
    p_val.add_argument("--config", required=True)
    p_val.add_argument("--run-id", required=True)
    
    p_stat = subparsers.add_parser("status")
    p_stat.add_argument("--config", required=True)
    p_stat.add_argument("--run-id", required=True)
    
    p_run = subparsers.add_parser("run")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--video", required=True)
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--max-frames", type=int)
    p_run.add_argument("--resume", action="store_true")
    
    p_batch = subparsers.add_parser("batch")
    p_batch.add_argument("--config", required=True)
    p_batch.add_argument("--run-id", required=True)
    p_batch.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()
    cfg = Config(args.config)
    
    if args.cmd == "doctor":
        return run_doctor(cfg)
    elif args.cmd == "inventory":
        return run_inventory(cfg)
    elif args.cmd == "validate":
        return run_validate(cfg, args.run_id)
    elif args.cmd == "status":
        return run_status(cfg, args.run_id)
    elif args.cmd == "run":
        return do_run(cfg, args.video, args.run_id, args.max_frames, args.resume)
    elif args.cmd == "batch":
        videos = sorted([f for f in os.listdir(cfg.video_dir) if f.endswith(".mp4")])
        for v in videos:
            ret = do_run(cfg, os.path.join(cfg.video_dir, v), args.run_id, None, args.resume)
            if ret != 0: return ret
        return 0
