import av
import os
import json

def extract_frames(video_path: str, output_dir: str, max_frames: int = None):
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    manifest = []
    
    container = av.open(video_path)
    stream = container.streams.video[0]
    
    first_pts = None
    time_base = stream.time_base
    fps = stream.average_rate
    
    frame_id = 0
    for frame in container.decode(stream):
        if max_frames is not None and frame_id >= max_frames:
            break
            
        pts = frame.pts
        if first_pts is None and pts is not None:
            first_pts = pts
            
        video_time_s = None
        if pts is not None and first_pts is not None and time_base is not None:
            video_time_s = float((pts - first_pts) * time_base)
        else:
            if fps and fps > 0:
                video_time_s = float(frame_id / fps)
                
        img = frame.to_image()
        frame_filename = f"{frame_id:06d}.png"
        img.save(os.path.join(output_dir, "frames", frame_filename))
        
        manifest.append({
            "frame_id": frame_id,
            "filename": frame_filename,
            "width": img.width,
            "height": img.height,
            "pts": pts,
            "video_time_s": video_time_s,
            "timebase_verified": False
        })
        frame_id += 1
        
    with open(os.path.join(output_dir, "manifest.jsonl"), "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
            
    return manifest
