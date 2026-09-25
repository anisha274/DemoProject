import torch
from diffusers import MarigoldDepthPipeline
from PIL import Image
import numpy as np

class DepthAdapter:
    def __init__(self, cfg):
        self.device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = cfg.raw["models"]["depth"]
        self.pipe = MarigoldDepthPipeline.from_pretrained(self.model_id, torch_dtype=torch.float32)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        if self.device == "cpu":
            self.steps = cfg.raw["models"]["cpu_depth_steps"]
            self.res = cfg.raw["models"]["cpu_depth_resolution"]
        else:
            self.steps = cfg.raw["models"]["depth_steps"]
            self.res = cfg.raw["models"]["depth_resolution"]
            
        self.ensemble = cfg.raw["models"]["depth_ensemble"]

    def __call__(self, rgb_img: Image.Image):
        # We use seeded randomness as requested.
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        # We need output to be the original resolution
        orig_size = rgb_img.size # (W, H)
        
        # Marigold outputs a Diffusers object with 'depth' field.
        # Check actual output of the pipeline
        with torch.no_grad():
            out = self.pipe(
                rgb_img,
                num_inference_steps=self.steps,
                ensemble_size=self.ensemble,
                processing_resolution=self.res,
                match_input_resolution=True,
                generator=generator
            )
            
        depth_map = out.prediction[0] # Usually an array or tensor of shape (H, W) or (1, H, W)
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()
        elif isinstance(depth_map, Image.Image):
            depth_map = np.array(depth_map)
            
        # Ensure H x W shape
        if depth_map.ndim == 3:
            if depth_map.shape[0] == 1:
                depth_map = depth_map[0]
            elif depth_map.shape[2] == 1:
                depth_map = depth_map[:, :, 0]
            else:
                depth_map = depth_map[0] if depth_map.shape[0] < depth_map.shape[2] else depth_map[:, :, 0]
            
        # Ensure float32
        depth_map = depth_map.astype(np.float32)
        
        if depth_map.shape != (orig_size[1], orig_size[0]):
            import cv2
            depth_map = cv2.resize(depth_map, orig_size, interpolation=cv2.INTER_LINEAR)
            
        return depth_map
