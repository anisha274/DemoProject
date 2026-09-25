import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np

class SegmentationAdapter:
    def __init__(self, cfg):
        self.device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = cfg.raw["models"]["segmentation"]
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        self.target_size = cfg.raw["models"]["segmentation_size"]
        
    def __call__(self, rgb_img: Image.Image):
        orig_size = rgb_img.size # (W, H)
        
        inputs = self.processor(images=rgb_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Resize logits to original size before argmax
        logits = torch.nn.functional.interpolate(
            logits,
            size=(orig_size[1], orig_size[0]), # (H, W)
            mode="bilinear",
            align_corners=False
        )
        
        seg_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
        # map to road=1, sidewalk=2, other=0
        # Check id2label to find road and sidewalk
        road_id = None
        sidewalk_id = None
        for k, v in self.id2label.items():
            if v == "road":
                road_id = int(k)
            elif v == "sidewalk":
                sidewalk_id = int(k)
                
        context_mask = np.zeros_like(seg_mask)
        if road_id is not None:
            context_mask[seg_mask == road_id] = 1
        if sidewalk_id is not None:
            context_mask[seg_mask == sidewalk_id] = 2
            
        return context_mask
