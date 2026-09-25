import torch
from ultralytics import YOLO

class DetectorAdapter:
    def __init__(self, cfg):
        self.device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(cfg.raw["models"]["detector"])
        self.model.to(self.device)
        self.tracker = cfg.raw["models"]["tracker"]
        self.conf = cfg.raw["models"]["detection_confidence"]
        self.iou = cfg.raw["models"]["detection_iou"]
        self.imgsz = cfg.raw["models"]["detection_size"]
        
        # person, bicycle, car, motorcycle, bus, train, truck
        self.allowed_classes = [0, 1, 2, 3, 5, 6, 7]
        
    def track_video_generator(self, frame_paths):
        # We process frame by frame using persist=True
        # To reset state across videos, we just start a new loop
        # But wait, YOLO track can take a generator, or we can just loop manually.
        pass

    def __call__(self, bgr_img):
        # The prompt says: "Process consecutive frames with persistence within one video; reset state for each video."
        # Actually I will just have the caller instantiate it anew or use persist=True and clear it.
        results = self.model.track(bgr_img, persist=True, tracker=self.tracker, conf=self.conf, iou=self.iou, imgsz=self.imgsz, classes=self.allowed_classes, verbose=False)
        result = results[0]
        boxes = result.boxes
        
        objs = []
        if boxes is not None:
            for box in boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), max(0.0, x2), max(0.0, y2)
                cls_id = int(box.cls[0].item())
                track_id = int(box.id[0].item())
                conf = float(box.conf[0].item())
                cls_name = result.names[cls_id]
                objs.append({
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
        return objs

    def reset(self):
        # We can clear the tracker by resetting self.model.predictor.trackers if it exists
        # Or just re-instantiate it in the caller.
        pass
