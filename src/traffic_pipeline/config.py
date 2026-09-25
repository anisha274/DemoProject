import yaml
import os

class Config:
    def __init__(self, cfg_path: str):
        self.cfg_path = os.path.abspath(cfg_path)
        self.base_dir = os.path.dirname(os.path.dirname(self.cfg_path))
        with open(cfg_path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
        
        self.validate()

    def validate(self):
        if self.data.get("schema_version") != "1.0":
            raise ValueError("Unsupported schema_version")
        if self.data["runtime"]["frame_stride"] != 1:
            raise ValueError("frame_stride must be 1 for this implementation")
        
    def resolve_path(self, p):
        if not p:
            return p
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(self.base_dir, p))

    @property
    def video_dir(self): return self.resolve_path(self.data["paths"]["video_dir"])
    @property
    def output_dir(self): return self.resolve_path(self.data["paths"]["output_dir"])
    @property
    def device(self): return self.data["runtime"]["device"]
    @property
    def max_gap_frames(self): return self.data["temporal"]["max_gap_frames"]
    @property
    def raw(self): return self.data
