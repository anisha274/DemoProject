"""Step 8 building blocks: observation quality, track segmentation, trailing-window velocity, labels with
hysteresis, and route-corridor geometry. Pure functions on plain numbers; no file access.

Conventions: world ground plane = (x, z) with y up (verified from floor heights in estimate_motion.py).
Heading = atan2(vx, vz) in degrees (0 = world +z, 90 = world +x).
"""

from __future__ import annotations

import math

import numpy as np

MOVING_LABELS = ("approaching", "moving_away", "moving_lateral")


# ---------------------------------------------------------------- observation quality

def pixel_source(selection_method: str) -> str:
    if selection_method.startswith("seg_class_"):
        return "class_mask"
    if selection_method == "box_minus_background":
        return "fallback"
    return "none"


def quality(obs: dict, rules: dict) -> tuple[bool, str]:
    """(reliable, reason). Uses only prediction-side fields of a Step 4 object state."""
    if not obs["position_valid"]:
        return False, "position_invalid"
    src = pixel_source(obs["selection_method"])
    if src == "fallback" and not rules["fallback_allowed"]:
        return False, "fallback_not_allowed"
    limit = rules["class_mask_max_depth_iqr_m"] if src == "class_mask" else rules["fallback_max_depth_iqr_m"]
    if obs["depth_iqr_m"] is None or not math.isfinite(obs["depth_iqr_m"]):
        return False, "no_depth_spread"
    if obs["depth_iqr_m"] > limit:
        return False, f"depth_spread_above_{src}_limit"
    return True, ""


# ---------------------------------------------------------------- segmentation (resets)

def segment_track(t: np.ndarray, xz: np.ndarray, cls: list[str], cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Segment index per reliable observation (time-ordered) and the reason for each reset."""
    seg = np.zeros(len(t), int)
    reasons = []
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        jump = float(np.hypot(*(xz[i] - xz[i - 1])))
        reason = None
        if dt > cfg["max_gap_s"]:
            reason = "gap"
        elif jump > cfg["jump_base_m"] + cfg["max_plausible_speed_mps"] * dt:
            reason = "jump"
        elif cfg["reset_on_class_change"] and cls[i] != cls[i - 1]:
            reason = "class_change"
        seg[i] = seg[i - 1] + (reason is not None)
        if reason:
            reasons.append(reason)
    return seg, reasons


# ---------------------------------------------------------------- trailing-window estimator

def line_fit(t: np.ndarray, y: np.ndarray, t_now: float) -> tuple[float, float, float]:
    """Least-squares line through (t, y); returns (value at t_now, slope, slope standard error)."""
    tc = t - t.mean()
    sxx = float((tc ** 2).sum())
    slope = float((tc * (y - y.mean())).sum() / sxx)
    resid = y - (y.mean() + slope * tc)
    dof = len(t) - 2
    se = math.sqrt(float((resid ** 2).sum()) / dof / sxx) if dof > 0 else float("nan")
    return float(y.mean() + slope * (t_now - t.mean())), slope, se


def window_estimate(t: np.ndarray, xz: np.ndarray, i: int, seg: np.ndarray, cfg: dict) -> dict | None:
    """Velocity at observation i from reliable observations of the same segment in [t_i - window, t_i]."""
    sel = (seg == seg[i]) & (t <= t[i] + 1e-9) & (t >= t[i] - cfg["window_s"] - 1e-9)
    tw, pw = t[sel], xz[sel]
    if len(tw) < cfg["min_observations"] or tw[-1] - tw[0] < cfg["min_window_span_s"]:
        return None
    x, vx, sx = line_fit(tw, pw[:, 0], t[i])
    z, vz, sz = line_fit(tw, pw[:, 1], t[i])
    return {"x": x, "z": z, "vx": vx, "vz": vz, "se_vx": sx, "se_vz": sz,
            "speed": math.hypot(vx, vz), "speed_se": math.hypot(sx, sz),
            "heading_deg": math.degrees(math.atan2(vx, vz)) if math.hypot(vx, vz) > 0 else float("nan"),
            "n": int(len(tw)), "span_s": float(tw[-1] - tw[0]), "lag_s": float(t[i] - tw.mean()),
            "idx": np.nonzero(sel)[0]}


# ---------------------------------------------------------------- labels

class Persistent:
    """Emits a candidate label only after it has persisted for `hold_s` seconds."""

    def __init__(self, hold_s: float):
        self.hold, self.current, self.pending, self.since = hold_s, None, None, None

    def update(self, cand: str, t: float) -> str:
        if cand == self.current:
            self.pending = None
        elif cand != self.pending:
            self.pending, self.since = cand, t
        if self.pending is not None and t - self.since >= self.hold - 1e-9:
            self.current, self.pending = self.pending, None
        return self.current or "unknown"


def moving_state(speed: float, previous: str | None, lab: dict) -> str:
    if speed >= lab["moving_enter_speed_mps"]:
        return "moving"
    if speed <= lab["moving_exit_speed_mps"]:
        return "stationary"
    return previous or "unknown"


def motion_candidate(state: str, radial_mps: float, lab: dict) -> str:
    """radial_mps = object's own velocity along camera_proxy -> object (positive = moving away)."""
    if state != "moving":
        return state
    if radial_mps <= -lab["radial_min_speed_mps"]:
        return "approaching"
    if radial_mps >= lab["radial_min_speed_mps"]:
        return "moving_away"
    return "moving_lateral"


def range_candidate(rate: float, lab: dict) -> str:
    if rate <= -lab["range_rate_deadband_mps"]:
        return "range_decreasing"
    if rate >= lab["range_rate_deadband_mps"]:
        return "range_increasing"
    return "range_steady"


# ---------------------------------------------------------------- route corridor

def polyline_distance(p: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Ground-plane distance from points p (..., 2) to a polyline (k, 2)."""
    a, b = line[:-1], line[1:]
    ab = b - a
    ap = p[..., None, :] - a
    s = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    d = np.linalg.norm(ap - s[..., None] * ab, axis=-1)
    return d.min(-1)


def time_to_route(p: np.ndarray, v: np.ndarray, line: np.ndarray, half_width: float, horizon: float,
                  step: float) -> np.ndarray:
    """First time (s) at which p + v*tau is inside the corridor; NaN if not within the horizon.

    p: (2,), v: (m, 2) velocity samples. Constant-velocity assumption. Not a time to collision.
    """
    taus = np.arange(0, horizon + step / 2, step)
    pts = p[None, None, :] + v[:, None, :] * taus[None, :, None]
    inside = polyline_distance(pts, line) <= half_width
    first = np.where(inside.any(1), inside.argmax(1), -1)
    return np.where(first >= 0, taus[np.maximum(first, 0)], np.nan)
