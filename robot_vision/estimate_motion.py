"""Step 8: temporal motion of tracked objects (people are the main result), from Step 4 world positions.

REFERENCE-DEPTH AND REFERENCE-CAMERA-POSE BASELINE: positions come from THUD reference depth and the recorded
camera poses. World positions already include camera-motion compensation (p_world = R p_cam + t), so camera
movement is NOT subtracted again.

Inputs (unchanged): a finished fuse_objects.py run (object_states.csv, frames.jsonl). No inference is rerun.
Reference instance masks and the Step 4 track-to-reference association are read ONLY in evaluate().

Per observation (motion_states.csv):
- quality: pixel_source, selected_pixel_count, depth_spread_m, invalid_reason, reliable (config rules)
- world motion: raw and filtered ground-plane position, velocity over a trailing time window, speed, heading,
  motion_label (stationary / approaching / moving_away / moving_lateral / unknown) and motion_status
- relative motion: distance to camera_proxy (camera position; camera-to-robot offset not defined yet) and its
  rate, split into the object's and the camera's contribution; range_label
- route: only with an explicit corridor config (THUD: "proxy_route"); otherwise not_evaluated.
  time_to_route_s assumes constant velocity and is NOT a time to collision.

Usage (from robot_vision/):
    python estimate_motion.py                      # config configs/motion_office.json
    python estimate_motion.py --config <file> --no-route
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import motion as mo  # noqa: E402
import thud_data as td  # noqa: E402
import thud_depth as tdd  # noqa: E402
from fuse_objects import depth_path, instance_masks, resolve, sensor_pose, stats, write_json  # noqa: E402
from track_objects import track_colour  # noqa: E402
from validate_depth import load_records, quat_matrix, sensor_roles  # noqa: E402

HERE = Path(__file__).resolve().parent
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
INK, MUTED, GRID = "#2b2b2a", "#77766f", "#e4e3de"

FIELDS = [
    "sequence_id", "step", "timestamp", "track_id", "class_name",
    "pixel_source", "selection_method", "selected_pixel_count", "valid_depth_pixel_count", "depth_spread_m",
    "position_valid", "invalid_reason", "reliable", "unreliable_reason",
    "raw_world_x", "raw_world_y", "raw_world_z",
    "segment", "resets_so_far", "motion_status",
    "filt_world_x", "filt_world_y", "filt_world_z", "vel_x_mps", "vel_z_mps", "vel_se_x_mps", "vel_se_z_mps",
    "speed_mps", "speed_se_mps", "heading_deg", "window_n", "window_span_s", "velocity_lag_s",
    "moving_state", "motion_label",
    "camera_proxy_x", "camera_proxy_z", "range_m", "range_rate_mps", "range_rate_object_mps",
    "range_rate_camera_mps", "range_label",
    "route_label", "route_status", "dist_to_route_centreline_m", "time_to_route_s", "time_to_route_p10_s",
    "time_to_route_p90_s", "p_enter_route_within_horizon",
]


# ---------------------------------------------------------------- helpers

def fnum(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def git_info() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- camera and ground plane

class Camera:
    """Reference camera position (camera_proxy) and its trailing-window velocity, per step."""

    def __init__(self, frames: list[dict], window_s: float):
        self.step_index = {f["step"]: i for i, f in enumerate(frames)}
        self.t = np.array([f["timestamp"] for f in frames], float)
        self.xz = np.array([[f["pose_translation"][0], f["pose_translation"][2]] for f in frames], float)
        self.y = np.array([f["pose_translation"][1] for f in frames], float)
        self.pose = {f["step"]: f for f in frames}
        self.vel = np.zeros_like(self.xz)
        for i, t in enumerate(self.t):
            sel = (self.t <= t + 1e-9) & (self.t >= t - window_s - 1e-9)
            if sel.sum() >= 2:
                self.vel[i] = [mo.line_fit(self.t[sel], self.xz[sel, a], t)[1] for a in (0, 1)]

    def at(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        i = self.step_index[step]
        return self.xz[i], self.vel[i]


def ground_plane_check(frames: list[dict], cfg: dict) -> dict:
    fy = np.array([f["floor_world_y"] for f in frames if f.get("floor_world_y") is not None], float)
    iqr = np.array([f["floor_world_y_iqr"] for f in frames if f.get("floor_world_y_iqr") is not None], float)
    lim = cfg["max_floor_height_spread_m"]
    cam_y = np.array([f["pose_translation"][1] for f in frames], float)
    ok = len(fy) >= 0.5 * len(frames) and float(np.median(iqr)) <= lim and float(fy.max() - fy.min()) <= lim
    return {"vertical_axis": cfg["vertical_axis"], "plane_axes": cfg["plane_axes"],
            "frames_with_floor": int(len(fy)), "frames": len(frames),
            "floor_world_y_median_m": float(np.median(fy)), "floor_world_y_range_across_frames_m": float(fy.max() - fy.min()),
            "floor_world_y_within_frame_iqr_median_m": float(np.median(iqr)), "limit_m": lim,
            "camera_height_above_floor_m": stats(cam_y - np.median(fy)),
            "method": "world y of predicted-floor pixels back-projected with reference depth and pose (Step 4)",
            "passed": bool(ok)}


# ---------------------------------------------------------------- estimation (prediction side)

def estimate_series(t, xz, steps, cls, cam: Camera, cfg: dict, route: dict | None, rng) -> tuple[list[dict], list[str]]:
    """Motion estimates for one time-ordered series of reliable positions (a track, or a reference instance)."""
    mcfg, lab = cfg["motion"], cfg["labels"]
    seg, reasons = mo.segment_track(t, xz, cls, mcfg)
    out = []
    state_prev, lab_track, rng_track, cur_seg = None, None, None, -1
    for i in range(len(t)):
        if seg[i] != cur_seg:
            cur_seg, state_prev = seg[i], None
            lab_track, rng_track = mo.Persistent(lab["persistence_s"]), mo.Persistent(lab["persistence_s"])
        c, vc = cam.at(int(steps[i]))
        rec = {"timestamp": float(t[i]), "segment": int(seg[i]), "resets_so_far": int(seg[i]), "camera_proxy_x": c[0], "camera_proxy_z": c[1],
               "range_m": float(np.hypot(*(xz[i] - c)))}
        e = mo.window_estimate(t, xz, i, seg, mcfg)
        if e is None:
            rec.update({"motion_status": "insufficient_history", "motion_label": "insufficient_history",
                        "range_label": "insufficient_history"})
            rec.update(route_fields(None, None, route, cfg, rng))
            out.append(rec)
            continue
        p, v = np.array([e["x"], e["z"]]), np.array([e["vx"], e["vz"]])
        ranges = np.array([np.hypot(*(xz[j] - cam.at(int(steps[j]))[0])) for j in e["idx"]])
        _, rate, _ = mo.line_fit(t[e["idx"]], ranges, t[i])
        u = (p - c) / max(np.linalg.norm(p - c), 1e-9)  # camera_proxy -> object
        radial_obj, radial_cam = float(v @ u), float(-vc @ u)
        state_prev = mo.moving_state(e["speed"], state_prev, lab)
        rec.update({
            "motion_status": "estimated", "filt_world_x": e["x"], "filt_world_z": e["z"],
            "vel_x_mps": e["vx"], "vel_z_mps": e["vz"], "vel_se_x_mps": e["se_vx"], "vel_se_z_mps": e["se_vz"],
            "speed_mps": e["speed"], "speed_se_mps": e["speed_se"], "heading_deg": e["heading_deg"],
            "window_n": e["n"], "window_span_s": e["span_s"], "velocity_lag_s": e["lag_s"],
            "moving_state": state_prev,
            "motion_label": lab_track.update(mo.motion_candidate(state_prev, radial_obj, lab), t[i]),
            "range_rate_mps": rate, "range_rate_object_mps": radial_obj, "range_rate_camera_mps": radial_cam,
            "range_label": rng_track.update(mo.range_candidate(rate, lab), t[i]),
            "_idx": e["idx"]})
        rec.update(route_fields(e, rec["motion_label"], route, cfg, rng))
        out.append(rec)
    return out, reasons


def route_fields(e: dict | None, label: str | None, route: dict | None, cfg: dict, rng) -> dict:
    if route is None:
        return {"route_label": "none", "route_status": "not_evaluated"}
    out = {"route_label": route["label"]}
    if e is None:
        return {**out, "route_status": "unknown"}
    p = np.array([e["x"], e["z"]])
    half = route["width_m"] / 2
    d = float(mo.polyline_distance(p, route["line"]))
    out["dist_to_route_centreline_m"] = d
    if d <= half:
        return {**out, "route_status": "in_route", "time_to_route_s": 0.0}
    if label == "stationary":
        return {**out, "route_status": "stationary_outside"}
    if label not in mo.MOVING_LABELS:
        return {**out, "route_status": "unknown"}
    r = cfg["route"]
    v = np.array([[e["vx"], e["vz"]]])
    ttr = float(mo.time_to_route(p, v, route["line"], half, r["horizon_s"], r["time_step_s"])[0])
    samples = v + rng.normal(size=(r["monte_carlo_samples"], 2)) * np.array([e["se_vx"], e["se_vz"]])
    ts = mo.time_to_route(p, samples, route["line"], half, r["horizon_s"], r["time_step_s"])
    fin = ts[np.isfinite(ts)]
    out.update({"route_status": "crossing_route" if math.isfinite(ttr) else "not_crossing",
                "p_enter_route_within_horizon": float(len(fin) / len(ts))})
    if math.isfinite(ttr):
        out["time_to_route_s"] = ttr
    if len(fin):
        out["time_to_route_p10_s"], out["time_to_route_p90_s"] = (float(x) for x in np.percentile(fin, [10, 90]))
    return out


def estimate_tracks(states: list[dict], cam: Camera, cfg: dict, route: dict | None, rng) -> tuple[list[dict], dict]:
    by_track = defaultdict(list)
    for s in states:
        by_track[s["track_id"]].append(s)
    rows, resets = [], defaultdict(Counter)
    for tid, obs in by_track.items():
        obs.sort(key=lambda s: s["timestamp"])
        for s in obs:
            s["reliable"], s["unreliable_reason"] = mo.quality(s, cfg["quality"])
        rel = [s for s in obs if s["reliable"]]
        est = {}
        if rel:
            t = np.array([s["timestamp"] for s in rel])
            xz = np.array([[s["world_x"], s["world_z"]] for s in rel])
            series, reasons = estimate_series(t, xz, [s["step"] for s in rel], [s["class_name"] for s in rel],
                                              cam, cfg, route, rng)
            resets[rel[0]["class_name"]].update(reasons)
            for s, r in zip(rel, series):
                idx = r.pop("_idx", None)
                if idx is not None:
                    r["filt_world_y"] = float(np.median([rel[j]["world_y"] for j in idx]))
                est[s["step"]] = r
        last_seg = 0
        for s in obs:
            row = {k: s[k] for k in ("sequence_id", "step", "timestamp", "track_id", "class_name", "selection_method",
                                     "position_valid", "invalid_reason", "reliable", "unreliable_reason")}
            row.update({"pixel_source": mo.pixel_source(s["selection_method"]), "selected_pixel_count": s["selected_pixels"],
                        "valid_depth_pixel_count": s["valid_depth_pixels"], "depth_spread_m": s["depth_iqr_m"],
                        "raw_world_x": s["world_x"], "raw_world_y": s["world_y"], "raw_world_z": s["world_z"]})
            if s["step"] in est:
                row.update(est[s["step"]])
                last_seg = row["segment"]
            else:
                row.update({"segment": last_seg, "resets_so_far": last_seg, "motion_status": "unreliable_observation",
                            "motion_label": "unknown", "range_label": "unknown",
                            "route_label": route["label"] if route else "none",
                            "route_status": "unknown" if route else "not_evaluated"})
                if s["position_valid"]:
                    c, _ = cam.at(s["step"])
                    row.update({"camera_proxy_x": c[0], "camera_proxy_z": c[1],
                                "range_m": float(np.hypot(s["world_x"] - c[0], s["world_z"] - c[1]))})
            rows.append(row)
    rows.sort(key=lambda r: (r["step"], int(r["track_id"])))
    return rows, {k: dict(v) for k, v in resets.items()}


# ---------------------------------------------------------------- evaluation (reference data used here only)

def reference_points(ctx: dict, ids: set[int], cfg: dict) -> dict[int, list[dict]]:
    """Per reference instance, per frame: world median of its instance-mask pixels (visible-surface proxy)."""
    out = defaultdict(list)
    k = ctx["K"]
    for f in ctx["frames"]:
        meta = ctx["records"][(ctx["label_sensor"], f["step"])]
        packed, colours = instance_masks(ctx["capture_dir"], meta)
        depth, valid = tdd.decode_depth(tdd.load_raw_depth(depth_path(ctx["capture_dir"], meta)), ctx["far_m"])
        R, t = sensor_pose(meta)
        for iid in ids:
            if iid not in colours:
                continue
            ys, xs = np.nonzero((packed == colours[iid]) & valid)
            if len(ys) < cfg["evaluation"]["reference_min_pixels"]:
                continue
            p = np.median(tdd.backproject(xs, ys, depth[ys, xs], k), axis=0) @ R.T + t
            out[iid].append({"step": f["step"], "timestamp": f["timestamp"], "x": p[0], "y": p[1], "z": p[2],
                             "pixels": int(len(ys))})
    return out


def angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def evaluate(rows: list[dict], assoc: list[dict], ctx: dict, cam: Camera, cfg: dict, route: dict | None, rng) -> dict:
    ev = cfg["evaluation"]
    link = {(int(a["step"]), a["track_id"]): (int(a["ref_instance_id"]), a["ref_label"])
            for a in assoc if a["associated"] == "True"}
    ref_label = {iid: lab for iid, lab in link.values()}
    refs = reference_points(ctx, set(ref_label), cfg)
    ref_est = {}
    for iid, pts in refs.items():
        t = np.array([p["timestamp"] for p in pts])
        xz = np.array([[p["x"], p["z"]] for p in pts])
        series, _ = estimate_series(t, xz, [p["step"] for p in pts], [ref_label[iid]] * len(pts), cam, cfg, route, rng)
        for p, r in zip(pts, series):
            r.pop("_idx", None)
            ref_est[(iid, p["step"])] = r

    # groups by the track's most frequent associated reference label (evaluation-only)
    track_labels = defaultdict(Counter)
    for (step, tid), (_, lab) in link.items():
        track_labels[tid][lab] += 1

    def group_of(lab):
        if lab in ev["person_reference_labels"]:
            return "people"
        if lab in ev["static_verified_reference_labels"]:
            return "static_verified"
        if lab in ev["static_unverified_reference_labels"]:
            return "static_unverified"
        return "other"
    track_group = {tid: group_of(c.most_common(1)[0][0]) for tid, c in track_labels.items()}

    known = lambda lab: lab not in ("unknown", "insufficient_history", None)  # noqa: E731
    pairs = defaultdict(list)  # group -> (row, ref estimate)
    for r in rows:
        r["eval_group"] = track_group.get(r["track_id"], "unassociated")
        a = link.get((r["step"], r["track_id"]))
        if a:
            r["ref_instance_id"], r["ref_label"] = a
            re = ref_est.get((a[0], r["step"]))
            if re is not None:
                pairs[group_of(a[1])].append((r, re))

    def motion_compare(prs):
        both = [(r, q) for r, q in prs if r["motion_status"] == "estimated" and q["motion_status"] == "estimated"]
        se = [r["speed_mps"] - q["speed_mps"] for r, q in both]
        de = [angle_diff(r["heading_deg"], q["heading_deg"]) for r, q in both
              if r["speed_mps"] >= ev["direction_min_speed_mps"] and q["speed_mps"] >= ev["direction_min_speed_mps"]]
        lab = [(r["motion_label"], q["motion_label"]) for r, q in prs if known(r["motion_label"]) and known(q["motion_label"])]
        rlab = [(r["range_label"], q["range_label"]) for r, q in prs if known(r["range_label"]) and known(q["range_label"])]
        conf = defaultdict(Counter)
        for a_, b_ in lab:
            conf[b_][a_] += 1
        return {"pairs_with_both_estimates": len(both),
                "speed_error_mps": stats(se), "abs_speed_error_mps": stats(abs(x) for x in se),
                "reference_speed_mps": stats(q["speed_mps"] for _, q in both),
                "estimated_speed_mps": stats(r["speed_mps"] for r, _ in both),
                "direction_error_deg": stats(de),
                "direction_rule": f"both speeds >= {ev['direction_min_speed_mps']} m/s",
                "motion_label_pairs": len(lab), "motion_label_agreement": float(np.mean([a_ == b_ for a_, b_ in lab])) if lab else None,
                "motion_label_confusion_ref_to_est": {k: dict(v) for k, v in conf.items()},
                "range_label_pairs": len(rlab),
                "range_label_agreement": float(np.mean([a_ == b_ for a_, b_ in rlab])) if rlab else None}

    def static_control(group):
        rs = [r for r in rows if r["eval_group"] == group and r["motion_status"] == "estimated"]
        lab = [r["motion_label"] for r in rs if known(r["motion_label"])]
        iids = {r["ref_instance_id"] for r in rows if r["eval_group"] == group and "ref_instance_id" in r}
        qs = [q for (iid, _), q in ref_est.items() if iid in iids and q["motion_status"] == "estimated"]
        qlab = [q["motion_label"] for q in qs if known(q["motion_label"])]
        return {"true_velocity": "0 (static; verified by fixed 3D boxes in Step 4)" if group == "static_verified"
                else "assumed 0, NOT verified (no 3D boxes)",
                "tracks": len({r["track_id"] for r in rs}), "estimates": len(rs),
                "speed_mps": stats(r["speed_mps"] for r in rs),
                "speed_p99_mps": float(np.percentile([r["speed_mps"] for r in rs], 99)) if rs else None,
                "labelled": len(lab), "false_motion_rate": float(np.mean([x in mo.MOVING_LABELS for x in lab])) if lab else None,
                "labels": dict(Counter(lab)),
                "reference_proxy_same_instances": {
                    "estimates": len(qs), "speed_mps": stats(q["speed_mps"] for q in qs),
                    "false_motion_rate": float(np.mean([x in mo.MOVING_LABELS for x in qlab])) if qlab else None,
                    "note": "the same estimator on the reference visible-surface point; motion here comes from the "
                            "visible point sliding over the surface as the camera moves, not from the object"}}

    # identity
    ids_per_track = defaultdict(list)
    for (step, tid), (iid, lab) in sorted(link.items()):
        ids_per_track[tid].append(iid)
    switches = {tid: sum(a_ != b_ for a_, b_ in zip(v, v[1:])) for tid, v in ids_per_track.items()}
    tracks_per_ref = defaultdict(set)
    for (step, tid), (iid, lab) in link.items():
        tracks_per_ref[iid].add(tid)
    people_ids = [iid for iid, lab in ref_label.items() if lab in ev["person_reference_labels"]]

    out = {
        "reference_use": "reference instance masks + reference depth + reference poses give a visible-surface-point "
                         "trajectory per reference instance (instance IDs are stable across frames). Its velocity is a "
                         "PROXY, not the exact person-centre velocity. Tracks link to instances through the Step 4 "
                         "per-frame association (IoU >= 0.5, class-compatible); comparisons use the instance associated "
                         "at that same step.",
        "identity": {"tracks_with_association": len(ids_per_track),
                     "tracks_with_reference_id_switches": sum(1 for v in switches.values() if v),
                     "reference_id_switches_total": int(sum(switches.values())),
                     "people_reference_instances": len(people_ids),
                     "tracks_per_people_instance": {str(i): len(tracks_per_ref[i]) for i in sorted(people_ids)}},
        "people": motion_compare(pairs["people"]),
        "static_control": {"static_verified": static_control("static_verified"),
                           "static_unverified": static_control("static_unverified")},
        "static_verified_vs_reference": motion_compare(pairs["static_verified"]),
    }
    if route is not None:
        out["time_to_route"] = ttr_eval(rows, refs, route, cfg)
    else:
        out["time_to_route"] = "not_evaluated (no route corridor supplied)"
    return out, refs, ref_est


def ttr_eval(rows: list[dict], refs: dict, route: dict, cfg: dict) -> dict:
    """Compare time_to_route with when the associated reference visible point actually entered the corridor."""
    half, H = route["width_m"] / 2, cfg["route"]["horizon_s"]
    pts = {iid: (np.array([p["timestamp"] for p in v]), mo.polyline_distance(np.array([[p["x"], p["z"]] for p in v]),
                                                                                route["line"]) <= half)
           for iid, v in refs.items()}
    res, realized, outcome = [], [], Counter()
    for r in rows:
        if r.get("eval_group") != "people" or "ref_instance_id" not in r or r["route_status"] not in (
                "crossing_route", "not_crossing", "stationary_outside"):
            continue
        if r["ref_instance_id"] not in pts:
            outcome["no_reference_points"] += 1
            continue
        t, inside = pts[r["ref_instance_id"]]
        fut = (t > r["timestamp"] + 1e-9) & (t <= r["timestamp"] + H + 1e-9)
        entered = t[fut & inside]
        observed_to_horizon = t.max() >= r["timestamp"] + H - 0.05 and fut.any()
        if r["route_status"] == "crossing_route":
            if len(entered):
                outcome["crossing_and_entered"] += 1
                realized.append(entered[0] - r["timestamp"])
                res.append(r["time_to_route_s"] - realized[-1])
            else:
                outcome["crossing_no_entry_observed" if observed_to_horizon else "crossing_censored"] += 1
        else:
            if len(entered):
                outcome["not_crossing_but_entered"] += 1
            else:
                outcome["not_crossing_no_entry_observed" if observed_to_horizon else "not_crossing_censored"] += 1
    return {"route": route["label"], "outcomes_people": dict(outcome),
            "error_s_estimated_minus_realized": stats(res), "abs_error_s": stats(abs(x) for x in res),
            "realized_time_to_route_s": stats(realized),
            "realized_entry": "first later frame (within the horizon) in which the associated reference visible point "
                              "is inside the corridor; censored = the sequence or the reference visibility ended first",
            "caveat": "time to reach a static proxy corridor under constant velocity; not a time to collision"}


# ---------------------------------------------------------------- drawing

def corridor_patches(ax, route: dict, colour: str) -> None:
    half, line = route["width_m"] / 2, route["line"]
    for a, b in zip(line[:-1], line[1:]):
        d = b - a
        n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9) * half
        ax.add_patch(plt.Polygon([a + n, b + n, b - n, a - n], color=colour, lw=0, zorder=0))
    for p in line:
        ax.add_patch(plt.Circle(p, half, color=colour, lw=0, zorder=0))


def style(ax) -> None:
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def person_tracks(rows: list[dict], n: int) -> list[str]:
    c = Counter(r["track_id"] for r in rows if r.get("eval_group") == "people" and r["motion_status"] == "estimated")
    return [tid for tid, _ in c.most_common(n)]


def draw_birdseye(rows, cam: Camera, route, refs, tracks, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    if route:
        corridor_patches(ax, route, "#e8eef7")
        ax.plot([], [], color="#c9d6ea", lw=8, label=f"proxy_route corridor ({route['width_m']:.2f} m wide)")
    ax.plot(cam.xz[:, 0], cam.xz[:, 1], color=INK, lw=2, label="camera path (camera_proxy, reference pose)")
    ax.plot(*cam.xz[0], "o", color=INK, ms=5)
    st = np.array([[r["raw_world_x"], r["raw_world_z"]] for r in rows if r.get("eval_group") == "static_verified"
                   and r["reliable"]], float)
    if len(st):
        ax.scatter(st[:, 0], st[:, 1], s=3, color="#b9b8b0", label="static furniture tracks (raw)", zorder=1)
    for i, tid in enumerate(tracks):
        col = SERIES[i % len(SERIES)]
        tr = [r for r in rows if r["track_id"] == tid]
        raw = np.array([[r["raw_world_x"], r["raw_world_z"]] for r in tr if r["reliable"]], float)
        ax.scatter(raw[:, 0], raw[:, 1], s=4, color=col, alpha=0.35, zorder=2)
        segs = defaultdict(list)
        for r in tr:
            if r["motion_status"] == "estimated":
                segs[r["segment"]].append([r["filt_world_x"], r["filt_world_z"]])
        for k, v in enumerate(segs.values()):
            v = np.array(v)
            ax.plot(v[:, 0], v[:, 1], color=col, lw=2, zorder=3, label=f"person track {tid} (filtered)" if k == 0 else None)
        iids = {r["ref_instance_id"] for r in tr if "ref_instance_id" in r}
        for iid in iids:
            p = np.array([[q["x"], q["z"]] for q in refs.get(iid, [])])
            if len(p):
                ax.plot(p[:, 0], p[:, 1], color=col, lw=1, ls="--", zorder=3)
    ax.plot([], [], color=MUTED, lw=1, ls="--", label="reference visible-point proxy (evaluation only)")
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)", color=INK)
    ax.set_ylabel("world z (m)", color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    style(ax)
    ax.legend(fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_speed_people(rows, ref_est, tracks, lab, path: Path, title: str) -> None:
    n = max(len(tracks), 1)
    cols = 2
    fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(11, 2.4 * math.ceil(n / cols)), squeeze=False,
                             sharex=True)
    for ax, (i, tid) in zip(axes.flat, enumerate(tracks)):
        col = SERIES[i % len(SERIES)]
        tr = [r for r in rows if r["track_id"] == tid and r["motion_status"] == "estimated"]
        segs = defaultdict(list)
        for r in tr:
            segs[r["segment"]].append(r)
        for k, s in enumerate(segs.values()):
            ax.plot([r["timestamp"] for r in s], [r["speed_mps"] for r in s], color=col, lw=2,
                    label="estimated (track)" if k == 0 else None)
        rp = [(r["timestamp"], ref_est.get((r.get("ref_instance_id"), r["step"]))) for r in tr]
        rp = [(t, q["speed_mps"]) for t, q in rp if q and q["motion_status"] == "estimated"]
        if rp:
            ax.plot(*zip(*rp), color=INK, lw=1, ls="--", label="reference proxy")
        for y in (lab["moving_enter_speed_mps"], lab["moving_exit_speed_mps"]):
            ax.axhline(y, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"person track {tid}", fontsize=9, color=INK, loc="left")
        ax.set_ylabel("speed (m/s)", fontsize=8, color=INK)
        style(ax)
        ax.legend(fontsize=7, frameon=False, loc="best")
    for ax in list(axes.flat)[len(tracks):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("timestamp (s)", fontsize=8, color=INK)
    fig.suptitle(title + "\ndotted lines: moving enter / exit thresholds", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_speed_static(rows, ref_est, lab, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    st = [r for r in rows if r.get("eval_group") == "static_verified" and r["motion_status"] == "estimated"]
    by = defaultdict(list)
    for r in st:
        by[(r["track_id"], r["segment"])].append(r)
    for s in by.values():
        axes[0].plot([r["timestamp"] for r in s], [r["speed_mps"] for r in s], color=SERIES[0], lw=1, alpha=0.7)
    iids = {r["ref_instance_id"] for r in rows if r.get("eval_group") == "static_verified" and "ref_instance_id" in r}
    byr = defaultdict(list)
    for (iid, step), q in sorted(ref_est.items(), key=lambda x: x[0][1]):
        if iid in iids and q["motion_status"] == "estimated":
            byr[(iid, q["segment"])].append((q["timestamp"], q["speed_mps"]))
    for s in byr.values():
        axes[1].plot(*zip(*s), color=INK, lw=1, alpha=0.6)
    axes[0].set_title("estimated (static Chair/Desk/Sofa tracks)", fontsize=9, color=INK, loc="left")
    axes[1].set_title("reference visible-point proxy, same instances", fontsize=9, color=INK, loc="left")
    for ax in axes:
        for y in (lab["moving_enter_speed_mps"], lab["moving_exit_speed_mps"]):
            ax.axhline(y, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("time (s)", fontsize=8, color=INK)
        style(ax)
    axes[0].set_ylabel("speed (m/s); true speed = 0", fontsize=8, color=INK)
    fig.suptitle(title, fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def project(pw: np.ndarray, frame: dict, k: dict) -> tuple[float, float] | None:
    x, y, z, w = frame["pose_rotation_xyzw"]
    R = quat_matrix({"x": x, "y": y, "z": z, "w": w})
    p = R.T @ (pw - np.array(frame["pose_translation"]))
    if p[2] <= 0.05:
        return None
    return k["fx"] * p[0] / p[2] + k["cx"] - 0.5, k["cy"] - k["fy"] * p[1] / p[2] - 0.5


def draw_overlay(rgb_path: Path, rows: list[dict], states: dict, frame: dict, k: dict, banner: str, path: Path) -> None:
    img = Image.open(rgb_path).convert("RGB")
    d = ImageDraw.Draw(img)
    for r in rows:
        s = states[(r["step"], r["track_id"])]
        col = track_colour(int(r["track_id"]))
        d.rectangle([s["x1"], s["y1"], s["x2"], s["y2"]], outline=col, width=2)
        if r["motion_status"] == "estimated":
            l1 = f"#{r['track_id']} {r['class_name']} {r['motion_label']} {r['speed_mps']:.2f}+-{r['speed_se_mps']:.2f} m/s"
            l2 = f"{r['range_label']}  route: {r['route_status']}" + (
                f" {r['time_to_route_s']:.1f}s" if r.get("time_to_route_s") not in (None, "") and r["route_status"] == "crossing_route" else "")
            y_ = r.get("filt_world_y", r["raw_world_y"])
            a = project(np.array([r["filt_world_x"], y_, r["filt_world_z"]]), frame, k)
            b = project(np.array([r["filt_world_x"] + r["vel_x_mps"], y_, r["filt_world_z"] + r["vel_z_mps"]]), frame, k)
            if a and b:
                d.line([a, b], fill=col, width=3)
                d.ellipse([a[0] - 3, a[1] - 3, a[0] + 3, a[1] + 3], fill=col)
        else:
            l1, l2 = f"#{r['track_id']} {r['class_name']} {r['motion_status']}", r["unreliable_reason"] or ""
        y = max(0, s["y1"] - 26)
        w_ = max(d.textlength(l1), d.textlength(l2)) + 4
        d.rectangle([s["x1"], y, s["x1"] + w_, y + 25], fill=(0, 0, 0))
        d.text((s["x1"] + 2, y), l1, fill=col)
        d.text((s["x1"] + 2, y + 12), l2, fill=col)
    d.rectangle([0, 0, img.width, 28], fill=(0, 0, 0))
    d.text((3, 1), f"step {frame['step']}  t={frame['timestamp']:.3f}s  {banner}", fill=(255, 255, 0))
    d.text((3, 14), "arrow = filtered world velocity x 1 s; camera position = camera_proxy", fill=(255, 255, 0))
    img.save(path)


# ---------------------------------------------------------------- main

def load_states(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    out = []
    for r in raw:
        s = dict(r)
        s["step"], s["timestamp"] = int(r["step"]), float(r["timestamp"])
        s["position_valid"] = r["position_valid"] == "True"
        for c in ("x1", "y1", "x2", "y2", "world_x", "world_y", "world_z", "depth_iqr_m"):
            s[c] = fnum(r[c])
        s["selected_pixels"], s["valid_depth_pixels"] = int(r["selected_pixels"]), int(r["valid_depth_pixels"])
        out.append(s)
    return out


def run(args: argparse.Namespace) -> Path:
    cfg_path = resolve(args.config)
    cfg = read_json(cfg_path)
    office = td.load_config(resolve(cfg["office_config"]))
    splits = read_json(resolve(cfg["splits_config"]))
    robot = read_json(resolve(cfg["robot_config"]))
    depth_cfg = tdd.load_depth_config(resolve(cfg["depth_config"]))
    route_cfg = None if args.no_route or not cfg.get("route_config") else read_json(resolve(cfg["route_config"]))
    route = None
    if route_cfg:
        from make_proxy_route import required_width
        w, _ = required_width(robot)
        if abs(w - route_cfg["width_m"]) > 1e-6:
            raise SystemExit(f"Route width {route_cfg['width_m']} differs from robot.json required width {w}; regenerate it.")
        route = {"label": route_cfg["label"], "width_m": route_cfg["width_m"], "line": np.array(route_cfg["centreline_xz_m"])}
    fusion = resolve(cfg["fusion_run"])
    if not (fusion / "COMPLETE").exists():
        raise SystemExit(f"Fusion run is not complete: {fusion}")
    fusion_info = read_json(fusion / "run_info.json")
    frames = [json.loads(l) for l in (fusion / "frames.jsonl").read_text(encoding="utf-8").splitlines() if l]
    states = load_states(fusion / "object_states.csv")
    capture = str(Path(office["capture_dir"]).as_posix()).split("Synthetic_Scenes/")[-1]
    split = next((n for n, caps in splits["splits"].items()
                  if capture.startswith(splits["scene"].split("Synthetic_Scenes/")[-1]) and capture.split("/")[-1] in caps), None)
    if split == "final_evaluation" and not args.final_evaluation:
        raise SystemExit("This capture is in the final-evaluation split; pass --final-evaluation for the one frozen-config run.")

    out = resolve(office["output_root"]) / f"motion_{len(frames)}f_{datetime.now():%Y%m%d_%H%M%S}"
    (out / "overlays").mkdir(parents=True)
    (out / "config_used").mkdir()
    for p in [cfg_path, resolve(cfg["robot_config"]), resolve(cfg["splits_config"])] + (
            [resolve(cfg["route_config"])] if route_cfg else []):
        (out / "config_used" / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    banner = cfg["baseline_label"].lower()
    info = {"created": datetime.now().isoformat(timespec="seconds"), "command": " ".join([Path(sys.executable).name] + sys.argv),
            "git": git_info(), "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "config_path": str(cfg_path), "config": cfg, "route_config": route_cfg, "robot_config": robot,
            "data_splits": splits, "capture": capture, "split_of_this_capture": split,
            "fusion_run": str(fusion), "tracking_run": fusion_info["tracking_run"],
            "depth_source": fusion_info["config"]["depth_source"]}
    write_json(out / "run_info.json", info)
    write_json(out / "run_status.json", {"status": "running"})
    try:
        gp = ground_plane_check(frames, cfg["ground_plane"])
        if not gp["passed"]:
            write_json(out / "motion_evaluation.json", {"status": "ground_plane_unverified", "ground_plane_check": gp})
            write_json(out / "run_status.json", {"status": "ground_plane_unverified"})
            print("Ground-plane check FAILED; no motion estimates produced.")
            return out
        rng = np.random.default_rng(cfg["route"]["seed"])
        cam = Camera(frames, cfg["motion"]["window_s"])
        rows, resets = estimate_tracks(states, cam, cfg, route, rng)

        records = load_records(Path(office["capture_dir"]))
        label_s, _ = sensor_roles(records)
        ctx = {"frames": frames, "records": records, "label_sensor": label_s, "capture_dir": Path(office["capture_dir"]),
               "far_m": depth_cfg["far_m"], "K": fusion_info["intrinsics"]}
        with (fusion / "association_evaluation.csv").open(encoding="utf-8") as f:
            assoc = list(csv.DictReader(f))
        ev, refs, ref_est = evaluate(rows, assoc, ctx, cam, cfg, route, rng)

        with (out / "motion_states.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            wr.writeheader()
            for r in rows:
                wr.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
        with (out / "reference_motion_eval_only.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["ref_instance_id", "step"] + [c for c in FIELDS if c in (
                "segment", "motion_status", "filt_world_x", "filt_world_z", "vel_x_mps", "vel_z_mps", "speed_mps",
                "heading_deg", "motion_label", "range_m", "range_rate_mps", "range_label", "route_status",
                "time_to_route_s")], extrasaction="ignore")
            wr.writeheader()
            for (iid, step), q in sorted(ref_est.items()):
                wr.writerow({"ref_instance_id": iid, "step": step,
                             **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in q.items()}})

        est = [r for r in rows if r["motion_status"] == "estimated"]
        lab = cfg["labels"]
        people = [r for r in rows if r.get("eval_group") == "people"]
        report = {
            "baseline": cfg["baseline_label"], "baseline_note": cfg["baseline_note"],
            "capture": capture, "split_of_this_capture": split, "data_splits": splits,
            "git": info["git"], "fusion_run": str(fusion), "settings_provenance": cfg["settings_provenance"],
            "ground_plane_check": gp,
            "definitions": {
                "position": "Step 4 visible-surface point (per-axis median of selected pixels), world frame, y up. "
                            "Ground plane = world x-z.",
                "camera_motion": "World positions already include camera-motion compensation (p_world = R p_cam + t "
                                 "with reference poses); camera movement is not subtracted again.",
                "velocity": f"least-squares slope of x(t), z(t) over the trailing {cfg['motion']['window_s']} s window of "
                            "reliable observations in the current segment; real timestamps, not frame counts.",
                "velocity_delay": f"for a changing velocity the slope describes the window's mean time: nominal delay "
                                  f"{cfg['motion']['window_s'] / 2:.2f} s (half the window); measured median "
                                  f"{np.median([r['velocity_lag_s'] for r in est]):.3f} s. Labels add up to "
                                  f"{lab['persistence_s']} s persistence.",
                "uncertainty": "velocity standard errors from the line-fit residuals (assumes independent residuals, so "
                               "it understates correlated pixel-selection drift); time-to-route p10/p90 and "
                               "p_enter from Monte Carlo velocity samples.",
                "motion_label": cfg["labels"]["rule"],
                "range_label": "sign of d(range)/dt to camera_proxy with dead band; mixes object and camera motion. "
                               "range_rate_object_mps / range_rate_camera_mps split it. A stationary person whose range "
                               "decreases because the camera moved is labelled stationary + range_decreasing, never "
                               "approaching.",
                "camera_proxy": "reference camera position; the camera-to-robot offset is not defined yet.",
                "route": "proxy_route: corridor around the recorded camera path, width from robot.json (robot only, "
                         "load not measured). time_to_route_s = first time the constant-velocity extrapolation enters "
                         "the corridor. It is NOT a time to collision." if route else "no corridor: not_evaluated",
            },
            "coverage": {
                "observations": len(rows), "valid_positions": sum(r["position_valid"] for r in rows),
                "invalid_observations": dict(Counter(r["invalid_reason"] for r in rows if not r["position_valid"])),
                "reliable": sum(r["reliable"] for r in rows),
                "unreliable_reasons": dict(Counter(r["unreliable_reason"] for r in rows if not r["reliable"])),
                "pixel_source_all": dict(Counter(r["pixel_source"] for r in rows)),
                "pixel_source_reliable": dict(Counter(r["pixel_source"] for r in rows if r["reliable"])),
                "motion_status": dict(Counter(r["motion_status"] for r in rows)),
                "motion_labels": dict(Counter(r["motion_label"] for r in rows)),
                "range_labels": dict(Counter(r["range_label"] for r in rows)),
                "route_status": dict(Counter(r["route_status"] for r in rows)),
                "resets_by_detector_class_and_reason": resets,
                "resets_total": int(sum(sum(v.values()) for v in resets.values())),
                "people_rows": len(people),
                "people_motion_status": dict(Counter(r["motion_status"] for r in people)),
                "people_pixel_source": dict(Counter(r["pixel_source"] for r in people)),
                "people_motion_labels": dict(Counter(r["motion_label"] for r in people)),
                "people_route_status": dict(Counter(r["route_status"] for r in people)),
            },
            "evaluation": ev,
        }
        write_json(out / "motion_evaluation.json", report)

        title = f"{capture} steps {frames[0]['step']}-{frames[-1]['step']} | {cfg['baseline_label'].lower()}"
        tracks = person_tracks(rows, cfg["plots"]["max_person_tracks"])
        draw_birdseye(rows, cam, route, refs, tracks, out / "birdseye_trajectories.png", "Bird's-eye (world x-z): " + title)
        draw_speed_people(rows, ref_est, tracks, lab, out / "speed_vs_time_people.png", "Person speed vs time: " + title)
        draw_speed_static(rows, ref_est, lab, out / "speed_vs_time_static_control.png",
                          "Static-furniture control (true speed 0): " + title)
        st_index = {(s["step"], s["track_id"]): s for s in states}
        fr = {f["step"]: f for f in frames}
        for step in cfg["overlay_steps"]:
            if step in fr:
                rgb = Path(office["capture_dir"]) / "RGB" / Path(records[(label_s, step)]["filename"]).name
                draw_overlay(rgb, [r for r in rows if r["step"] == step], st_index, fr[step], fusion_info["intrinsics"],
                             banner, out / "overlays" / f"step_{step:06d}.png")
        write_json(out / "run_status.json", {"status": "complete"})
        (out / "COMPLETE").write_text(datetime.now().isoformat(), encoding="utf-8")
    except Exception as e:
        write_json(out / "run_status.json", {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
        raise
    print(f"Output: {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/motion_office.json")
    p.add_argument("--no-route", action="store_true", help="ignore the route config: route fields = not_evaluated")
    p.add_argument("--final-evaluation", action="store_true",
                   help="required to run on a final-evaluation capture (one run, frozen config)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
