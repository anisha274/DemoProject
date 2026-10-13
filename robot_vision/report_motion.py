"""Step 8 report supplement: velocity window and delay, resets by cause and class, per-person errors and the
time-to-route verdict, computed from a finished estimate_motion.py run (no inference, no re-estimation).

Reference data (the Step 4 association and the reference-proxy estimates saved by the run) is used only for
the per-person comparison and for checking whether a reset coincides with a change of reference identity.

Usage (from robot_vision/):
    python report_motion.py --motion-run outputs/motion_300f_20261013_210331
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import motion as mo
import run_paths as rp
import thud_data as td
from fuse_objects import resolve, stats, write_json


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def window_section(cfg: dict, rows: list[dict]) -> dict:
    m, lab = cfg["motion"], cfg["labels"]
    ts = np.array(sorted({float(r["timestamp"]) for r in rows}))
    dt = float(np.median(np.diff(ts)))
    est = [r for r in rows if r["motion_status"] == "estimated"]
    n = np.array([int(r["window_n"]) for r in est])
    lag = np.array([float(r["velocity_lag_s"]) for r in est])
    # time from a segment's first reliable observation to its first velocity estimate
    first_rel, first_est = {}, {}
    for r in rows:
        if r["reliable"] != "True":
            continue
        key = (r["track_id"], r["segment"])
        t = float(r["timestamp"])
        first_rel[key] = min(first_rel.get(key, t), t)
        if r["motion_status"] == "estimated":
            first_est[key] = min(first_est.get(key, t), t)
    warmup = [first_est[k] - first_rel[k] for k in first_est]
    return {
        "frame_interval_s": dt, "frame_rate_hz": 1 / dt,
        "window_s": m["window_s"], "window_frame_intervals": m["window_s"] / dt,
        "max_observations_in_window": int(round(m["window_s"] / dt)) + 1,
        "observed_window_n": {"median": float(np.median(n)), "min": int(n.min()), "max": int(n.max())},
        "minimum_to_estimate": {"observations": m["min_observations"], "span_s": m["min_window_span_s"],
                                "span_frames": m["min_window_span_s"] / dt},
        "delay": {"nominal_s": m["window_s"] / 2, "nominal_frames": m["window_s"] / 2 / dt,
                  "measured_median_s": float(np.median(lag)), "measured_median_frames": float(np.median(lag)) / dt,
                  "meaning": "the least-squares slope over a trailing window describes the velocity at the window's "
                             "mean time, so a change in velocity appears about half a window late"},
        "label_persistence": {"s": lab["persistence_s"], "frames": lab["persistence_s"] / dt,
                              "meaning": "a new label is emitted only after it persists this long (adds to the delay)"},
        "warmup_after_segment_start_s": stats(warmup),
        "worst_case_label_delay_s": m["window_s"] / 2 + lab["persistence_s"],
    }


def resets_section(cfg: dict, rows: list[dict], link: dict, reported_total: int) -> tuple[dict, list[dict]]:
    by_track = defaultdict(list)
    for r in rows:
        if r["reliable"] == "True":
            by_track[r["track_id"]].append(r)
    events = []
    for tid, obs in by_track.items():
        obs.sort(key=lambda r: float(r["timestamp"]))
        t = np.array([float(r["timestamp"]) for r in obs])
        xz = np.array([[float(r["raw_world_x"]), float(r["raw_world_z"])] for r in obs])
        seg, reasons = mo.segment_track(t, xz, [r["class_name"] for r in obs], cfg["motion"])
        k = 0
        for i in range(1, len(obs)):
            if seg[i] == seg[i - 1]:
                continue
            a, b = obs[i - 1], obs[i]
            ra, rb = link.get((int(a["step"]), tid)), link.get((int(b["step"]), tid))
            ident = "unknown_not_associated" if ra is None or rb is None else (
                "same_reference_object" if ra == rb else "different_reference_object")
            stored = b.get("reset_reason")
            events.append({"track_id": tid, "class_before": a["class_name"], "class_after": b["class_name"],
                           "reason": reasons[k], "stored_reset_reason": stored if stored is not None else "",
                           "step_before": int(a["step"]), "step_after": int(b["step"]),
                           "gap_s": round(t[i] - t[i - 1], 4), "ground_jump_m": round(float(np.hypot(*(xz[i] - xz[i - 1]))), 4),
                           "reference_identity_across_reset": ident})
            k += 1
    by_cause_class = defaultdict(Counter)
    for e in events:
        by_cause_class[e["reason"]][e["class_before"]] += 1
    mismatched = [e for e in events if e["stored_reset_reason"] not in ("", e["reason"])]
    return {
        "total": len(events), "matches_run_report": len(events) == reported_total,
        "stored_reasons_agree": not mismatched,
        "by_cause": dict(Counter(e["reason"] for e in events)),
        "by_cause_and_detector_class": {k: dict(v) for k, v in by_cause_class.items()},
        "by_detector_class": dict(Counter(e["class_before"] for e in events)),
        "class_changes": dict(Counter(f"{e['class_before']} -> {e['class_after']}" for e in events
                                      if e["reason"] == "class_change")),
        "reference_identity_across_reset_eval_only": dict(Counter(e["reference_identity_across_reset"] for e in events)),
        "causes_defined": {
            "gap": f"no reliable observation for more than {cfg['motion']['max_gap_s']} s",
            "jump": f"ground-plane step > {cfg['motion']['jump_base_m']} m + {cfg['motion']['max_plausible_speed_mps']} m/s x dt",
            "class_change": "the detector class of the track changed",
        },
        "id_switch_note": ("A ByteTrack ID switch is not observable at inference time: a new ID starts a new track, "
                           "not a reset. Class changes are the inference-side sign of an identity problem. Whether a "
                           "reset coincides with a change of reference object is shown separately (evaluation only)."),
    }, events


def per_person_section(cfg: dict, rows: list[dict], link: dict, ref_est: dict) -> dict:
    ev = cfg["evaluation"]
    people = defaultdict(list)
    for r in rows:
        a = link.get((int(r["step"]), r["track_id"]))
        if a is None or a[1] not in ev["person_reference_labels"]:
            continue
        q = ref_est.get((a[0], int(r["step"])))
        if q is not None:
            people[a[0]].append((r, q))
    out = {}
    for iid, prs in sorted(people.items()):
        both = [(r, q) for r, q in prs if r["motion_status"] == "estimated" and q["motion_status"] == "estimated"]
        se = [float(r["speed_mps"]) - float(q["speed_mps"]) for r, q in both]
        de = [angle_diff(float(r["heading_deg"]), float(q["heading_deg"])) for r, q in both
              if float(r["speed_mps"]) >= ev["direction_min_speed_mps"] and float(q["speed_mps"]) >= ev["direction_min_speed_mps"]]
        known = lambda s: s not in ("unknown", "insufficient_history", "")  # noqa: E731
        lab = [(r["motion_label"], q["motion_label"]) for r, q in prs if known(r["motion_label"]) and known(q["motion_label"])]
        ts = [float(r["timestamp"]) for r, _ in both]
        out[str(iid)] = {
            "tracks": sorted({r["track_id"] for r, _ in prs}, key=int),
            "pairs_with_both_estimates": len(both),
            "time_span_s": (max(ts) - min(ts)) if ts else 0.0,
            "approx_independent_windows": round((max(ts) - min(ts)) / cfg["motion"]["window_s"], 1) if ts else 0,
            "reference_speed_mps": stats(float(q["speed_mps"]) for _, q in both),
            "speed_error_mps": stats(se), "abs_speed_error_mps": stats(abs(x) for x in se),
            "heading_error_deg": stats(de),
            "motion_label_agreement": float(np.mean([a == b for a, b in lab])) if lab else None,
            "motion_label_pairs": len(lab),
        }
    return {"n_people": len(out), "people": out,
            "statement": (f"n_people = {len(out)}. Consecutive estimates are highly correlated: frames are 1/60 s apart "
                          f"and each estimate shares most of its {cfg['motion']['window_s']} s window with the next, so "
                          "the pair counts overstate the independent evidence. approx_independent_windows = covered "
                          "time / window length is a rough count of non-overlapping windows.")}


def ttr_section(ev: dict, rows: list[dict]) -> dict:
    t = ev["evaluation"].get("time_to_route")
    if not isinstance(t, dict):
        return {"verdict": "not_evaluated", "detail": t}
    o = t["outcomes_people"]
    entered = o.get("crossing_and_entered", 0)
    censored = {k: v for k, v in o.items() if k.endswith("censored")}
    ts = sorted({float(r["timestamp"]) for r in rows})
    seq = ts[-1] - ts[0] + (ts[1] - ts[0])
    ahead = t.get("realized_time_to_route_s", {}).get("median")
    verdict = (f"inconclusive: {entered} entries, ~{ahead:.2f} s horizon, {seq:.0f} s sequence" if ahead is not None
               else f"inconclusive: {entered} entries, {seq:.0f} s sequence")
    return {"verdict": verdict, "entries_evaluated": entered,
            "realized_time_to_route_s": t.get("realized_time_to_route_s"),
            "error_s_estimated_minus_realized": t["error_s_estimated_minus_realized"],
            "censored_total": sum(censored.values()), "censored": censored,
            "not_censored_other": {k: v for k, v in o.items() if k not in censored and k != "crossing_and_entered"},
            "why_inconclusive": ("Only entries less than half a second ahead could be checked, because the sequence "
                                 f"lasts {seq:.1f} s and most estimates' 5 s horizon runs past its end (censored). "
                                 "This does not test time-to-route at useful look-ahead times."),
            "sequence_duration_s": seq}


def write_markdown(rep: dict, path: Path) -> None:
    w, r, p, t = rep["velocity_window"], rep["resets"], rep["per_person"], rep["time_to_route"]
    L = [f"# Step 8 report supplement — {rep['motion_run']}", "",
         f"**{rep['baseline']}.** Capture {rep['capture']} ({rep['split_of_this_capture']} split).", "",
         "## Velocity window", "",
         f"- Window {w['window_s']} s = {w['window_frame_intervals']:.1f} frame intervals at {w['frame_rate_hz']:.2f} Hz "
         f"(up to {w['max_observations_in_window']} observations; observed median {w['observed_window_n']['median']:.0f}).",
         f"- Needs >= {w['minimum_to_estimate']['observations']} observations spanning >= {w['minimum_to_estimate']['span_s']} s "
         f"({w['minimum_to_estimate']['span_frames']:.0f} frames); first estimate median "
         f"{w['warmup_after_segment_start_s'].get('median', float('nan')):.2f} s after a segment starts.",
         f"- Delay: nominal {w['delay']['nominal_s']:.2f} s ({w['delay']['nominal_frames']:.0f} frames), measured median "
         f"{w['delay']['measured_median_s']:.3f} s; label persistence adds {w['label_persistence']['s']} s "
         f"({w['label_persistence']['frames']:.0f} frames); worst case {w['worst_case_label_delay_s']:.2f} s.", "",
         f"## Resets ({r['total']})", "", "| Cause | " + " | ".join(sorted(r["by_detector_class"])) + " | total |",
         "|---" * (len(r["by_detector_class"]) + 2) + "|"]
    for cause, cnt in sorted(r["by_cause_and_detector_class"].items()):
        L.append(f"| {cause} | " + " | ".join(str(cnt.get(c, 0)) for c in sorted(r["by_detector_class"]))
                 + f" | {sum(cnt.values())} |")
    L += ["", f"- Class changes: {r['class_changes']}",
          f"- Reference identity across the reset (evaluation only): {r['reference_identity_across_reset_eval_only']}",
          f"- {r['id_switch_note']}", "", "## Per person", "", p["statement"], "",
          "| Reference person | Tracks | Pairs | ~independent windows | Ref speed median | Abs speed error median / p90 / max | Heading error median / p90 / max | Label agreement |",
          "|---|---|---|---|---|---|---|---|"]
    for iid, v in p["people"].items():
        s, h = v["abs_speed_error_mps"], v["heading_error_deg"]
        fmt = lambda d, k, f: format(d[k], f) if d.get("n") else "—"  # noqa: E731
        L.append(f"| {iid} | {', '.join(v['tracks'])} | {v['pairs_with_both_estimates']} | {v['approx_independent_windows']} | "
                 f"{fmt(v['reference_speed_mps'], 'median', '.2f')} m/s | {fmt(s, 'median', '.3f')} / {fmt(s, 'p90', '.3f')} / "
                 f"{fmt(s, 'max', '.2f')} m/s | {fmt(h, 'median', '.1f')} / {fmt(h, 'p90', '.1f')} / {fmt(h, 'max', '.0f')} deg | "
                 + ("—" if v["motion_label_agreement"] is None else f"{v['motion_label_agreement']:.0%} of {v['motion_label_pairs']}")
                 + " |")
    L += ["", "## Time to route", "", f"**{t['verdict']}**", "",
          f"- Evaluated entries: {t.get('entries_evaluated')}; censored cases reported separately: {t.get('censored_total')} "
          f"{t.get('censored')}.", f"- {t.get('why_inconclusive', '')}"]
    path.write_text(rp.portable_text("\n".join(L) + "\n", rp.active_roots())[0], encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion-run", required=True)
    a = p.parse_args()
    run = resolve(a.motion_run)
    if not (run / "COMPLETE").exists():
        raise SystemExit(f"Motion run is not complete: {run}")
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    cfg = info["config"]
    office = td.load_config(resolve(cfg["office_config"]))
    rp.set_active_roots(rp.roots_from_config(office))
    fusion = rp.resolve(info["fusion_run"], rp.make_roots(office["dataset_root"], run.parent))
    ev = json.loads((run / "motion_evaluation.json").read_text(encoding="utf-8"))
    rows = read_csv(run / "motion_states.csv")
    link = {(int(r["step"]), r["track_id"]): (int(r["ref_instance_id"]), r["ref_label"])
            for r in read_csv(fusion / "association_evaluation.csv") if r["associated"] == "True"}
    ref_est = {(int(r["ref_instance_id"]), int(r["step"])): r for r in read_csv(run / "reference_motion_eval_only.csv")}
    resets, events = resets_section(cfg, rows, link, ev["coverage"]["resets_total"])
    rep = {
        "created": datetime.now().isoformat(timespec="seconds"), "motion_run": str(run), "fusion_run": str(fusion),
        "baseline": ev["baseline"], "capture": ev["capture"], "split_of_this_capture": ev["split_of_this_capture"],
        "data_splits": ev["data_splits"],
        "velocity_window": window_section(cfg, rows),
        "resets": resets,
        "per_person": per_person_section(cfg, rows, link, ref_est),
        "time_to_route": ttr_section(ev, rows),
    }
    out = resolve(office["output_root"]) / f"motion_supplement_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    write_json(out / "motion_supplement.json", rep)
    with (out / "resets.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(events[0]) if events else ["track_id"])
        wr.writeheader()
        wr.writerows(events)
    write_markdown(rep, out / "motion_supplement.md")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
