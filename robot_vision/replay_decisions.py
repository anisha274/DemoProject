"""Step 9 demo: replay the frozen Step 8 motion output through the decision rules.

"replay with placeholder passage and gestures": people (positions, velocities, motion_trusted) come from a
finished estimate_motion.py run made with the frozen motion config; the passage and the gestures are scripted
placeholders (configs/replay_script_office_capture2.json). The corridor is the THUD proxy_route and the robot
position is camera_proxy (the recorded camera position). Positions are a reference-depth and
reference-camera-pose baseline. This is a demonstration, not an evaluation: no inference is rerun and no
reference data is read.

Usage (from robot_vision/):
    python replay_decisions.py --motion-run outputs/motion_1388f_20261013_223619 \
        --route-config configs/proxy_route_office_capture2.json \
        --script configs/replay_script_office_capture2.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import data_splits as ds
import decision_io as dio
import run_paths as rp
from decision import DecisionEngine, DecisionFrame, PersonState, RouteCorridor

LABEL = "replay with placeholder passage and gestures"
DYNAMIC = "person"


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_persons(path: Path) -> dict[int, list[PersonState]]:
    """PersonState per step from motion_states.csv (rows detected or voted as person)."""
    by_step: dict[int, list[PersonState]] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if DYNAMIC not in (r["class_name"], r["motion_class"]):
                continue
            pos = None
            if r["position_valid"] == "True":
                fx, fz = fnum(r["filt_world_x"]), fnum(r["filt_world_z"])
                pos = (fx, fz) if fx is not None and fz is not None else (fnum(r["raw_world_x"]), fnum(r["raw_world_z"]))
            vx, vz = fnum(r["vel_x_mps"]), fnum(r["vel_z_mps"])
            by_step.setdefault(int(r["step"]), []).append(PersonState(
                int(r["track_id"]), float(r["timestamp"]), pos, fnum(r["distance_m"]), r["motion_label"],
                (vx, vz) if vx is not None and vz is not None else None, r["motion_trusted"] == "True"))
    return by_step


def plot(decisions, out: Path, title_note: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    status = {"MOVE": "#0ca30c", "WAIT": "#fab219", "STOP": "#d03b3b"}
    s1, s2 = "#2a78d6", "#eb6834"
    t = [d.timestamp for d in decisions]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, facecolor=surface,
                             gridspec_kw={"height_ratios": [1.1, 2.2, 1.6], "hspace": 0.12})
    for ax in axes:
        ax.set_facecolor(surface)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=muted, labelcolor=ink, labelsize=9)
        ax.grid(axis="x", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)

    # decision: one labelled lane per decision, so identity never rests on colour alone
    lanes = ["STOP", "WAIT", "MOVE"]
    ax = axes[0]
    dt = [b - a for a, b in zip(t, t[1:])] + [t[-1] - t[-2]]
    for i, name in enumerate(lanes):
        spans, start = [], None
        for k, d in enumerate(decisions):
            if d.decision == name and start is None:
                start = t[k]
            if d.decision != name and start is not None:
                spans.append((start, t[k] - start)); start = None
        if start is not None:
            spans.append((start, t[-1] + dt[-1] - start))
        ax.broken_barh(spans, (i - 0.32, 0.64), facecolors=status[name], edgecolor="none")
        share = sum(d.decision == name for d in decisions) / len(decisions)
        ax.text(1.005, i, f"{share:.0%}", transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=ink)
    ax.set_yticks(range(3), lanes)
    ax.set_ylim(-0.6, 2.6)
    ax.set_title("Decision", loc="left", fontsize=10, color=ink)

    # rule that decided, as a step line with markers
    ax = axes[1]
    rules = [d.rule for d in decisions]
    ax.step(t, rules, where="post", color=muted, linewidth=1)
    ax.scatter(t, rules, s=8, c=[status[d.decision] for d in decisions], linewidths=0, zorder=3)
    names = {1: "1 person in route", 2: "2 person entering", 3: "3 STOP gesture", 4: "4 movement unconfirmed",
             5: "5 passage unconfirmed", 6: "6 load too wide", 7: "7 passage clear (MOVE)", 8: "8 waiting"}
    ax.set_yticks(list(names), list(names.values()))
    ax.set_ylim(8.6, 0.4)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_title("Rule that decided (first match, rules checked 1 to 8)", loc="left", fontsize=10, color=ink)

    # people: observed vs. inside the route or its caution zone
    ax = axes[2]
    obs = [sum(a["observed"] for a in d.snapshot["persons"]) for d in decisions]
    near = [sum(a["in_route"] or a["in_caution_zone"] for a in d.snapshot["persons"]) for d in decisions]
    ax.step(t, obs, where="post", color=s1, linewidth=2, label="people tracked this frame")
    ax.step(t, near, where="post", color=s2, linewidth=2, linestyle=(0, (4, 2)), label="in route or caution zone")
    top = max(max(obs), max(near), 1)
    ax.set_ylim(-0.2, top + 0.8)
    ax.set_yticks(range(0, top + 1))
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=ink, ncol=2)
    ax.set_title("People (latest valid positions, incl. 1 s memory)", loc="left", fontsize=10, color=ink)
    ax.set_xlabel("capture time (s)", color=ink, fontsize=9)

    fig.suptitle(f"Step 9 decision timeline: {LABEL}", x=0.01, ha="left", fontsize=12, color=ink)
    fig.text(0.01, 0.948, title_note, ha="left", va="top", fontsize=8.5, color=muted, linespacing=1.5)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.86, bottom=0.07)
    fig.savefig(out, dpi=130, facecolor=surface)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=dio.DEFAULT_CONFIG)
    p.add_argument("--motion-run", required=True)
    p.add_argument("--route-config", required=True)
    p.add_argument("--script", required=True, help="placeholder passage and scripted gestures")
    p.add_argument("--frozen-record", default="configs/motion_frozen.json")
    a = p.parse_args()

    cfg_path = dio.resolve(a.config)
    cfg, robot, load, req = dio.load_decision_config(cfg_path)
    out_root = dio.resolve(cfg["paths"]["output_root"])
    rts = rp.make_roots(output_root=out_root)
    motion = dio.resolve(a.motion_run)
    if not (motion / "COMPLETE").exists():
        raise SystemExit(f"Motion run is not complete: {motion}")
    minfo = dio.read_json(motion / "run_info.json")
    frozen = dio.read_json(a.frozen_record)
    if minfo["config_sha256"] != frozen["sha256"]:
        raise SystemExit(f"Motion run was not made with the frozen motion config ({frozen['sha256'][:8]}...).")
    splits = ds.load()
    split = ds.guard("Synthetic_Scenes/" + minfo["capture"], False, splits)
    route_cfg = dio.read_json(a.route_config)
    if route_cfg["source"]["capture"] != minfo["capture"]:
        raise SystemExit(f"Route is for {route_cfg['source']['capture']}, motion run is {minfo['capture']}")
    script = dio.read_json(a.script)

    fusion = rp.resolve(minfo["fusion_run"], rts)
    frames = [json.loads(l) for l in (fusion / "frames.jsonl").read_text(encoding="utf-8").splitlines() if l]
    persons = load_persons(motion / "motion_states.csv")
    corridor = RouteCorridor(tuple(tuple(q) for q in route_cfg["centreline_xz_m"]), route_cfg["width_m"],
                             route_cfg["label"])
    engine = DecisionEngine(cfg, req, corridor)
    passage = dio.ScriptedPassage(script["passage"], req["required_width_m"])
    gestures = dio.ScriptedGestures(script["gestures"])
    decisions = []
    for f in frames:
        t = f["timestamp"]
        robot_xz = (f["pose_translation"][0], f["pose_translation"][2])  # camera_proxy
        decisions.append(engine.step(DecisionFrame(t, tuple(persons.get(f["step"], [])), passage.passage_at(t),
                                                   tuple(gestures.gestures_at(t)), robot_xz)))

    out = dio.new_run_dir(cfg, f"decisions_replay_{len(frames)}f")
    paths = [cfg_path, dio.resolve(cfg["paths"]["robot_config"]), dio.resolve(cfg["paths"]["load_config"]),
             dio.resolve(a.route_config), dio.resolve(a.script), dio.resolve(a.frozen_record)]
    hashes = dio.copy_configs(out, paths)
    labels = {"run": LABEL, "positions": "reference-depth and reference-camera-pose baseline (Step 8 frozen run)",
              "corridor": "proxy_route", "robot_position": "camera_proxy", "passage": script["passage_note"],
              "gestures": script["gestures_note"], "load": load["label"]}
    info = dio.run_info(cfg_path, {
        "labels": labels, "config_sha256": hashes, "required_width": req,
        "corridor_width_m": engine.corridor_width_m, "route_config_width_m": route_cfg["width_m"],
        "corridor_width_note": "proxy_route is robot-only (0.80 m); the decision corridor is widened to the "
                               "robot-and-load required width",
        "motion_run": str(motion), "motion_config_sha256": minfo["config_sha256"], "capture": minfo["capture"],
        "split_of_this_capture": split, "fusion_run_frames": str(fusion / "frames.jsonl"),
        "frames": len(frames), "person_rows": sum(len(v) for v in persons.values()),
        "decision_config": cfg, "replay_script": script})
    dio.write_json(out / "run_info.json", info)
    dio.write_decisions(out, decisions, {"labels": labels, "git": info["git"], "required_width": req})
    note = (f"{minfo['capture']} ({split} split) | people: frozen Step 8 motion, reference-depth and "
            f"reference-camera-pose baseline
corridor: proxy_route, {engine.corridor_width_m:.2f} m | robot: "
            f"camera_proxy | passage and gestures: scripted placeholders | load: {load['label']}")
    plot(decisions, out / "decision_timeline.png", note)
    dio.mark_complete(out)
    counts = {}
    for d in decisions:
        counts[(d.decision, d.rule)] = counts.get((d.decision, d.rule), 0) + 1
    for (dname, r), n in sorted(counts.items(), key=lambda kv: kv[0][1]):
        print(f"  {dname} rule {r}: {n} frames ({n / len(decisions):.1%})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
