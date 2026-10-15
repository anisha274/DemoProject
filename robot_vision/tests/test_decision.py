"""Step 9 decision rules: scenario files (one per rule, conflicts, timing, motion trust), required-width maths,
geometry and output files. No dataset needed."""

import ast
import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import decision as dec  # noqa: E402
import decision_io as dio  # noqa: E402

SCENARIOS = sorted((ROOT / "tests" / "scenarios").glob("*.json"))
ROBOT = {"robot_width_m": 0.60, "safety_margin_per_side_m": 0.10}
CORRIDOR = dec.RouteCorridor(((0.0, 0.0), (0.0, 20.0)), 0.9)
CLEAR = dec.PassageState("clear", 1.5, 0.9)


def person(tid, pos, dist=None, label="unknown", vel=None, trusted=False, t=0.0):
    return dec.PersonState(tid, t, pos, dist, label, vel, trusted)


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, _, _, cls.req = dio.load_decision_config()

    def test_every_rule_and_case_has_a_scenario(self):
        names = {p.stem for p in SCENARIOS}
        for need in [f"rule{i}_" for i in range(1, 9)] + [
                "conflict_stop_vs_move", "conflict_person_in_route_with_move", "conflict_narrow_passage_with_move",
                "gesture_timeout", "asymmetric_timing", "unknown_motion", "beyond_motion_range"]:
            self.assertTrue(any(n.startswith(need) for n in names), need)

    def test_scenarios(self):
        for p in SCENARIOS:
            with self.subTest(scenario=p.stem):
                sc = json.loads(p.read_text(encoding="utf-8"))
                self.assertEqual(dio.check_assumptions(sc, self.cfg, self.req), [])
                ds = dio.run_scenario(sc, self.cfg, self.req)
                self.assertEqual(dio.check_expectations(sc, ds), [])
                rules = {e["rule"] for e in sc["expect"]}
                self.assertTrue(rules, "scenario has no expectations")

    def test_every_rule_reached(self):
        seen = set()
        for p in SCENARIOS:
            sc = json.loads(p.read_text(encoding="utf-8"))
            seen |= {(d.decision, d.rule) for d in dio.run_scenario(sc, self.cfg, self.req)}
        self.assertEqual({r for _, r in seen}, set(range(1, 9)))
        self.assertTrue({("STOP", 1), ("STOP", 2), ("STOP", 3), ("STOP", 4), ("WAIT", 5), ("STOP", 6),
                         ("MOVE", 7), ("WAIT", 8)} <= seen)


class RequiredWidthTests(unittest.TestCase):
    def test_centred_load(self):
        r = dec.required_width(ROBOT, {"width_m": 0.70, "lateral_offset_m": 0.0, "label": "configured, not measured"})
        self.assertAlmostEqual(r["required_width_m"], 0.70 + 0.20)
        self.assertEqual((r["left_extent_m"], r["right_extent_m"]), (-0.35, 0.35))
        self.assertEqual(r["load_label"], "configured, not measured")

    def test_off_centre_load(self):
        # robot -0.30..0.30, load 0.20 +- 0.35 = -0.15..0.55 -> 0.85 + 0.20
        r = dec.required_width(ROBOT, {"width_m": 0.70, "lateral_offset_m": 0.20})
        self.assertAlmostEqual(r["required_width_m"], 1.05)
        self.assertEqual((r["left_extent_m"], r["right_extent_m"]), (-0.30, 0.55))
        left = dec.required_width(ROBOT, {"width_m": 0.70, "lateral_offset_m": -0.20})
        self.assertAlmostEqual(left["required_width_m"], 1.05)
        self.assertEqual((left["left_extent_m"], left["right_extent_m"]), (-0.55, 0.30))

    def test_load_narrower_than_robot_and_no_load(self):
        self.assertAlmostEqual(dec.required_width(ROBOT, {"width_m": 0.40, "lateral_offset_m": 0.1})["required_width_m"], 0.80)
        self.assertAlmostEqual(dec.required_width(ROBOT, None)["required_width_m"], 0.80)

    def test_configured_files(self):
        _, robot, load, req = dio.load_decision_config()
        self.assertEqual(load["label"], "configured, not measured")
        self.assertAlmostEqual(req["required_width_m"],
                               max(robot["robot_width_m"] / 2, load["lateral_offset_m"] + load["width_m"] / 2)
                               - min(-robot["robot_width_m"] / 2, load["lateral_offset_m"] - load["width_m"] / 2)
                               + 2 * robot["safety_margin_per_side_m"])


class GeometryAndEdgeCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, _, _, cls.req = dio.load_decision_config()

    def engine(self, corridor=CORRIDOR):
        return dec.DecisionEngine(self.cfg, self.req, corridor)

    def test_polyline_nearest_and_portion(self):
        line = dec.Polyline([(0, 0), (0, 10), (10, 10)])
        self.assertEqual(line.length, 20)
        d, s = line.nearest((1, 5))
        self.assertAlmostEqual(d, 1)
        self.assertAlmostEqual(s, 5)
        d, s = line.nearest((1, 5), s0=12)  # only (2, 10) onwards counts
        self.assertAlmostEqual(d, 26 ** 0.5)
        self.assertAlmostEqual(s, 12)
        d, s = line.nearest((5, 0), s1=4)  # only (0, 0)..(0, 4) counts
        self.assertAlmostEqual(d, 5)
        self.assertAlmostEqual(s, 0)

    def test_person_behind_robot_is_not_in_route(self):
        e = self.engine()
        d = e.step(dec.DecisionFrame(0.0, (person(1, (0.0, 2.0), 3.0),), CLEAR, (), (0.0, 5.0)))
        self.assertEqual((d.decision, d.rule), ("WAIT", 8))
        d = e.step(dec.DecisionFrame(0.1, (person(1, (0.0, 7.0), 2.0),), CLEAR, (), (0.0, 5.0)))
        self.assertEqual((d.decision, d.rule), ("STOP", 1))

    def test_unlocated_person(self):
        e = self.engine()
        d = e.step(dec.DecisionFrame(0.0, (person(1, None, 2.0),), CLEAR, (), (0.0, 0.0)))
        self.assertEqual((d.decision, d.rule), ("STOP", 4))
        self.assertIn("position unknown", d.reason)
        d = e.step(dec.DecisionFrame(0.1, (person(2, None, 8.0),), CLEAR, (), (0.0, 0.0)))
        self.assertEqual((d.decision, d.rule), ("WAIT", 8))
        d = e.step(dec.DecisionFrame(0.2, (person(3, None, None),), CLEAR, (), (0.0, 0.0)))
        self.assertEqual((d.decision, d.rule), ("STOP", 4))

    def test_corridor_never_narrower_than_required(self):
        e = self.engine(dec.RouteCorridor(((0.0, 0.0), (0.0, 20.0)), 0.5))
        self.assertAlmostEqual(e.corridor_width_m, self.req["required_width_m"])
        d = e.step(dec.DecisionFrame(0.0, (person(1, (0.4, 2.0), 2.0),), CLEAR, (), (0.0, 0.0)))
        self.assertEqual(d.rule, 1)  # 0.4 m off-centre: outside a 0.5 m corridor, inside the 0.9 m footprint

    def test_passage_for_other_required_width_is_unknown(self):
        d = self.engine().step(dec.DecisionFrame(0.0, (), dec.PassageState("clear", 1.5, 0.8), (), (0.0, 0.0)))
        self.assertEqual((d.decision, d.rule), ("WAIT", 5))
        self.assertIn("evaluated for 0.80 m", d.reason)

    def test_timestamps_must_increase(self):
        e = self.engine()
        e.step(dec.DecisionFrame(1.0, (), CLEAR))
        with self.assertRaises(ValueError):
            e.step(dec.DecisionFrame(1.0, (), CLEAR))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            dec.PassageState("open", None, None)
        with self.assertRaises(ValueError):
            dec.GestureEvent(0.0, 1, "WAVE", 1.0)

    def test_stop_hold_counts_frames_not_time(self):
        e = self.engine()
        for i, t in enumerate([0.0, 0.01, 0.02]):
            d = e.step(dec.DecisionFrame(t, (), CLEAR, (dec.GestureEvent(t, 1, "STOP", 0.9),)))
        self.assertEqual((d.decision, d.rule), ("STOP", 3))


class IsolationAndOutputTests(unittest.TestCase):
    def test_decision_core_uses_standard_library_only(self):
        tree = ast.parse((ROOT / "decision.py").read_text(encoding="utf-8"))
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertTrue(mods <= {"__future__", "math", "dataclasses"}, mods)

    def test_no_reference_loaders_imported(self):
        banned = {"thud_data", "fuse_objects", "estimate_motion", "validate_depth", "report_motion", "report_step4"}
        for f in ["decision.py", "decision_io.py", "run_decisions.py", "replay_decisions.py"]:
            p = ROOT / f
            if not p.exists():
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            self.assertFalse(mods & banned, f"{f} imports {mods & banned}")

    def test_outputs_written(self):
        cfg, _, _, req = dio.load_decision_config()
        sc = json.loads((ROOT / "tests" / "scenarios" / "rule7_move.json").read_text(encoding="utf-8"))
        ds = dio.run_scenario(sc, cfg, req)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            dio.write_decisions(out, ds, {"scenario": sc["name"]})
            with (out / "decisions.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), len(ds))
            self.assertEqual(rows[-1]["decision"], "MOVE")
            self.assertEqual(rows[-1]["rule_name"], "passage clear")
            log = json.loads((out / "decision_log.json").read_text(encoding="utf-8"))
            self.assertEqual(log["n_decisions"], len(ds))
            self.assertIn("snapshot", log["decisions"][0])
            self.assertIn("persons", log["decisions"][0]["snapshot"])

    def test_new_run_dir_never_reuses(self):
        fixed = datetime(2026, 1, 1, 12, 0, 0)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(dio, "datetime") as dt:
            dt.now.return_value = fixed
            cfg = {"paths": {"output_root": d}}
            first = dio.new_run_dir(cfg, "x")
            (first / "keep.txt").write_text("existing", encoding="utf-8")
            with self.assertRaises(SystemExit):
                dio.new_run_dir(cfg, "x")
            self.assertEqual((first / "keep.txt").read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
