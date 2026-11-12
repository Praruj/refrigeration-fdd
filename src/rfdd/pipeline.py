"""One command, every number in the README.

python -m rfdd.pipeline --config configs/ornl.yaml     # real data (downloads ~1 GB on first run)
python -m rfdd.pipeline --config configs/smoke.yaml    # synthetic fixture, ~1 minute, used by CI
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import data, evaluate, models, plots, prepare, report, studies
from .events import AlarmLogic
from .features import FEATURE_NAMES

MAIN_METHODS = ["rules", "raw", "hybrid", "temp_alarm"]
EXTRA_METHODS = ["raw_iforest", "raw_knn", "hybrid_no_guard"]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", double_precision=4))


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    out = Path(cfg["out_dir"])
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if not cfg.get("synthetic"):
        data.download(Path(cfg["raw_dir"]))

    logic = AlarmLogic(**cfg["alarm"])
    margin = float(cfg["threshold_margin"])
    days = prepare.load(cfg)
    events = prepare.events(cfg, days)
    normal = [k for k, d in days.items() if d.fault == "normal"]
    print(
        f"[{time.time() - t0:5.0f}s] {len(days)} days loaded ({len(normal)} normal), {len(events)} events", flush=True
    )

    folds = evaluate.run_folds(days, logic, MAIN_METHODS + EXTRA_METHODS)
    ev = evaluate.event_table(folds, days, events, logic, margin)
    fa = evaluate.false_alarm_table(folds, days, logic, margin)
    summary = evaluate.summarise(ev, fa)
    sweep = evaluate.sweep(folds, days, events, logic, cfg["margin_grid"])
    print(f"[{time.time() - t0:5.0f}s] leave-one-normal-day-out benchmark done", flush=True)

    # Final models on every normal day: used for the figures, the physics check, the API and the demo.
    final = {m: models.make(m).fit([days[k] for k in normal], logic) for m in ["rules", "hybrid", "raw"]}
    hybrid: models.HybridDetector = final["hybrid"]
    physics = studies.physics_check(hybrid, days, events)

    minutes, _ = prepare.minutes(cfg)
    sensor_days = cfg.get("sensor_study_days", normal)
    sensor = studies.sensor_study(days, minutes, logic, margin, sensor_days)
    masking = studies.state_mask_value(cfg, logic, cfg["margin_grid"], normal)
    fingerprint = studies.fingerprint_trap(days)
    print(f"[{time.time() - t0:5.0f}s] side studies done", flush=True)

    plots.timelines(final, days, events, logic, margin, out / "figures" / "timelines.png")
    plots.tradeoff(sweep, out / "figures" / "tradeoff.png")
    plots.physics_heatmap(physics, out / "figures" / "physics_check.png")
    plots.defrost_masking(cfg, hybrid, logic, out / "figures" / "defrost_masking.png")

    with (out / "model.pkl").open("wb") as f:
        pickle.dump({"hybrid": hybrid, "rules": final["rules"], "logic": logic, "margin": margin}, f)
    demo = pd.concat({k: d.table for k, d in days.items()}, names=["day", "time"])
    demo.to_csv(out / "demo_features.csv.gz", float_format="%.4f")

    results = {
        "config": {k: v for k, v in cfg.items() if k not in ("events", "anomalies")},
        "events": cfg["events"],
        "anomalies": cfg.get("anomalies", []),
        "summary": _records(summary),
        "event_rows": _records(ev),
        "false_alarm_rows": _records(fa),
        "sweep": _records(sweep),
        "physics_check": _records(physics),
        "sensor_study": _records(sensor),
        "state_masking": _records(masking),
        "fingerprint_trap": _records(fingerprint),
        "hybrid_model": {
            "chosen": hybrid.chosen,
            "sigma": hybrid.sigma,
            "thresholds": hybrid.thresholds,
            "envelope": hybrid.envelope,
        },
        "sensor_flags": {k: {c: float(v) for c, v in d.bad.mean().items() if v > 0} for k, d in days.items()},
        "features": FEATURE_NAMES,
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "results.json").write_text(json.dumps(results, indent=1, default=_json_default))
    (out / "tables.md").write_text(report.tables(results))
    print(f"[{time.time() - t0:5.0f}s] wrote {out}", flush=True)
    print(summary.round(2).to_string(index=False))
    return results


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def figures(cfg: dict[str, Any]) -> None:
    """Redraw the figures from a finished run (results.json + model.pkl) without re-running the benchmark."""
    out = Path(cfg["out_dir"])
    results = json.loads((out / "results.json").read_text())
    with (out / "model.pkl").open("rb") as f:
        bundle = pickle.load(f)
    days = prepare.load(cfg)
    events = prepare.events(cfg, days)
    logic, margin = bundle["logic"], bundle["margin"]
    final = {"rules": bundle["rules"], "hybrid": bundle["hybrid"]}
    plots.timelines(final, days, events, logic, margin, out / "figures" / "timelines.png")
    plots.tradeoff(pd.DataFrame(results["sweep"]), out / "figures" / "tradeoff.png")
    plots.physics_heatmap(pd.DataFrame(results["physics_check"]), out / "figures" / "physics_check.png")
    plots.defrost_masking(cfg, bundle["hybrid"], logic, out / "figures" / "defrost_masking.png")
    (out / "tables.md").write_text(report.tables(results))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/ornl.yaml")
    ap.add_argument("--figures-only", action="store_true", help="redraw figures and tables from a finished run")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    figures(cfg) if args.figures_only else run(cfg)


if __name__ == "__main__":
    main()
