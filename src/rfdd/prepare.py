"""Turn cached 1-minute files into everything the detectors need, one `Day` per test day."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import data, features, sensors, signals, synth
from .events import Event


@dataclass
class Day:
    key: str
    fault: str  # file-level label from the data descriptor
    raw: pd.DataFrame  # every live raw channel, original units
    tidy: pd.DataFrame  # named SI signals with untrusted readings removed
    bad: pd.DataFrame  # sensor-check flags
    table: pd.DataFrame  # physics features, drivers, states
    usable: pd.Series  # minutes that may be used as normal training data / counted for false alarms


def at(day_index: pd.DatetimeIndex, clock: str) -> pd.Timestamp:
    return day_index[0].normalize() + pd.Timedelta(clock + ":00")


def build_day(
    key: str, minute: pd.DataFrame, fault: str = "normal", check_sensors: bool = True, mask_states: bool = True
) -> Day:
    tidy_all = signals.tidy(minute)
    bad = sensors.check(tidy_all)
    if not check_sensors:
        bad = bad & False
    tidy = sensors.apply(tidy_all, bad)
    table = features.build(tidy, mask_states)
    return Day(key, fault, signals.raw_sensor_matrix(minute), tidy, bad, table, pd.Series(True, index=minute.index))


def minutes(cfg: dict[str, Any], keys: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """1-minute raw frames and file-level labels, from the real data or (smoke config) the synthetic fixture."""
    if cfg.get("synthetic"):
        smoke = synth.smoke_days()
        smoke = {k: v for k, v in smoke.items() if keys is None or k in keys}
        return {k: v[0] for k, v in smoke.items()}, {k: v[1] for k, v in smoke.items()}
    mins = data.load_days(Path(cfg["raw_dir"]), Path(cfg["minute_dir"]), keys)
    return mins, {k: data.BY_KEY[k].fault for k in mins}


def load(
    cfg: dict[str, Any], keys: list[str] | None = None, check_sensors: bool = True, mask_states: bool = True
) -> dict[str, Day]:
    mins, labels = minutes(cfg, keys)
    days = {k: build_day(k, m, labels[k], check_sensors, mask_states) for k, m in mins.items()}
    for a in cfg.get("anomalies", []):
        if a["day"] in days:
            d = days[a["day"]]
            d.usable.loc[at(d.usable.index, a["start"]) : at(d.usable.index, a["end"])] = False
    return days


def events(cfg: dict[str, Any], days: dict[str, Day]) -> list[Event]:
    out = []
    for e in cfg["events"]:
        if e["day"] in days:
            idx = days[e["day"]].table.index
            out.append(Event(e["day"], e["fault"], e.get("severity", ""), at(idx, e["start"]), at(idx, e["end"])))
    return out
