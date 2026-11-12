"""Figures for the README. Static PNGs, light theme, one y-axis per panel, colour by method in a fixed order."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import features, models, prepare
from .events import AlarmLogic, Event
from .faults import BY_KEY
from .states import DEFROST, RECOVERY

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CRITICAL, NEUTRAL = "#d03b3b", "#f0efec"
# Fixed order, never cycled: the same method keeps the same colour in every figure.
METHOD_COLOR = {
    "rules": "#2a78d6",
    "raw": "#eb6834",
    "hybrid": "#1baf7a",
    "temp_alarm": "#eda100",
    "raw_iforest": "#e87ba4",
    "raw_knn": "#008300",
    "hybrid_no_guard": "#6250d6",
}
METHOD_LABEL = {
    "rules": "Physics rules",
    "raw": "Black box (PCA, raw sensors)",
    "hybrid": "Hybrid (physics + ML)",
    "temp_alarm": "Case temperature alarm",
    "raw_iforest": "Black box (isolation forest)",
    "raw_knn": "Black box (nearest neighbours)",
    "hybrid_no_guard": "Hybrid without range guard",
}
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#86b6ef", NEUTRAL, "#ec835a", "#b42b2b"])


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _hours(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((index - index[0].normalize()).total_seconds() / 3600.0)


def timelines(
    final: dict[str, Any],
    days: dict[str, prepare.Day],
    events: list[Event],
    logic: AlarmLogic,
    margin: float,
    path: Path,
) -> None:
    """Alarm level of rules and hybrid through each fault day, with the labelled fault window."""
    _style()
    n = len(events)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.9 * n + 0.6), sharex=True, squeeze=False)
    cap = 4.0
    for ax, ev in zip(axes[:, 0], events, strict=True):
        day = days[ev.day]
        h = _hours(day.table.index)
        lo, hi = _hours(pd.DatetimeIndex([ev.start, ev.end]))
        ax.axvspan(lo, hi, color=NEUTRAL, zorder=0)
        for m in ("rules", "hybrid"):
            level = models.persist(final[m].score(day).max(axis=1), logic).clip(lower=-0.2, upper=cap)
            ax.plot(h, level, color=METHOD_COLOR[m], lw=2, label=METHOD_LABEL[m])
            alarm = (level > margin).to_numpy()
            y = -0.35 if m == "rules" else -0.65
            ax.fill_between(h, y - 0.12, y + 0.12, where=alarm, color=METHOD_COLOR[m], lw=0, step="mid")
        ax.axhline(margin, color=CRITICAL, lw=1, ls=(0, (4, 3)))
        ax.set_ylim(-0.9, cap + 0.2)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_title(f"{BY_KEY[ev.fault].name}  (file '{ev.day}')", loc="left", fontsize=9)
        ax.set_ylabel("alarm level")
    axes[0, 0].text(0.2, margin + 0.15, "alarm threshold", color=CRITICAL, fontsize=8)
    axes[0, 0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("hour of the test day   (grey band = labelled fault window, bars below 0 = alarm on)")
    axes[-1, 0].set_xticks(range(0, 25, 2))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def tradeoff(sweep: pd.DataFrame, path: Path) -> None:
    """Detection rate against false alarms per day as the threshold moves."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for m in METHOD_COLOR:
        s = sweep[sweep["method"] == m].sort_values(["false_alarms_per_day", "detection_rate"])
        if s.empty:
            continue
        # Best detection reachable at each false-alarm budget (the threshold that achieves it is kept).
        frontier = s.groupby("false_alarms_per_day")["detection_rate"].max().cummax()
        main = m in ("rules", "raw", "hybrid", "temp_alarm")
        ax.step(
            frontier.index,
            frontier.to_numpy(),
            where="post",
            color=METHOD_COLOR[m],
            lw=2.2 if main else 1.4,
            ls="-" if main else (0, (4, 2)),
            label=METHOD_LABEL[m],
        )
        ax.plot(frontier.index, frontier.to_numpy(), "o", color=METHOD_COLOR[m], ms=4)
    ax.set_xlim(-0.1, 4.0)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("false alarms per normal day (held-out days)")
    ax.set_ylabel("share of fault events detected")
    ax.set_title("Best detection each detector reaches within a false-alarm budget (up and left is better)", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def physics_heatmap(physics: pd.DataFrame, path: Path) -> None:
    """Observed change of every predicted symptom, signed so that 'as predicted' is positive."""
    _style()
    p = physics.copy()
    sign = np.where(p["predicted"] == "up", 1.0, -1.0)
    p["agree"] = p["observed_z"] * sign
    grid = p.pivot_table(index="fault", columns="feature", values="agree", aggfunc="first")
    grid = grid[[f for f in features.FEATURE_NAMES if f in grid.columns]]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(grid) + 1.8))
    ax.grid(False)
    vmax = 6.0
    im = ax.imshow(grid.clip(-vmax, vmax).to_numpy(), cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, color=INK if abs(v) < 4 else "#ffffff")
    ax.set_xticks(range(grid.shape[1]), grid.columns, rotation=45, ha="right")
    ax.set_yticks(range(grid.shape[0]), [BY_KEY[f].name for f in grid.index])
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("z, signed: + = moved the way the fault table predicts")
    ax.set_title("Physics check: predicted symptoms vs what the data showed (blank = not predicted)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def defrost_masking(cfg: dict[str, Any], hybrid: models.HybridDetector, logic: AlarmLogic, path: Path) -> None:
    """One normal day: what the MT coil temperature difference looks like with and without state masking."""
    _style()
    key = next(k for k, d in prepare.load(cfg).items() if d.fault == "normal")
    masked = prepare.load(cfg, [key])[key]
    open_ = prepare.load(cfg, [key], mask_states=False)[key]
    h = _hours(masked.table.index)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    state = masked.table["mt_state"]
    for ax in axes:
        for s, color in ((DEFROST, "#cde2fb"), (RECOVERY, NEUTRAL)):
            ax.fill_between(
                h, 0, 1, where=(state == s).to_numpy(), color=color, transform=ax.get_xaxis_transform(), lw=0
            )
    axes[0].plot(h, masked.tidy["t_mtc_sup"], color=INK2, lw=1.2)
    axes[0].set_ylabel("MT case air off coil, °C")
    axes[0].set_title(
        f"Normal day '{key}': blue = defrost, grey = recovery (both masked); 6 MT defrosts a day", loc="left"
    )
    axes[1].plot(h, open_.table["mt_td"], color=METHOD_COLOR["raw"], lw=2, label="30-min mean, no masking")
    axes[1].plot(h, masked.table["mt_td"], color=METHOD_COLOR["hybrid"], lw=2, label="30-min mean, steady minutes only")
    axes[1].set_ylabel("MT coil TD, K")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("hour of the day")
    axes[1].set_xticks(range(0, 25, 2))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
