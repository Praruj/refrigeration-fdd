"""Interactive demo: replay a real test day minute by minute and watch the detector think.

streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from rfdd import models, service
from rfdd.evaluate import diagnose
from rfdd.faults import BY_KEY, FAULTS, GROUP_OF
from rfdd.features import BY_NAME, FEATURE_NAMES
from rfdd.prepare import Day

OUT = Path(os.environ.get("RFDD_RESULTS", Path(__file__).resolve().parents[1] / "results" / "ornl"))
DAY_NAMES = {
    "base_A": "Normal day A (Nov, cool)",
    "base_B": "Normal day B (Jul, hot)",
    "base_C": "Day C (Aug, labelled normal; MT case off from 07:25)",
    "base_D": "Normal day D (Nov, cold)",
    "base_E": "Normal day E (Oct; one broken probe)",
    "door": "Fault: LT case door left open",
    "ice": "Fault: ice on the LT evaporator",
    "valve": "Fault: LT expansion valve stuck closed",
    "fan": "Fault: MT evaporator fan failure",
    "cond": "Fault: gas cooler air path blocked",
    "evapblock": "Fault: MT evaporator air path blocked",
}

st.set_page_config(page_title="Refrigeration fault detection", layout="wide")


@st.cache_resource
def load() -> tuple[service.Bundle, pd.DataFrame, dict]:
    bundle = service.load_bundle(OUT / "model.pkl")
    table = pd.read_csv(OUT / "demo_features.csv.gz", index_col=["day", "time"], parse_dates=["time"])
    results = json.loads((OUT / "results.json").read_text())
    return bundle, table, results


@st.cache_data
def replay(day_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    bundle, table, _ = load()
    t = table.loc[day_key]
    day = Day(day_key, "", t, t, t[[]], t, pd.Series(True, index=t.index))
    scores = bundle.hybrid.score(day)
    level = models.persist(scores.max(axis=1), bundle.logic)
    rules_level = models.persist(bundle.rules.score(day).max(axis=1), bundle.logic)
    return t, bundle.hybrid.z(day), bundle.hybrid.expected(day), level, rules_level


def group_name(g: str) -> str:
    names = [f.name for k, f in BY_KEY.items() if GROUP_OF[k] == g]
    return " / ".join(names) if names else "abnormal, but not a fault in the table"


if not (OUT / "model.pkl").exists():
    st.error(f"No model in {OUT}. Run `python -m rfdd.pipeline --config configs/ornl.yaml` first.")
    st.stop()

bundle, table, results = load()
st.title("Fault detection for a supermarket refrigeration system")
st.caption(
    "Open data: ORNL CO2 supermarket rack with two display cases (Sun et al., Scientific Data 2021, CC BY 4.0). "
    "Detector: physics features from the refrigeration cycle + a model of their normal values."
)

tab_replay, tab_table, tab_results = st.tabs(["Replay a test day", "Fault table", "Benchmark"])

with tab_replay:
    days = [k for k in DAY_NAMES if k in table.index.get_level_values("day")]
    key = st.selectbox("Test day", days, format_func=lambda k: DAY_NAMES[k], index=days.index("cond"))
    t, z, expected, level, rules_level = replay(key)
    minute = st.slider("Time of day (minutes after midnight)", 30, len(t) - 1, len(t) - 1, step=5)
    now = t.index[minute]
    history = pd.DataFrame(
        {"Hybrid": level.clip(upper=4.0), "Rules": rules_level.clip(upper=4.0), "Threshold": bundle.margin},
        index=t.index,
    ).iloc[: minute + 1]
    st.line_chart(history, height=260, y_label="alarm level (capped at 4)")

    ev = next((e for e in results["events"] if e["day"] == key), None)
    if ev:
        st.info(f"Labelled fault window: {ev['start']} to {ev['end']}. Evidence: {ev['evidence']}")

    lv = level.iloc[minute]
    on = bool(pd.notna(lv) and lv > bundle.margin)
    c1, c2, c3 = st.columns(3)
    c1.metric("Hybrid alarm", "ON" if on else "off", f"level {lv:.2f}" if pd.notna(lv) else "no valid features")
    c2.metric("Rules alarm", "ON" if rules_level.iloc[minute] > bundle.margin else "off")
    c3.metric("Case states", f"LT {t['lt_state'].iloc[minute]} / MT {t['mt_state'].iloc[minute]}")
    if on:
        scores = bundle.hybrid.score(Day(key, "", t, t, t[[]], t, pd.Series(True, index=t.index)))
        g = diagnose(scores.iloc[: minute + 1], bundle.logic, bundle.margin).iloc[-1]
        st.error(f"Diagnosis: **{group_name(g)}**")

    st.subheader(f"Evidence at {now:%H:%M}: features furthest from their expected value")
    zr = z.iloc[minute].dropna()
    top = zr.abs().sort_values(ascending=False).index[:6]
    st.dataframe(
        pd.DataFrame(
            {
                "feature": top,
                "meaning": [BY_NAME[f].meaning for f in top],
                "now": [t[f].iloc[minute] for f in top],
                "expected": [expected[f].iloc[minute] for f in top],
                "unit": [BY_NAME[f].unit for f in top],
                "z (day-to-day noise units)": [zr[f] for f in top],
            }
        ).round(2),
        hide_index=True,
        use_container_width=True,
    )
    feat = st.selectbox(
        "Plot one feature against its expected value", FEATURE_NAMES, index=FEATURE_NAMES.index("gc_app")
    )
    st.line_chart(
        pd.DataFrame({"measured": t[feat], "expected": expected[feat]}).iloc[: minute + 1],
        height=220,
        y_label=f"{feat} ({BY_NAME[feat].unit})",
    )

with tab_table:
    st.write(
        "Each fault, why it happens, and the symptoms the physics predicts (checked against the data in the README)."
    )
    for f in FAULTS:
        with st.expander(f.name):
            st.write(f.why)
            st.write(", ".join(f"{k} {'up' if v > 0 else 'down'}" for k, v in f.signature.items()))

with tab_results:
    s = pd.DataFrame(results["summary"]).set_index("method")
    st.dataframe(s.round(2), use_container_width=True)
    st.image(str(OUT / "figures" / "tradeoff.png"))
    flags = {k: v for k, v in results["sensor_flags"].items() if v}
    st.write("Sensors flagged by the sensor check (share of the day):", flags)
    st.caption(f"Pipeline runtime {results['runtime_s']} s on a laptop CPU. Numbers: {OUT / 'results.json'}")
