"""The three detectors, behind one interface.

rules   physics features against fixed engineering margins around each feature's normal value
raw     black-box model on every raw sensor (PCA reconstruction error), no physics, no state logic
hybrid  physics features, with a learned model of what each should be under the current conditions

All three are trained on fault-free days only. No detector ever sees a labelled fault, because each fault
exists on exactly one day and a classifier trained on it would mostly learn which day it is.

Every detector returns scores normalised so that 1.0 is its own alarm threshold before the safety margin.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .events import AlarmLogic
from .faults import GROUP_SIGNATURE, GROUPS, RULE_MARGIN
from .features import ACCURACY_FLOOR, BY_NAME, FEATURE_NAMES, WINDOW
from .prepare import Day

Z_CLIP = 25.0
UNKNOWN = "unknown"


def persist(score: pd.Series, logic: AlarmLogic) -> pd.Series:
    """k-th largest score in the trailing n minutes. An alarm at threshold T is exactly `persist > T`."""
    x = np.where(np.isnan(score.to_numpy(dtype=float)), -np.inf, score.to_numpy(dtype=float))
    padded = np.concatenate([np.full(logic.n - 1, -np.inf), x])
    windows = np.lib.stride_tricks.sliding_window_view(padded, logic.n)
    kth = np.sort(windows, axis=1)[:, -logic.k]
    return pd.Series(np.where(np.isfinite(kth), kth, np.nan), index=score.index)


def symptom_count_scores(signed: pd.DataFrame, features: dict[str, list[str]], need: int = 2) -> pd.DataFrame:
    """Per fault group: the `need`-th largest deviation among its symptoms, each in its predicted direction.

    A score above 1 means at least `need` separate symptoms of that fault are past their margin and none of
    its symptoms is past its margin in the opposite direction.
    """
    out = {}
    for group, names in features.items():
        sig = GROUP_SIGNATURE[group]
        dev = np.column_stack([signed[f].to_numpy() * sig[f] for f in names])
        kth = np.sort(np.where(np.isnan(dev), -np.inf, dev), axis=1)[:, -min(need, len(names))]
        # A technician rules a fault out when one of its symptoms clearly points the other way.
        vetoed = np.nanmin(np.where(np.isnan(dev), np.inf, dev), axis=1) < -1.0
        out[group] = np.where(np.isfinite(kth), np.where(vetoed, np.minimum(kth, 0.0), kth), np.nan)
    return pd.DataFrame(out, index=signed.index)


def evidence_scores(z: pd.DataFrame, min_valid: int = 2) -> pd.DataFrame:
    """Per fault group: evidence along the fault's signature, sum(sign * z) / sqrt(n).

    Symptoms that agree with the fault table add up and symptoms that contradict it subtract. A last column,
    "unknown", is the single largest deviation of any feature in any direction: something is off that the
    fault table does not describe.
    """
    out = {}
    for group in GROUPS:
        sig = GROUP_SIGNATURE[group]
        dev = np.column_stack([z[f].to_numpy() * s for f, s in sig.items()])
        n = (~np.isnan(dev)).sum(axis=1)
        total = np.nansum(dev, axis=1) / np.sqrt(np.maximum(n, 1))
        out[group] = np.where(n >= min_valid, total, np.nan)
    out[UNKNOWN] = z.abs().max(axis=1).to_numpy()
    return pd.DataFrame(out, index=z.index)


def _train_rows(day: Day, cols: list[str]) -> pd.DataFrame:
    return day.table.loc[day.usable, cols]


@dataclass
class RuleDetector:
    """Fault-table rules with fixed margins, the way a service checklist is written.

    Only features that do not swing with the weather are used (superheat, approach, air-side differences),
    because a fixed limit on head pressure or compressor power is just a hot-day detector. The only thing
    learned is each feature's normal value on this unit.
    """

    name: str = "rules"
    nominal: dict[str, float] = field(default_factory=dict)
    features: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, train: list[Day], logic: AlarmLogic) -> RuleDetector:
        pooled = pd.concat([_train_rows(d, FEATURE_NAMES) for d in train])
        self.nominal = pooled.median().to_dict()
        self.features = {g: [f for f in GROUP_SIGNATURE[g] if not BY_NAME[f].weather] for g in GROUPS}
        return self

    def signed(self, day: Day) -> pd.DataFrame:
        t = day.table
        return pd.DataFrame({f: (t[f] - self.nominal[f]) / RULE_MARGIN[f] for f in FEATURE_NAMES}, index=t.index)

    def score(self, day: Day) -> pd.DataFrame:
        return symptom_count_scores(self.signed(day), self.features)


ENVELOPE_SLACK = {"t_amb": 2.0, "t_room": 1.0}  # K outside the training range before the model abstains


def _expand(drivers: pd.DataFrame) -> np.ndarray:
    """Regression terms: outdoor air with hinges (the plant changes control mode as it warms up), room air,
    and a decaying memory of the last defrost of each case."""
    cols = []
    for name in drivers.columns:
        x = drivers[name].to_numpy()
        if name == "t_amb":
            cols += [x, np.maximum(x - 10.0, 0.0), np.maximum(x - 22.0, 0.0)]
        elif name == "t_room":
            cols.append(x)
        else:
            cols.append(np.exp(-x / (120.0 if name == "lt_since" else 45.0)))
    return np.column_stack(cols)


class _Median:
    def fit(self, x: np.ndarray, y: np.ndarray) -> _Median:
        self.m = float(np.median(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.m)


def _candidates() -> dict[str, object]:
    return {
        "median": _Median(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
        "trees": HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.06, random_state=0),
    }


@dataclass
class HybridDetector:
    """Physics features + learned expected values.

    For each feature a small regression predicts its normal value from the operating conditions that feature
    legitimately depends on (outdoor air, room air, time since defrost). The model family is chosen per feature
    by leave-one-day-out error inside the training days. Residuals are scaled by how wrong the model is on a
    day it has not seen, so day-to-day variation is part of the noise estimate, and are then read through the
    same fault table as the rules.

    Outside the range of conditions seen in training the model abstains for that feature instead of
    extrapolating.
    """

    name: str = "hybrid"
    models: dict[str, object] = field(default_factory=dict)
    chosen: dict[str, str] = field(default_factory=dict)
    sigma: dict[str, float] = field(default_factory=dict)
    envelope: dict[str, tuple[float, float]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    min_threshold: float = 3.0
    guard: bool = True

    def _rows(self, day: Day, feature: str, usable_only: bool, guard: bool) -> pd.Series:
        t = day.table
        drivers = list(BY_NAME[feature].drivers)
        ok = t[feature].notna() & t[drivers].notna().all(axis=1)
        if usable_only:
            ok &= day.usable
        if guard and self.guard:
            for d in drivers:
                if d in self.envelope:
                    lo, hi = self.envelope[d]
                    ok &= (t[d] >= lo) & (t[d] <= hi)
        return ok

    def _xy(self, day: Day, feature: str, ok: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        t = day.table
        return _expand(t.loc[ok, list(BY_NAME[feature].drivers)]), t.loc[ok, feature].to_numpy(), t.index[ok]

    @staticmethod
    def _envelope(train: list[Day]) -> dict[str, tuple[float, float]]:
        pooled = pd.concat([d.table.loc[d.usable, list(ENVELOPE_SLACK)] for d in train])
        return {d: (float(pooled[d].min()) - s, float(pooled[d].max()) + s) for d, s in ENVELOPE_SLACK.items()}

    def fit(self, train: list[Day], logic: AlarmLogic) -> HybridDetector:
        ood_z = {d.key: pd.DataFrame(np.nan, index=d.table.index, columns=FEATURE_NAMES) for d in train}
        for f in FEATURE_NAMES:
            # Leave-one-day-out residuals for every candidate, each judged inside the envelope of its own fold.
            resid: dict[str, dict[str, np.ndarray]] = {c: {} for c in _candidates()}
            index: dict[str, pd.Index] = {}
            for held in train:
                rest = [d for d in train if d.key != held.key]
                self.envelope = self._envelope(rest)
                parts = [self._xy(d, f, self._rows(d, f, True, False)) for d in rest]
                parts = [p for p in parts if len(p[1])]
                xh, yh, idx = self._xy(held, f, self._rows(held, f, True, True))
                if not parts or not len(yh):
                    continue
                x, y = np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
                index[held.key] = idx
                for cname, model in _candidates().items():
                    resid[cname][held.key] = yh - model.fit(x, y).predict(xh)  # type: ignore[attr-defined]
            rms = {c: float(np.sqrt(np.mean(np.concatenate(list(r.values())) ** 2))) for c, r in resid.items() if r}
            best = min(rms, key=lambda c: rms[c])
            self.chosen[f] = best
            self.sigma[f] = max(rms[best], ACCURACY_FLOOR[f])
            for key, r in resid[best].items():
                ood_z[key].loc[index[key], f] = r / self.sigma[f]
            parts = [self._xy(d, f, self._rows(d, f, True, False)) for d in train]
            parts = [p for p in parts if len(p[1])]
            x, y = np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
            self.models[f] = _candidates()[best].fit(x, y)  # type: ignore[attr-defined]
        self.envelope = self._envelope(train)
        # Per-group threshold: the highest alarm-level score any training day reached when it was held out.
        held_scores = [evidence_scores(z.clip(-Z_CLIP, Z_CLIP)) for z in ood_z.values()]
        for g in [*GROUPS, UNKNOWN]:
            peak = np.nanmax([persist(s[g], logic).max() for s in held_scores])
            self.thresholds[g] = float(max(peak, self.min_threshold))
        return self

    def z(self, day: Day) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=day.table.index, columns=FEATURE_NAMES)
        for f in FEATURE_NAMES:
            x, y, idx = self._xy(day, f, self._rows(day, f, False, True))
            if len(y):
                out.loc[idx, f] = (y - self.models[f].predict(x)) / self.sigma[f]  # type: ignore[attr-defined]
        return out.clip(-Z_CLIP, Z_CLIP)

    def expected(self, day: Day) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=day.table.index, columns=FEATURE_NAMES)
        t = day.table
        for f in FEATURE_NAMES:
            drivers = list(BY_NAME[f].drivers)
            ok = t[drivers].notna().all(axis=1)
            out.loc[ok, f] = self.models[f].predict(_expand(t.loc[ok, drivers]))  # type: ignore[attr-defined]
        return out

    def score(self, day: Day) -> pd.DataFrame:
        s = evidence_scores(self.z(day))
        return s / pd.Series(self.thresholds)


@dataclass
class RawDetector:
    """Black box: PCA on every raw sensor channel, scored by reconstruction error.

    It gets the same 30-minute trailing average and the same alarm logic as the others, and nothing else:
    no unit conversion, no cycle physics, no defrost states, no sensor check.
    """

    name: str = "raw"
    variance: float = 0.90
    columns: list[str] = field(default_factory=list)
    center: pd.Series | None = None
    scale: pd.Series | None = None
    pca: PCA | None = None
    threshold: float = 1.0

    @staticmethod
    def _smooth(day: Day) -> pd.DataFrame:
        return day.raw.rolling(WINDOW, min_periods=1).mean()

    def _fit_core(self, frames: list[pd.DataFrame]) -> None:
        x = pd.concat(frames)
        self.columns = [c for c in x.columns if x[c].notna().mean() > 0.99 and x[c].std() > 1e-9]
        x = x[self.columns]
        self.center = x.median()
        self.scale = (x.quantile(0.75) - x.quantile(0.25)).replace(0.0, 1.0) / 1.349
        self.pca = PCA(n_components=self.variance, svd_solver="full").fit(self._standardise(x))

    def _standardise(self, x: pd.DataFrame) -> np.ndarray:
        assert self.center is not None and self.scale is not None
        z = (x[self.columns] - self.center) / self.scale
        return z.fillna(0.0).to_numpy()

    def _spe(self, frame: pd.DataFrame) -> pd.Series:
        assert self.pca is not None
        z = self._standardise(frame)
        resid = z - self.pca.inverse_transform(self.pca.transform(z))
        return pd.Series((resid**2).sum(axis=1), index=frame.index)

    def fit(self, train: list[Day], logic: AlarmLogic) -> RawDetector:
        peaks = []
        for held in train:
            self._fit_core([self._smooth(d)[d.usable] for d in train if d.key != held.key])
            peaks.append(persist(self._spe(self._smooth(held)[held.usable]), logic).max())
        self._fit_core([self._smooth(d)[d.usable] for d in train])
        self.threshold = float(np.nanmax(peaks))
        return self

    def score(self, day: Day) -> pd.DataFrame:
        return pd.DataFrame({"any": self._spe(self._smooth(day)) / self.threshold})

    def contributions(self, day: Day, top: int = 5) -> pd.DataFrame:
        """Per-minute names of the sensors that contribute most to the reconstruction error."""
        assert self.pca is not None
        z = self._standardise(self._smooth(day))
        resid = (z - self.pca.inverse_transform(self.pca.transform(z))) ** 2
        order = np.argsort(-resid, axis=1)[:, :top]
        return pd.DataFrame(np.array(self.columns)[order], index=day.raw.index)


@dataclass
class TemperatureAlarm:
    """What a case controller does today: alarm when the case is too warm outside defrost. Reference only."""

    name: str = "temp_alarm"
    lt_limit: float = -15.0
    mt_limit: float = 5.0

    def fit(self, train: list[Day], logic: AlarmLogic) -> TemperatureAlarm:
        return self

    def score(self, day: Day) -> pd.DataFrame:
        t = day.table
        return pd.DataFrame(
            {"lt_case_warm": 1.0 + (t["lt_air"] - self.lt_limit), "mt_case_warm": 1.0 + (t["mt_prod"] - self.mt_limit)}
        )


@dataclass
class RawForest(RawDetector):
    """Black-box variant: isolation forest on the same raw channels."""

    name: str = "raw_iforest"
    forest: IsolationForest | None = None

    def _fit_core(self, frames: list[pd.DataFrame]) -> None:
        x = pd.concat(frames)
        self.columns = [c for c in x.columns if x[c].notna().mean() > 0.99 and x[c].std() > 1e-9]
        self.center, self.scale = x[self.columns].mean(), x[self.columns].std()
        self.forest = IsolationForest(n_estimators=200, random_state=0).fit(self._standardise(x))

    def _spe(self, frame: pd.DataFrame) -> pd.Series:
        assert self.forest is not None
        return pd.Series(-self.forest.score_samples(self._standardise(frame)), index=frame.index)


@dataclass
class RawNeighbours(RawDetector):
    """Black-box variant: mean distance to the 10 nearest normal minutes."""

    name: str = "raw_knn"
    knn: NearestNeighbors | None = None

    def _fit_core(self, frames: list[pd.DataFrame]) -> None:
        x = pd.concat(frames)
        self.columns = [c for c in x.columns if x[c].notna().mean() > 0.99 and x[c].std() > 1e-9]
        self.center, self.scale = x[self.columns].mean(), x[self.columns].std()
        self.knn = NearestNeighbors(n_neighbors=10).fit(self._standardise(x)[::3])

    def _spe(self, frame: pd.DataFrame) -> pd.Series:
        assert self.knn is not None
        return pd.Series(self.knn.kneighbors(self._standardise(frame))[0].mean(axis=1), index=frame.index)


FACTORIES: dict[str, Callable[[], Any]] = {
    "rules": RuleDetector,
    "raw": RawDetector,
    "hybrid": HybridDetector,
    "temp_alarm": TemperatureAlarm,
    "raw_iforest": RawForest,
    "raw_knn": RawNeighbours,
    "hybrid_no_guard": lambda: HybridDetector(name="hybrid_no_guard", guard=False),
}


def make(name: str) -> Any:
    return FACTORIES[name]()
