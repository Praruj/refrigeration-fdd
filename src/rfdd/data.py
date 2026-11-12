"""Download, clean and resample the ORNL supermarket refrigeration FDD data set.

Source: J. Sun et al., "Dataset of low global warming potential refrigerant refrigeration system for fault
detection and diagnostics", Scientific Data 8, 144 (2021). Files: figshare, DOI 10.6084/m9.figshare.13232627,
CC BY 4.0. One file is one test day at 1 Hz (3 s for the ice test), in degF / psig / kW.

Raw files are never committed. `download` fetches them and checks every MD5 against the figshare record.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FIGSHARE = "https://ndownloader.figshare.com/files/"


@dataclass(frozen=True)
class DayFile:
    key: str  # short name used everywhere in the project
    filename: str
    file_id: int
    md5: str
    fault: str  # "normal" or a fault id from faults.FAULTS
    baseline: str | None = None  # key of the paired baseline day (from the data descriptor)


MANIFEST: tuple[DayFile, ...] = (
    DayFile("base_A", "BaselineTestA.csv", 27479363, "9d07ecbce4489842ef33f9768a7d22be", "normal"),
    DayFile("base_B", "BaselineTestB.csv", 27479366, "60291563ecff0c806462db2e037bd951", "normal"),
    DayFile("base_C", "BaselineTestC.csv", 27479369, "08b318c4c6802ee5b493f823e1549d2d", "normal"),
    DayFile("base_D", "BaselineTestD.csv", 27479375, "69f9cf2df5be434f4c99bb95d97e475c", "normal"),
    DayFile("base_E", "BaselineTestE.csv", 27479381, "caca678b5335ae4ebfbf86317327c51d", "normal"),
    DayFile("door", "Fault1_DisplayCaseDoorOpen.csv", 27479384, "d69d751a0f0c07507286f1414767243a", "lt_door_open",
            "base_A"),
    DayFile("ice", "Fault2_IceAccumulation.csv", 27479387, "8779f7f7c41e0ee79eb5d7d3a8ceb14b", "lt_evap_ice",
            "base_B"),
    DayFile("valve", "Fault3_EvapValveFailure.csv", 27479390, "d690450ace608e7814ad18651460a422", "lt_eev_stuck",
            "base_C"),
    DayFile("fan", "Fault4_MTEvapFanFailure.csv", 27479393, "24c4f30fe254359910abd2f77f977ba6", "mt_evap_fan",
            "base_D"),
    DayFile("cond", "Fault5_CondAPBlock.csv", 27479402, "97cde9128f99462b59aa81836b775587", "cond_blockage",
            "base_E"),
    DayFile("evapblock", "Fault6_MTEvapAPBlock.csv", 27479408, "d508a4d1b5d7164846b7a8ae7fb42262",
            "mt_evap_blockage", "base_A"),
)  # fmt: skip

BY_KEY = {d.key: d for d in MANIFEST}

# From this column on the CSV header lists flow and calculated columns that are dead, empty or misaligned.
FIRST_DEAD_COLUMN = "F-LT-BPHX"
# One baseline file repeats a header name; pandas renames the second copy.
RENAME = {"T-LT_BPHX_C02_EXIT.1": "T-MT_BPHX_C02_EXIT"}
# An open thermocouple logs about -98,000 and a few channels spike to +1,000. Anything outside is not a reading.
VALID_RANGE = (-200.0, 1500.0)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path, keys: list[str] | None = None) -> None:
    """Fetch missing raw files (about 1 GB in total) and verify their MD5."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for d in MANIFEST:
        if keys is not None and d.key not in keys:
            continue
        path = raw_dir / d.filename
        if not path.exists():
            print(f"downloading {d.filename} ...", flush=True)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(f"{FIGSHARE}{d.file_id}", tmp)
            tmp.replace(path)
        if _md5(path) != d.md5:
            raise RuntimeError(f"{path} does not match the figshare MD5; delete it and run again")


def load_raw(path: Path) -> pd.DataFrame:
    """One raw file as a time-indexed frame of real readings (original names and units, bad values as NaN)."""
    df = pd.read_csv(path, low_memory=False)
    df = df.iloc[:, : list(df.columns).index(FIRST_DEAD_COLUMN)].rename(columns=RENAME)
    ts = pd.to_datetime(df.pop("Timestamp"), errors="coerce")
    num = df.apply(pd.to_numeric, errors="coerce")
    num.index = pd.DatetimeIndex(ts, name="time")
    num = num[num.index.notna()].sort_index()
    return num.where((num > VALID_RANGE[0]) & (num < VALID_RANGE[1]))


def to_minutes(raw: pd.DataFrame) -> pd.DataFrame:
    """1-minute means. A minute needs at least 10 valid samples, otherwise it is NaN."""
    grouped = raw.resample("1min")
    mean = grouped.mean()
    return mean.where(grouped.count() >= 10)


def load_days(raw_dir: Path, cache_dir: Path, keys: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """All requested days at 1-minute resolution, cached as compressed CSV after the first read."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for d in MANIFEST:
        if keys is not None and d.key not in keys:
            continue
        cache = cache_dir / f"{d.key}.csv.gz"
        if cache.exists():
            out[d.key] = pd.read_csv(cache, index_col="time", parse_dates=["time"])
        else:
            out[d.key] = to_minutes(load_raw(raw_dir / d.filename))
            out[d.key].to_csv(cache, float_format="%.4f")
    return out
