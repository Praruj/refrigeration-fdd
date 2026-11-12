import math
import pickle

import pytest
from fastapi.testclient import TestClient

from rfdd import models, service, signals, synth


@pytest.fixture(scope="module")
def client(tmp_path_factory, smoke_days, logic):
    normal = [d for d in smoke_days.values() if d.fault == "normal"]
    path = tmp_path_factory.mktemp("m") / "model.pkl"
    bundle = {
        "hybrid": models.HybridDetector().fit(normal, logic),
        "rules": models.RuleDetector().fit(normal, logic),
        "logic": logic,
        "margin": 1.25,
    }
    with path.open("wb") as f:
        pickle.dump(bundle, f)
    import app.api as api

    api.MODEL, api._bundle = path, None
    return TestClient(api.app)


def _window(fault: str, start: str, end: str, last: int) -> dict:
    tidy = signals.tidy(synth.day("2019-10-21", 16.0, 6, fault, start, end))[service.REQUIRED + service.OPTIONAL].iloc[
        :last
    ]
    return {
        "time": [t.isoformat() for t in tidy.index],
        "signals": {c: [None if math.isnan(v) else float(v) for v in tidy[c]] for c in tidy.columns},
    }


def test_health_and_fault_list(client) -> None:
    assert client.get("/health").json()["model_present"]
    assert {f["key"] for f in client.get("/faults").json()} >= {"cond_blockage", "lt_door_open"}


def test_score_normal_then_fault(client) -> None:
    ok = client.post("/score", json=_window("normal", "00:00", "23:59", 720)).json()
    assert ok["alarm"] is False
    assert ok["diagnosis"] is None
    bad = client.post("/score", json=_window("cond_blockage", "13:05", "15:20", 15 * 60)).json()
    assert bad["alarm"] is True
    assert bad["diagnosis"] == "cond_blockage"
    assert bad["evidence"][0]["feature"] in {"gc_app", "gc_air_dt", "p_gc", "pr_mt", "w_mt", "mt_dis_sh"}


def test_bad_requests(client) -> None:
    assert client.post("/score", json=_window("normal", "00:00", "23:59", 10)).status_code == 422
    w = _window("normal", "00:00", "23:59", 60)
    w["signals"]["nonsense"] = w["signals"]["t_amb"]
    assert client.post("/score", json=w).status_code == 422
