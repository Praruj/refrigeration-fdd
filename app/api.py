"""Small scoring service.

    uvicorn app.api:app --port 8000
    curl localhost:8000/health

POST /score takes a window of 1-minute readings (tidy signal names, SI units; at least 30 minutes) and returns
the alarm state, the diagnosis and the five features furthest from their expected values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rfdd import service
from rfdd.faults import FAULTS

MODEL = Path(os.environ.get("RFDD_MODEL", "results/ornl/model.pkl"))
MIN_MINUTES = 30

app = FastAPI(title="Refrigeration fault detection", version="0.1.0")
_bundle: service.Bundle | None = None


def bundle() -> service.Bundle:
    global _bundle
    if _bundle is None:
        if not MODEL.exists():
            raise HTTPException(503, f"model file {MODEL} not found; run the pipeline first")
        _bundle = service.load_bundle(MODEL)
    return _bundle


class Window(BaseModel):
    time: list[str] = Field(description="ISO timestamps, one per minute, oldest first")
    signals: dict[str, list[float | None]] = Field(description="signal name -> values, same length as time")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": str(MODEL), "model_present": MODEL.exists()}


@app.get("/faults")
def faults() -> list[dict[str, Any]]:
    return [{"key": f.key, "name": f.name, "group": f.group, "why": f.why, "signature": f.signature} for f in FAULTS]


@app.post("/score")
def score(window: Window) -> dict[str, Any]:
    n = len(window.time)
    if n < MIN_MINUTES:
        raise HTTPException(422, f"need at least {MIN_MINUTES} minutes, got {n}")
    bad = [k for k, v in window.signals.items() if len(v) != n]
    if bad:
        raise HTTPException(422, f"signals with the wrong length: {bad}")
    unknown = [k for k in window.signals if k not in service.REQUIRED + service.OPTIONAL]
    if unknown:
        raise HTTPException(422, f"unknown signals: {unknown}")
    frame = pd.DataFrame(window.signals, index=pd.DatetimeIndex(pd.to_datetime(window.time), name="time"), dtype=float)
    return service.score(bundle(), frame)
