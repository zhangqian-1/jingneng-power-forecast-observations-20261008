"""Training-center states with persistent, causal four-observation confirmation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def seed_columns(station: str) -> list[str]:
    return [f"__state_before::{station}::{name}" for name in ("current", "candidate", "count")]


def add_causal_states(frame: pd.DataFrame, centers_by_station: dict, seed=None) -> pd.DataFrame:
    result = frame.copy()
    breaks = result["ts"].diff().ne(pd.Timedelta(minutes=15)).to_numpy()
    for station, centers in centers_by_station.items():
        centers = np.asarray(centers, dtype=float)
        raw = np.abs(result[station].to_numpy()[:, None] - centers[None, :]).argmin(axis=1)
        current, candidate, count = int(raw[0]), int(raw[0]), 0
        columns = seed_columns(station)
        if seed is not None:
            current, candidate, count = (int(seed[c]) for c in columns)
        states = np.empty(len(raw), dtype=int)
        before = np.empty((len(raw), 3), dtype=int)
        for i, label in enumerate(raw):
            if i > 0 and breaks[i]:
                current, candidate, count = int(label), int(label), 0
            before[i] = current, candidate, count
            if label == current:
                candidate, count = current, 0
            else:
                count = count + 1 if label == candidate else 1
                candidate = int(label)
                if count >= 4:
                    current, count = candidate, 0
            states[i] = current
        result[columns] = before
        result[f"{station}_state"] = states.astype(float)
        result[f"{station}_state_center"] = centers[states]
    return result
