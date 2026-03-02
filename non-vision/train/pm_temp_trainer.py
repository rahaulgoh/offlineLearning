#!/usr/bin/env python3
import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import psycopg2
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


# ----------------------------
# CONFIG
# ----------------------------
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "mt10ma18",
    "host": "192.168.0.83",
    "port": "5432",
}

TABLE_NAME = "raw_temp_record"
TAG_COL = "tag_id"
VALUE_COL = "value"
TIME_COL = "created_on"
ORDER_COL = "idx"

OUTPUT_FOLDER = "non-vision/sensor_models/temp_health"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MAX_ROWS_PER_TAG = 20000
MIN_ROWS_PER_TAG = 1500

WINDOW_SIZE = 120
STRIDE = 10

N_ESTIMATORS = 300
CONTAMINATION = "auto"
RANDOM_STATE = 42

HEALTH_CLIP_LO = 0.5
HEALTH_CLIP_HI = 99.5


# ----------------------------
# DB helpers
# ----------------------------
def db_fetchall(query: str, params: Tuple = ()) -> List[Tuple]:
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        if conn is not None:
            conn.close()


def get_all_tag_ids() -> List[str]:
    q = f"SELECT DISTINCT {TAG_COL} FROM {TABLE_NAME};"
    rows = db_fetchall(q)
    return [str(t[0]) for t in rows if t and t[0] is not None]


def fetch_series_for_tag(tag_id: str, limit: int) -> Optional[np.ndarray]:
    q = f"""
        SELECT {VALUE_COL}
        FROM {TABLE_NAME}
        WHERE {TAG_COL} = %s
            AND {VALUE_COL} = {VALUE_COL}
            AND {VALUE_COL} <> 'Infinity'::float8
            AND {VALUE_COL} <> '-Infinity'::float8
        ORDER BY {ORDER_COL} DESC
        LIMIT %s;
    """
    rows = db_fetchall(q, (tag_id, limit))
    if not rows:
        return None

    x = np.array([r[0] for r in rows], dtype=np.float64)
    x = x[np.isfinite(x)].astype(np.float32)

    if x.shape[0] < MIN_ROWS_PER_TAG:
        return None

    return x[::-1]


# ----------------------------
# Feature extraction
# ----------------------------
def _slope(y: np.ndarray) -> float:
    n = y.size
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float32)
    t -= np.mean(t)
    yy = y - np.mean(y)
    denom = float(np.sum(t * t))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(t * yy) / denom)


def window_features(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    mn = float(np.min(x))
    mx = float(np.max(x))
    p2p = mx - mn

    dx = np.diff(x)
    mad_delta = float(np.mean(np.abs(dx))) if dx.size else 0.0

    sl = _slope(x)
    t = np.arange(x.size, dtype=np.float32)
    fit = (sl * (t - np.mean(t))) + mu
    resid = x - fit
    resid_std = float(np.std(resid))

    energy = float(np.mean((x - mu) ** 2))

    return np.array(
        [mu, sd, mn, mx, p2p, mad_delta, sl, resid_std, energy],
        dtype=np.float32
    )


FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "peak_to_peak",
    "mean_abs_delta",
    "slope_per_sample",
    "resid_std",
    "energy_centered",
]


def build_feature_matrix(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    n = x.size
    if n < window:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    feats = []
    for start in range(0, n - window + 1, stride):
        w = x[start:start + window]
        feats.append(window_features(w))

    if not feats:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    return np.vstack(feats).astype(np.float32)


# ----------------------------
# Baseline normalization per tag
# ----------------------------
@dataclass
class TagBaseline:
    mean: float
    std: float


def compute_baseline(x: np.ndarray) -> TagBaseline:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-8:
        sd = 1.0
    return TagBaseline(mean=mu, std=sd)


def normalize_features_per_tag(F: np.ndarray, baseline: TagBaseline) -> np.ndarray:
    G = F.copy()

    idx_mean = 0
    idx_min = 2
    idx_max = 3
    idx_p2p = 4

    s = baseline.std if baseline.std != 0 else 1.0
    G[:, idx_mean] = (G[:, idx_mean] - baseline.mean) / s
    G[:, idx_min]  = (G[:, idx_min]  - baseline.mean) / s
    G[:, idx_max]  = (G[:, idx_max]  - baseline.mean) / s
    G[:, idx_p2p]  = G[:, idx_p2p] / s

    idx_mad = 5
    idx_slope = 6
    G[:, idx_mad] = G[:, idx_mad] / s
    G[:, idx_slope] = G[:, idx_slope] / s

    return G.astype(np.float32)


# ----------------------------
# Main training
# ----------------------------
def main():
    tag_ids = get_all_tag_ids()
    print(f"[temp_health_train] Found {len(tag_ids)} temp tag_ids")

    per_tag_baseline: Dict[str, TagBaseline] = {}
    X_all: List[np.ndarray] = []
    windows_used_total = 0
    tags_used = 0

    for tag_id in tag_ids:
        x = fetch_series_for_tag(tag_id, MAX_ROWS_PER_TAG)
        if x is None:
            continue

        baseline = compute_baseline(x)
        F = build_feature_matrix(x, WINDOW_SIZE, STRIDE)
        if F.shape[0] < 100:
            continue

        Fn = normalize_features_per_tag(F, baseline)

        per_tag_baseline[tag_id] = baseline
        X_all.append(Fn)
        windows_used_total += Fn.shape[0]
        tags_used += 1

    if tags_used == 0:
        raise RuntimeError("No tags had enough clean data to train a health model.")

    X = np.vstack(X_all).astype(np.float32)
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite values detected in feature matrix.")

    print(f"[temp_health_train] Training windows={X.shape[0]} features={X.shape[1]} tags_used={tags_used}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(Xs)

    raw_scores = model.score_samples(Xs).astype(np.float32)

    lo = float(np.percentile(raw_scores, HEALTH_CLIP_LO))
    hi = float(np.percentile(raw_scores, HEALTH_CLIP_HI))

    artifact = {
        "scaler": scaler,
        "model": model,
    }
    model_path = os.path.join(OUTPUT_FOLDER, "model_temp_health.joblib")
    joblib.dump(artifact, model_path)

    meta = {
        "sensor_type": "temperature_health",
        "source_table": TABLE_NAME,
        "tag_col": TAG_COL,
        "value_col": VALUE_COL,
        "time_col": TIME_COL,
        "order_col": ORDER_COL,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "feature_names": FEATURE_NAMES,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "tags_used": tags_used,
        "training_windows": int(X.shape[0]),
        "health_score_mapping": {
            "raw_score_percentile_lo": HEALTH_CLIP_LO,
            "raw_score_percentile_hi": HEALTH_CLIP_HI,
            "raw_score_lo": lo,
            "raw_score_hi": hi,
        },
        "per_tag_baseline": {
            tag_id: asdict(baseline) for tag_id, baseline in per_tag_baseline.items()
        },
    }

    meta_path = os.path.join(OUTPUT_FOLDER, "temp_health_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[temp_health_train] Saved model: {model_path}")
    print(f"[temp_health_train] Saved metadata: {meta_path}")
    print(f"[temp_health_train] Raw score mapping lo={lo:.6f} hi={hi:.6f}")


if __name__ == "__main__":
    main()
