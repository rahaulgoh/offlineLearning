#! /usr/bin/env python3
"""
Temperature Health Inference (feature-based, unsupervised)

Reads:
  raw_temp_record(tag_id, idx, value, created_on)

Writes:
  edge_infer_state(sensor_type='temperature_health', tag_id, last_idx, updated_on)
  edge_temp_health_score(sensor_type, tag_id, window_end_idx, window_end_time,
                         raw_score, health_score, health_threshold, is_unhealthy,
                         model_name, created_on)

Assumptions:
- raw_temp_record.idx is BIGINT and monotonic (autoincrement)
- Model artifact is a joblib dict: {"scaler": StandardScaler, "model": IsolationForest}
- temp_health_metadata.json contains:
    - per_tag_baseline[tag_id] = {mean, std}
    - health_score_mapping raw_score_lo/raw_score_hi (percentile bounds)
    - window_size, stride, feature_names, etc.
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
import joblib


# ----------------------------
# CONFIG
# ----------------------------
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "mt10ma18",
    "host": "192.168.0.86",
    "port": "5432",
}

SOURCE_TABLE = "raw_temp_record"
STATE_TABLE  = "edge_infer_state"
SCORE_TABLE  = "edge_temp_health_score"

TAG_COL  = "tag_id"
IDX_COL  = "idx"
VAL_COL  = "value"
TIME_COL = "created_on"

SENSOR_TYPE = "temperature_health"

MODEL_PATH = "/opt/edge/models/temp_health/model_temp_health.joblib"
META_PATH  = "/opt/edge/models/temp_health/temp_health_metadata.json"

POLL_SECONDS = 2.0
FETCH_LIMIT = 2000
MAX_TAGS_PER_CYCLE = 500
BACKFILL_ROWS = 5000

# Default alert threshold (can be overridden by metadata if you want)
DEFAULT_HEALTH_THRESHOLD = 70.0


# ----------------------------
# TYPES
# ----------------------------
@dataclass(frozen=True)
class TagBaseline:
    mean: float
    std: float

@dataclass
class TagRuntime:
    last_idx: int
    window: Deque[float]
    window_time: Deque[object]


# ----------------------------
# DB UTILS
# ----------------------------
def db_connect():
    return psycopg2.connect(**DB_CONFIG)

def fetch_distinct_tags() -> List[str]:
    q = f"SELECT DISTINCT {TAG_COL} FROM {SOURCE_TABLE}"
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
    return [str(r[0]) for r in rows if r and r[0] is not None]

def load_state(sensor_type: str, tags: List[str]) -> Dict[str, int]:
    out = {t: 0 for t in tags}
    if not tags:
        return out

    q = f"""
        SELECT tag_id, last_idx
        FROM {STATE_TABLE}
        WHERE sensor_type = %s
          AND tag_id = ANY(%s);
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (sensor_type, tags))
            for tag_id, last_idx in cur.fetchall():
                out[str(tag_id)] = int(last_idx)
    return out

def upsert_state(sensor_type: str, rows: List[Tuple[str, int]]) -> None:
    if not rows:
        return

    q = f"""
        INSERT INTO {STATE_TABLE}(sensor_type, tag_id, last_idx)
        VALUES %s
        ON CONFLICT (sensor_type, tag_id)
        DO UPDATE SET last_idx = EXCLUDED.last_idx, updated_on = NOW();
    """
    values = [(sensor_type, tag_id, last_idx) for tag_id, last_idx in rows]

    with db_connect() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, q, values, page_size=500)
        conn.commit()

def insert_scores(rows: List[Tuple]) -> None:
    if not rows:
        return

    q = f"""
        INSERT INTO {SCORE_TABLE}
        (sensor_type, tag_id, window_end_idx, window_end_time,
         raw_score, health_score, health_threshold, is_unhealthy, model_name)
        VALUES %s
        ON CONFLICT (sensor_type, tag_id, window_end_idx) DO NOTHING;
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, q, rows, page_size=500)
        conn.commit()

def get_max_idx_for_tag(tag_id: str) -> int:
    q = f"SELECT COALESCE(MAX({IDX_COL}), 0) FROM {SOURCE_TABLE} WHERE {TAG_COL} = %s;"
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (tag_id,))
            (mx,) = cur.fetchone()
    return int(mx or 0)

def fetch_rows_since(tag_id: str, last_idx: int, limit: int):
    q = f"""
        SELECT {IDX_COL}, {VAL_COL}, {TIME_COL}
        FROM {SOURCE_TABLE}
        WHERE {TAG_COL} = %s
          AND {IDX_COL} > %s
          AND {VAL_COL} = {VAL_COL}
          AND {VAL_COL} <> 'Infinity'::float8
          AND {VAL_COL} <> '-Infinity'::float8
        ORDER BY {IDX_COL} ASC
        LIMIT %s;
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (tag_id, last_idx, limit))
            return cur.fetchall()


# ----------------------------
# FEATURE ENGINEERING
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
        dtype=np.float32,
    )

def normalize_features_per_tag(F: np.ndarray, baseline: TagBaseline) -> np.ndarray:
    G = F.copy()

    s = baseline.std if baseline.std != 0 else 1.0

    # magnitude-ish
    G[0] = (G[0] - baseline.mean) / s  # mean
    G[2] = (G[2] - baseline.mean) / s  # min
    G[3] = (G[3] - baseline.mean) / s  # max
    G[4] = G[4] / s                    # p2p

    # rate-ish
    G[5] = G[5] / s                    # mean_abs_delta
    G[6] = G[6] / s                    # slope_per_sample

    return G.astype(np.float32)


# ----------------------------
# SCORE -> HEALTH MAPPING
# ----------------------------
def raw_to_health(raw_score: float, lo: float, hi: float) -> float:
    """
    IsolationForest.score_samples: higher = more normal.
    Health score: 0..100 where higher = worse.

    Map raw_score in [lo, hi] to [100, 0] linearly (inverted),
    clamp outside.
    """
    if hi <= lo:
        # fallback: if mapping is broken, treat everything as "neutral"
        return 50.0

    # clamp
    r = min(max(raw_score, lo), hi)
    # invert
    t = (r - lo) / (hi - lo)      # 0..1 where 0 is "bad" (low raw)
    health = (1.0 - t) * 100.0
    return float(health)


# ----------------------------
# METADATA LOAD
# ----------------------------
def load_metadata(path: str) -> Tuple[Dict[str, TagBaseline], float, float, float, int]:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    per_tag = {}
    for tag_id, d in meta.get("per_tag_baseline", {}).items():
        per_tag[str(tag_id)] = TagBaseline(mean=float(d["mean"]), std=float(d["std"]))

    mapping = meta.get("health_score_mapping", {})
    lo = float(mapping.get("raw_score_lo"))
    hi = float(mapping.get("raw_score_hi"))

    window_size = int(meta.get("window_size", 120))
    stride = int(meta.get("stride", 10))  # used for when to emit scores
    # Optional override:
    health_threshold = float(meta.get("health_threshold", DEFAULT_HEALTH_THRESHOLD))

    return per_tag, lo, hi, health_threshold, window_size, stride


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model at {MODEL_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing metadata at {META_PATH}")

    artifact = joblib.load(MODEL_PATH)
    scaler = artifact["scaler"]
    model = artifact["model"]
    model_name = os.path.basename(MODEL_PATH)

    baselines, raw_lo, raw_hi, health_threshold, window_size, stride = load_metadata(META_PATH)

    tags = [t for t in fetch_distinct_tags() if t in baselines]
    tags = tags[:MAX_TAGS_PER_CYCLE]

    if not tags:
        print("[infer_temp_health] No tags found that match metadata baselines. Exiting.")
        return

    state = load_state(SENSOR_TYPE, tags)

    runtimes: Dict[str, TagRuntime] = {}
    for tag_id in tags:
        last_idx = state.get(tag_id, 0)
        if last_idx <= 0:
            mx = get_max_idx_for_tag(tag_id)
            last_idx = max(0, mx - BACKFILL_ROWS)

        runtimes[tag_id] = TagRuntime(
            last_idx=last_idx,
            window=deque(maxlen=window_size),
            window_time=deque(maxlen=window_size),
        )

    print(
        f"[infer_temp_health] start: tags={len(tags)} window={window_size} stride={stride} poll={POLL_SECONDS}s"
    )

    cycle = 0
    while True:
        cycle += 1
        score_rows: List[Tuple] = []
        state_rows: List[Tuple[str, int]] = []
        inserted = 0

        for tag_id in tags:
            baseline = baselines[tag_id]
            rt = runtimes[tag_id]

            rows = fetch_rows_since(tag_id, rt.last_idx, FETCH_LIMIT)
            if not rows:
                continue

            for idx, val, ts in rows:
                if val is None:
                    continue

                fv = float(val)
                if not np.isfinite(fv):
                    continue

                rt.window.append(fv)
                rt.window_time.append(ts)
                rt.last_idx = int(idx)

                # wait for full window
                if len(rt.window) < window_size:
                    continue

                # Only emit every 'stride' samples (reduce DB spam)
                # (Emit when window_end_idx is aligned)
                if (rt.last_idx % stride) != 0:
                    continue

                raw = np.array(rt.window, dtype=np.float32)

                # Safety check: window must be finite
                if not np.isfinite(raw).all():
                    continue

                F = window_features(raw)                     # (9,)
                Fn = normalize_features_per_tag(F, baseline)  # (9,)

                Xs = scaler.transform(Fn.reshape(1, -1))      # (1,9)
                raw_score = float(model.score_samples(Xs)[0])

                health = raw_to_health(raw_score, raw_lo, raw_hi)
                is_unhealthy = bool(health >= health_threshold)

                score_rows.append((
                    SENSOR_TYPE,
                    tag_id,
                    rt.last_idx,
                    rt.window_time[-1],
                    raw_score,
                    health,
                    health_threshold,
                    is_unhealthy,
                    model_name,
                ))

            state_rows.append((tag_id, rt.last_idx))

        if score_rows:
            insert_scores(score_rows)
            inserted = len(score_rows)

        if state_rows:
            upsert_state(SENSOR_TYPE, state_rows)

        if cycle % 30 == 0:
            print(f"[infer_temp_health] cycle={cycle} inserted={inserted}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
