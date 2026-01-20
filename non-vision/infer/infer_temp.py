#!/usr/bin/env python3
"""
infer_temp.py

Reads:
    raw_temp_record(tag_id, idx, value, created_on)

Writes:
    edge_infer_state(sensor_type, tag_id, last_idx, updated_on)
    edge_temp_anomaly_score(sensor_type, tag_id, window_end_idx, window_end_time,
                            score, threshold, is_anomaly, model_name, created_on)

Assumes:
- raw_temp_record.idx is BIGINT and autoincrements
- temp_metadata.json contains per_tag mean/std/threshold info
- model_temp.pt is TorchScript taking (1, WINDOW_SIZE) float32.
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
import torch


# ---------
# CONFIG
#----------

DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "mt10ma18",
    "host": "192.168.0.86",
    "port": "5432",
}

SOURCE_TABLE = "raw_temp_record"
STATE_TABLE = "edge_infer_state"
SCORE_TABLE = "edge_temp_anomaly_score"

TAG_COL = "tag_id"
IDX_COL = "idx"
VAL_COL = "value"
TIME_COL = "created_on"

MODEL_PATH = "/opt/edge/models/temp/model_temp.pt"
META_PATH = "/opt/edge/models/temp/temp_metadata.json"

POLL_SECONDS = 2.0
FETCH_LIMIT = 2000
MAX_TAGS_PER_CYCLE = 500
TORCH_THREADS = 1
BACKFILL_ROWS = 5000

# Your gateway cadence:
EXPECTED_PERIOD_SECONDS = 30.0   # expected spacing between samples
WINDOW_SPAN_FACTOR = 2.0        # allow up to 2× expected span before declaring window "too sparse"

# Numeric safety
MIN_STD = 1e-6


# ----------------------------
# TYPES
# ----------------------------
@dataclass(frozen=True)
class TagMeta:
    mean: float
    std: float
    threshold: float


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
    """
    rows: [(tag_id, last_idx), ...]
    """
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
         score, threshold, is_anomaly, model_name)
        VALUES %s
        ON CONFLICT DO NOTHING;
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
    # Filter NaN and +/-Infinity at SQL level too.
    q = f"""
        SELECT {IDX_COL}, {VAL_COL}, {TIME_COL}
        FROM {SOURCE_TABLE}
        WHERE {TAG_COL} = %s
          AND {IDX_COL} > %s
          AND {VAL_COL} = {VAL_COL}                  -- not NaN
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
# MODEL UTILS
# ----------------------------
def load_metadata(path: str):
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sensor_type = meta.get("sensor_type", "temperature")
    window_size = int(meta["window_size"])
    zclip = float(meta.get("zscore_clip", 8.0))

    per_tag: Dict[str, TagMeta] = {}
    for tag_id, d in meta.get("per_tag", {}).items():
        per_tag[str(tag_id)] = TagMeta(
            mean=float(d["mean"]),
            std=float(d["std"]),
            threshold=float(d["threshold"]),
        )

    return sensor_type, window_size, zclip, per_tag


def zscore_clip(arr: np.ndarray, mean: float, std: float, clip: float) -> np.ndarray:
    # Guard mean/std
    if not np.isfinite(mean) or not np.isfinite(std):
        return np.full_like(arr, np.nan, dtype=np.float32)

    s = std
    if s == 0.0 or abs(s) < MIN_STD:
        s = MIN_STD

    z = (arr - mean) / s

    # If z is poisoned, return it (caller will skip)
    if not np.all(np.isfinite(z)):
        return z.astype(np.float32)

    z = np.clip(z, -clip, clip).astype(np.float32)
    return z


def recon_mse(model, win: np.ndarray) -> float:
    # win should already be checked, but be paranoid
    if not np.all(np.isfinite(win)):
        return float("nan")

    x = torch.from_numpy(win).unsqueeze(0)  # (1, W)

    with torch.no_grad():
        y = model(x)
        if not torch.isfinite(y).all():
            return float("nan")

        err = torch.mean((y - x) ** 2, dim=1)
        if not torch.isfinite(err).all():
            return float("nan")

        return float(err.item())


# ----------------------------
# MAIN
# ----------------------------
def main():
    torch.set_num_threads(TORCH_THREADS)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model at {MODEL_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing metadata at {META_PATH}")

    sensor_type, window_size, zclip, meta_by_tag = load_metadata(META_PATH)

    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    model_name = os.path.basename(MODEL_PATH)

    # Only infer for tags that exist in metadata
    tags = [t for t in fetch_distinct_tags() if t in meta_by_tag]
    tags = tags[:MAX_TAGS_PER_CYCLE]

    if not tags:
        print("[infer_temp] No tags found that match metadata. Exiting.")
        return

    state = load_state(sensor_type, tags)

    runtimes: Dict[str, TagRuntime] = {}
    for tag_id in tags:
        last_idx = state.get(tag_id, 0)

        # If never processed before, start near end (backfill)
        if last_idx <= 0:
            mx = get_max_idx_for_tag(tag_id)
            last_idx = max(0, mx - BACKFILL_ROWS)

        runtimes[tag_id] = TagRuntime(
            last_idx=last_idx,
            window=deque(maxlen=window_size),
            window_time=deque(maxlen=window_size),
        )

    expected_span = window_size * EXPECTED_PERIOD_SECONDS
    max_span = expected_span * WINDOW_SPAN_FACTOR

    print(
        f"[infer_temp] start: tags={len(tags)} window={window_size} "
        f"poll={POLL_SECONDS}s expected_period={EXPECTED_PERIOD_SECONDS}s "
        f"expected_span~{expected_span:.0f}s max_span~{max_span:.0f}s"
    )

    cycle = 0
    while True:
        cycle += 1
        score_rows: List[Tuple] = []
        state_rows: List[Tuple[str, int]] = []
        inserted = 0

        for tag_id in tags:
            meta = meta_by_tag[tag_id]
            rt = runtimes[tag_id]

            rows = fetch_rows_since(tag_id, rt.last_idx, FETCH_LIMIT)
            if not rows:
                continue

            for idx, val, ts in rows:
                if val is None:
                    continue

                # Robust float conversion + finite check
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(fval):
                    continue

                rt.window.append(fval)
                rt.window_time.append(ts)
                rt.last_idx = int(idx)

                # Only score once we have a full window
                if len(rt.window) < window_size:
                    continue

                # TIME-SPAN GUARD: skip sparse/dropout windows
                try:
                    span = (rt.window_time[-1] - rt.window_time[0]).total_seconds()
                except Exception:
                    continue

                if span > max_span:
                    # If your 60-sample window took > ~60 minutes (for 30s sampling),
                    # you're missing lots of samples -> skip scoring
                    continue

                # FINITE GUARD: raw window
                raw = np.array(rt.window, dtype=np.float32)
                if not np.all(np.isfinite(raw)):
                    continue

                # z-score + clip
                win = zscore_clip(raw, meta.mean, meta.std, zclip)
                if not np.all(np.isfinite(win)):
                    continue

                # model score
                score = recon_mse(model, win)
                if not np.isfinite(score):
                    continue

                # threshold must be finite
                if not np.isfinite(meta.threshold):
                    continue

                is_anom = score > meta.threshold

                score_rows.append((
                    sensor_type,
                    tag_id,
                    rt.last_idx,        # window_end_idx
                    rt.window_time[-1], # window_end_time
                    score,
                    meta.threshold,
                    is_anom,
                    model_name,
                ))

            state_rows.append((tag_id, rt.last_idx))

        if score_rows:
            insert_scores(score_rows)
            inserted = len(score_rows)

        if state_rows:
            upsert_state(sensor_type, state_rows)

        if cycle % 30 == 0:
            print(f"[infer_temp] cycle={cycle} inserted={inserted}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
