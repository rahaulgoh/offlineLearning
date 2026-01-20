#! /usr/bin/env python3
"""
vibration_health Inference (time-windowed, resampled, unsupervised)

Reads:
  raw_vibration_record(tag_id, idx, x_value, y_value, z_value, created_on)

Writes:
  edge_infer_state(sensor_type='vibration_health', tag_id, last_idx, updated_on)
  edge_vibration_health_score(sensor_type, tag_id, window_end_idx, window_end_time,
                             raw_score, health_score, health_threshold, is_unhealthy,
                             model_name, created_on)

Key changes vs old version:
- Windowing is TIME-based (WINDOW_SECONDS), not row-count based.
- Startup backfill uses BACKFILL_MINUTES (by created_on), not BACKFILL_ROWS.
- Model input still fixed-length: we RESAMPLE each time window into exactly window_size
  samples (window_size comes from metadata), so your trained model stays valid.

Assumptions:
- metadata.window_size exists (as before).
- metadata.feature_names ordering exists (as before).
- metadata.health_mapper {xq,yq} exists (as before).
- metadata.tags_used exists (as before).
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple
from collections import deque
from datetime import datetime, timedelta, timezone

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

SOURCE_TABLE = "raw_vibration_record"
STATE_TABLE  = "edge_infer_state"
SCORE_TABLE  = "edge_vibration_health_score"

TAG_COL  = "tag_id"
IDX_COL  = "idx"
X_COL    = "x_value"
Y_COL    = "y_value"
Z_COL    = "z_value"
TIME_COL = "created_on"

SENSOR_TYPE = "vibration_health"

MODEL_PATH = "/opt/edge/models/vibration_health/model_vibration_health.joblib"
META_PATH  = "/opt/edge/models/vibration_health/vibration_health_metadata.json"

# Polling & backfill
POLL_SECONDS = 2.0
FETCH_LIMIT = 8000
BACKFILL_MINUTES = 30

# Time-windowing
WINDOW_SECONDS = 60          # health features computed over last 60s of data
EMIT_EVERY_SECONDS = 5       # write a point every 5s (per tag), if enough data

# Buffer safety (how much raw history we keep in RAM per tag)
BUFFER_MAX_SECONDS = 15 * 60  # keep 15 minutes in memory; plenty for a 60s window

DEFAULT_HEALTH_THRESHOLD = 70.0


# ----------------------------
# TYPES
# ----------------------------
@dataclass
class TagRuntime:
    last_idx: int
    # store raw points as deques aligned by index
    ts: Deque[datetime]
    x: Deque[float]
    y: Deque[float]
    z: Deque[float]
    idx: Deque[int]
    next_emit_time: datetime | None


# ----------------------------
# DB UTILS
# ----------------------------
def db_connect():
    return psycopg2.connect(**DB_CONFIG)

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

def fetch_rows_since_idx(tag_id: str, last_idx: int, limit: int):
    # using idx for incremental fetch is fast + avoids timezone weirdness
    q = f"""
        SELECT {IDX_COL}, {X_COL}, {Y_COL}, {Z_COL}, {TIME_COL}
        FROM {SOURCE_TABLE}
        WHERE {TAG_COL} = %s
          AND {IDX_COL} > %s
          AND {X_COL} = {X_COL} AND {Y_COL} = {Y_COL} AND {Z_COL} = {Z_COL}
          AND {X_COL} <> 'Infinity'::float8 AND {X_COL} <> '-Infinity'::float8
          AND {Y_COL} <> 'Infinity'::float8 AND {Y_COL} <> '-Infinity'::float8
          AND {Z_COL} <> 'Infinity'::float8 AND {Z_COL} <> '-Infinity'::float8
        ORDER BY {IDX_COL} ASC
        LIMIT %s;
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (tag_id, last_idx, limit))
            return cur.fetchall()

def get_start_idx_for_backfill(tag_id: str, backfill_minutes: int) -> int:
    # Find the first idx at/after (NOW - backfill_minutes). If none, return 0.
    q = f"""
        SELECT COALESCE(MIN({IDX_COL}), 0)
        FROM {SOURCE_TABLE}
        WHERE {TAG_COL} = %s
          AND {TIME_COL} >= NOW() - (%s || ' minutes')::interval;
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (tag_id, backfill_minutes))
            (start_idx,) = cur.fetchone()
    return int(start_idx or 0)


# ----------------------------
# FEATURES (tri-axis)
# ----------------------------
def _kurtosis(x: np.ndarray) -> float:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-8:
        return 0.0
    z = (x - mu) / sd
    return float(np.mean(z**4))

def _basic(v: np.ndarray):
    v = v.astype(np.float64)
    mu = float(np.mean(v))
    sd = float(np.std(v))
    rms = float(np.sqrt(np.mean(v * v)))
    ptp = float(np.max(v) - np.min(v))
    maxabs = float(np.max(np.abs(v)))
    crest = float(maxabs / (rms + 1e-8))
    kurt = _kurtosis(v)
    return mu, sd, rms, ptp, maxabs, crest, kurt

def _slope(v: np.ndarray) -> float:
    n = v.size
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float32)
    t -= np.mean(t)
    y = v.astype(np.float32) - np.mean(v)
    denom = float(np.sum(t * t))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(t * y) / denom)

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def compute_feature_dict(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    m = np.sqrt(x*x + y*y + z*z)

    fx = _basic(x); fy = _basic(y); fz = _basic(z); fm = _basic(m)

    feats = {
        "corr_xy": _corr(x, y),
        "corr_xz": _corr(x, z),
        "corr_yz": _corr(y, z),

        "m_crest": fm[5], "m_kurt": fm[6], "m_maxabs": fm[4], "m_mean": fm[0],
        "m_ptp": fm[3], "m_rms": fm[2], "m_slope": _slope(m), "m_std": fm[1],

        "x_crest": fx[5], "x_kurt": fx[6], "x_maxabs": fx[4], "x_mean": fx[0],
        "x_ptp": fx[3], "x_rms": fx[2], "x_std": fx[1],

        "y_crest": fy[5], "y_kurt": fy[6], "y_maxabs": fy[4], "y_mean": fy[0],
        "y_ptp": fy[3], "y_rms": fy[2], "y_std": fy[1],

        "z_crest": fz[5], "z_kurt": fz[6], "z_maxabs": fz[4], "z_mean": fz[0],
        "z_ptp": fz[3], "z_rms": fz[2], "z_std": fz[1],
    }
    return feats


# ----------------------------
# RESAMPLING (time -> fixed length)
# ----------------------------
def _to_epoch_seconds(t: datetime) -> float:
    # Ensure timezone-aware; if DB returns naive, treat as local time (still monotonic)
    if t.tzinfo is None:
        return t.timestamp()
    return t.astimezone(timezone.utc).timestamp()

def resample_window(
    ts: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
    t_end: datetime, window_seconds: int, n_samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample irregular samples into exactly n_samples spanning [t_end - window_seconds, t_end].
    Uses linear interpolation on epoch seconds.
    """
    te = _to_epoch_seconds(t_end)
    t0 = te - float(window_seconds)

    # target timeline (inclusive end)
    t_target = np.linspace(t0, te, num=n_samples, dtype=np.float64)

    # Need strictly increasing ts for np.interp
    t_src = np.array([_to_epoch_seconds(t) for t in ts], dtype=np.float64)
    order = np.argsort(t_src)
    t_src = t_src[order]
    x = x[order]; y = y[order]; z = z[order]

    # If too few points, bail
    if t_src.size < 2:
        raise ValueError("Not enough points to resample")

    # clip to window (keep a bit wider so interpolation works)
    mask = (t_src >= t0 - 1.0) & (t_src <= te + 1.0)
    t_src = t_src[mask]; x = x[mask]; y = y[mask]; z = z[mask]
    if t_src.size < 2:
        raise ValueError("Not enough points in window to resample")

    xr = np.interp(t_target, t_src, x)
    yr = np.interp(t_target, t_src, y)
    zr = np.interp(t_target, t_src, z)
    return xr.astype(np.float32), yr.astype(np.float32), zr.astype(np.float32)


# ----------------------------
# HEALTH MAPPING (metadata.health_mapper)
# ----------------------------
def map_health_piecewise(raw_score: float, xq: List[float], yq: List[float]) -> float:
    if not xq or not yq or len(xq) != len(yq):
        return 50.0

    if raw_score <= xq[0]:
        return float(yq[0])
    if raw_score >= xq[-1]:
        return float(yq[-1])

    for i in range(len(xq) - 1):
        x0, x1 = xq[i], xq[i + 1]
        if x0 <= raw_score <= x1:
            y0, y1 = yq[i], yq[i + 1]
            if abs(x1 - x0) < 1e-12:
                return float(y1)
            t = (raw_score - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))

    return 50.0


# ----------------------------
# METADATA LOAD
# ----------------------------
def load_metadata(path: str):
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    window_size = int(meta.get("window_size", 200))
    stride = int(meta.get("stride", 50))
    tags_used = [str(t) for t in meta.get("tags_used", [])]
    feature_names = meta.get("feature_names", []) or []

    hm = meta.get("health_mapper", {}) or {}
    xq = [float(v) for v in hm.get("xq", [])]
    yq = [float(v) for v in hm.get("yq", [])]

    health_threshold = float(meta.get("health_threshold", DEFAULT_HEALTH_THRESHOLD))

    # Optional (if you later add these to metadata)
    window_seconds = int(meta.get("window_seconds", WINDOW_SECONDS))
    emit_every_seconds = int(meta.get("emit_every_seconds", EMIT_EVERY_SECONDS))

    return window_size, stride, tags_used, feature_names, xq, yq, health_threshold, window_seconds, emit_every_seconds


# ----------------------------
# BUFFER MGMT
# ----------------------------
def trim_buffer(rt: TagRuntime, keep_seconds: int):
    if not rt.ts:
        return
    newest = rt.ts[-1]
    cutoff = newest - timedelta(seconds=keep_seconds)
    while rt.ts and rt.ts[0] < cutoff:
        rt.ts.popleft()
        rt.x.popleft()
        rt.y.popleft()
        rt.z.popleft()
        rt.idx.popleft()


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model at {MODEL_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing metadata at {META_PATH}")

    artifact = joblib.load(MODEL_PATH)
    scaler = artifact.get("scaler")
    iso = artifact.get("model") or artifact.get("iso")
    if scaler is None or iso is None:
        raise ValueError("Model artifact must contain 'scaler' and ('model' or 'iso').")

    model_name = os.path.basename(MODEL_PATH)

    window_size, stride, tags_used, feature_names, xq, yq, health_threshold, window_seconds, emit_every_seconds = load_metadata(META_PATH)

    if not tags_used:
        raise ValueError("metadata.tags_used is empty; model has no tags to run on.")

    # NOTE: stride is row-based in old trainer; for time-based emit we use emit_every_seconds instead.
    tags = tags_used
    state = load_state(SENSOR_TYPE, tags)

    runtimes: Dict[str, TagRuntime] = {}
    for tag_id in tags:
        last_idx = int(state.get(tag_id, 0) or 0)

        # Time-based backfill on cold start
        if last_idx <= 0:
            last_idx = max(0, get_start_idx_for_backfill(tag_id, BACKFILL_MINUTES) - 1)

        runtimes[tag_id] = TagRuntime(
            last_idx=last_idx,
            ts=deque(),
            x=deque(),
            y=deque(),
            z=deque(),
            idx=deque(),
            next_emit_time=None,
        )

    print(f"[infer_vibration_health] start: tags={len(tags)} window={window_seconds}s "
          f"n_samples={window_size} emit_every={emit_every_seconds}s poll={POLL_SECONDS}s "
          f"backfill={BACKFILL_MINUTES}min")

    cycle = 0
    while True:
        cycle += 1
        score_rows: List[Tuple] = []
        state_rows: List[Tuple[str, int]] = []
        inserted = 0

        for tag_id in tags:
            rt = runtimes[tag_id]

            rows = fetch_rows_since_idx(tag_id, rt.last_idx, FETCH_LIMIT)
            if rows:
                for idx, xv, yv, zv, ts in rows:
                    if xv is None or yv is None or zv is None:
                        continue
                    x = float(xv); y = float(yv); z = float(zv)
                    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                        continue
                    rt.idx.append(int(idx))
                    rt.ts.append(ts)
                    rt.x.append(x)
                    rt.y.append(y)
                    rt.z.append(z)
                    rt.last_idx = int(idx)

                trim_buffer(rt, keep_seconds=BUFFER_MAX_SECONDS)

            # Establish next_emit_time once we have data
            if rt.next_emit_time is None and rt.ts:
                # Start emitting at the first timestamp we can reasonably score:
                # need at least `window_seconds` coverage ending at emit time.
                rt.next_emit_time = rt.ts[0] + timedelta(seconds=window_seconds)

            # Nothing to do if we still have no usable schedule
            if rt.next_emit_time is None or not rt.ts:
                state_rows.append((tag_id, rt.last_idx))
                continue

            latest_time = rt.ts[-1]
            # Emit as many points as we can up to latest_time
            while rt.next_emit_time <= latest_time:
                t_end = rt.next_emit_time
                t_start = t_end - timedelta(seconds=window_seconds)

                # Slice buffer to just the window
                # (linear scan is ok with small deque; if needed we can optimize)
                ts_list = []
                x_list = []
                y_list = []
                z_list = []
                idx_list = []

                for i in range(len(rt.ts)):
                    if rt.ts[i] < t_start:
                        continue
                    if rt.ts[i] > t_end:
                        break
                    ts_list.append(rt.ts[i])
                    x_list.append(rt.x[i])
                    y_list.append(rt.y[i])
                    z_list.append(rt.z[i])
                    idx_list.append(rt.idx[i])

                # Need enough points to resample reliably
                if len(ts_list) >= 2:
                    try:
                        Xr, Yr, Zr = resample_window(
                            ts=np.array(ts_list, dtype=object),
                            x=np.array(x_list, dtype=np.float64),
                            y=np.array(y_list, dtype=np.float64),
                            z=np.array(z_list, dtype=np.float64),
                            t_end=t_end,
                            window_seconds=window_seconds,
                            n_samples=window_size
                        )

                        feats = compute_feature_dict(Xr, Yr, Zr)
                        F = np.array([feats[n] for n in feature_names], dtype=np.float32)
                        Xs = scaler.transform(F.reshape(1, -1))

                        # trainer used raw_score = -score_samples(...)
                        raw_score = -float(iso.score_samples(Xs)[0])

                        health = map_health_piecewise(raw_score, xq, yq)
                        is_unhealthy = bool(health >= health_threshold)

                        window_end_idx = int(idx_list[-1]) if idx_list else rt.last_idx

                        score_rows.append((
                            SENSOR_TYPE,
                            tag_id,
                            window_end_idx,
                            t_end,
                            raw_score,
                            health,
                            health_threshold,
                            is_unhealthy,
                            model_name,
                        ))
                    except Exception:
                        # If resampling/scoring fails for this emit tick, just skip it.
                        pass

                rt.next_emit_time = rt.next_emit_time + timedelta(seconds=emit_every_seconds)

            state_rows.append((tag_id, rt.last_idx))

        if score_rows:
            insert_scores(score_rows)
            inserted = len(score_rows)

        if state_rows:
            upsert_state(SENSOR_TYPE, state_rows)

        if cycle % 10 == 0:
            print(f"[infer_vibration_health] cycle={cycle} inserted={inserted}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
