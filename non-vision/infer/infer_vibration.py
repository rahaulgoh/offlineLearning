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

SOURCE_TABLE = "derived_vibration_record"
STATE_TABLE = "edge_infer_state"
SCORE_TABLE = "edge_vibration_anomaly_score"

TAG_COL = "tag_id"
IDX_COL = "idx"
VAL_COL = "value"
TIME_COL = "created_on"

MODEL_PATH = "/opt/edge/models/vibration/model_vibration.pt"
META_PATH  = "/opt/edge/models/vibration/vibration_metadata.json"

POLL_SECONDS = 2.0
FETCH_LIMIT = 2000
MAX_TAGS_PER_CYCLE = 500
TORCH_THREADS = 1

BACKFILL_ROWS = 5000


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
# MODEL UTILS
# ----------------------------
def load_metadata(path: str):
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sensor_type = meta.get("sensor_type", "vibration")
    window_size = int(meta["window_size"])
    zclip = float(meta.get("zscore_clip", 8.0))

    per_tag = {}
    for tag_id, d in meta.get("per_tag", {}).items():
        per_tag[str(tag_id)] = TagMeta(
            mean=float(d["mean"]),
            std=float(d["std"]),
            threshold=float(d["threshold"]),
        )

    return sensor_type, window_size, zclip, per_tag

def zscore_clip(arr: np.ndarray, mean: float, std: float, clip: float) -> np.ndarray:
    s = std if std != 0 else 1.0
    z = (arr - mean) / s
    z = np.clip(z, -clip, clip).astype(np.float32)
    return z

def recon_mse(model, win: np.ndarray) -> float:
    x = torch.from_numpy(win).unsqueeze(0)  # (1, W)
    with torch.no_grad():
        y = model(x)
        err = torch.mean((y - x) ** 2, dim=1).item()
    return float(err)


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

    tags = [t for t in fetch_distinct_tags() if t in meta_by_tag]
    tags = tags[:MAX_TAGS_PER_CYCLE]

    if not tags:
        print("[infer_vibration] No tags found that match metadata. Exiting.")
        return

    state = load_state(sensor_type, tags)

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

    print(f"[infer_vibration] start: tags={len(tags)} window={window_size} poll={POLL_SECONDS}s")

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

                rt.window.append(float(val))
                rt.window_time.append(ts)
                rt.last_idx = int(idx)

                if len(rt.window) < window_size:
                    continue

                raw = np.array(rt.window, dtype=np.float32)
                win = zscore_clip(raw, meta.mean, meta.std, zclip)
                score = recon_mse(model, win)
                is_anom = score > meta.threshold

                score_rows.append((
                    sensor_type,
                    tag_id,
                    rt.last_idx,
                    rt.window_time[-1],
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
            print(f"[infer_vibration] cycle={cycle} inserted={inserted}")

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
