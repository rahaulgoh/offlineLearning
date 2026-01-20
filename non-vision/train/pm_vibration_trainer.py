import os, json
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
    "host": "192.168.0.86",
    "port": "5432",
}

TABLE_NAME = "raw_vibration_record"   # <-- your tri-axis table
TAG_COL = "tag_id"
X_COL, Y_COL, Z_COL = "x_value", "y_value", "z_value"   # <-- adjust if your columns differ
TIME_COL = "created_on"
ORDER_COL = "idx"

OUTPUT_FOLDER = "sensor_models/vibration_health"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MAX_ROWS_PER_TAG = 20000
MIN_ROWS_PER_TAG = 2000

# Windowing in SAMPLES (adjust for your sampling rate)
WINDOW_SIZE = 200
STRIDE = 50

# IF model
N_ESTIMATORS = 200
CONTAMINATION = "auto"
RANDOM_STATE = 42

# Quantiles used for health mapping
HEALTH_QUANTS = [0.50, 0.90, 0.95, 0.99]

# ----------------------------
# DB HELPERS
# ----------------------------
def db_fetchall(query: str, params: Tuple = ()) -> List[Tuple]:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        conn.close()

def get_all_tag_ids() -> List[str]:
    q = f"SELECT DISTINCT {TAG_COL} FROM {TABLE_NAME}"
    return [str(r[0]) for r in db_fetchall(q) if r[0] is not None]

def fetch_xyz_for_tag(tag_id: str, limit: int) -> Optional[np.ndarray]:
    q = f"""
        SELECT {X_COL}, {Y_COL}, {Z_COL}
        FROM {TABLE_NAME}
        WHERE {TAG_COL} = %s
        ORDER BY {ORDER_COL} DESC
        LIMIT %s
    """
    rows = db_fetchall(q, (tag_id, limit))
    if not rows:
        return None
    arr = np.array(rows, dtype=np.float64)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.shape[0] < MIN_ROWS_PER_TAG:
        return None
    return arr[::-1].astype(np.float32)  # chronological

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def kurtosis(x: np.ndarray) -> float:
    # excess kurtosis-ish, stable enough for monitoring
    mu = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-8:
        return 0.0
    z = (x - mu) / s
    return float(np.mean(z**4))

def basic_feats(v: np.ndarray) -> Dict[str, float]:
    v = v.astype(np.float64)
    mn = float(np.mean(v))
    sd = float(np.std(v))
    rms = float(np.sqrt(np.mean(v*v)))
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    ptp = vmax - vmin
    maxabs = float(np.max(np.abs(v)))
    cf = float(maxabs / (rms + 1e-8))
    return {
        "mean": mn,
        "std": sd,
        "rms": rms,
        "ptp": ptp,
        "maxabs": maxabs,
        "crest": cf,
        "kurt": kurtosis(v),
    }

def slope(v: np.ndarray) -> float:
    # simple linear regression slope vs index
    n = v.shape[0]
    t = np.arange(n, dtype=np.float64)
    y = v.astype(np.float64)
    t -= np.mean(t)
    y -= np.mean(y)
    denom = np.sum(t*t)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(t*y) / denom)

def window_features(xyz: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    m = np.sqrt(x*x + y*y + z*z)

    feats = {}
    for name, v in [("x", x), ("y", y), ("z", z), ("m", m)]:
        bf = basic_feats(v)
        for k, val in bf.items():
            feats[f"{name}_{k}"] = val

    feats["m_slope"] = slope(m)

    # Cross-axis correlation (handle degenerate variance)
    def corr(a, b) -> float:
        sa, sb = np.std(a), np.std(b)
        if sa < 1e-8 or sb < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0,1])

    feats["corr_xy"] = corr(x, y)
    feats["corr_xz"] = corr(x, z)
    feats["corr_yz"] = corr(y, z)

    names = sorted(feats.keys())
    vec = np.array([feats[n] for n in names], dtype=np.float32)
    return vec, names

def to_feature_windows(xyz: np.ndarray, window: int, stride: int) -> Tuple[np.ndarray, List[str]]:
    n = xyz.shape[0]
    if n < window:
        return np.empty((0,0), dtype=np.float32), []
    X = []
    names = None
    for start in range(0, n - window + 1, stride):
        w = xyz[start:start+window]
        vec, nm = window_features(w)
        if names is None:
            names = nm
        X.append(vec)
    return np.vstack(X).astype(np.float32), (names or [])

# ----------------------------
# HEALTH MAPPING
# ----------------------------
def build_health_mapper(train_scores: np.ndarray) -> Dict:
    # train_scores: higher = worse
    qs = np.quantile(train_scores, HEALTH_QUANTS).astype(np.float64)
    # Map q50->0, q90->50, q95->70, q99->100 (tweakable but sane)
    xq = qs
    yq = np.array([0, 50, 70, 100], dtype=np.float64)
    return {"xq": xq.tolist(), "yq": yq.tolist()}

def score_to_health(raw_score: float, mapper: Dict) -> float:
    xq = np.array(mapper["xq"], dtype=np.float64)
    yq = np.array(mapper["yq"], dtype=np.float64)
    return float(np.clip(np.interp(raw_score, xq, yq, left=0.0, right=100.0), 0.0, 100.0))

# ----------------------------
# MAIN
# ----------------------------
def main():
    tags = get_all_tag_ids()
    print(f"Found {len(tags)} vibration tags.")

    all_X = []
    kept = []
    feature_names = None

    for tag in tags:
        xyz = fetch_xyz_for_tag(tag, MAX_ROWS_PER_TAG)
        if xyz is None:
            continue
        X, names = to_feature_windows(xyz, WINDOW_SIZE, STRIDE)
        if X.shape[0] < 100:
            continue
        if feature_names is None:
            feature_names = names
        all_X.append(X)
        kept.append(tag)

    if not all_X:
        raise RuntimeError("No tags had enough data to train vibration health model.")

    X_train = np.vstack(all_X).astype(np.float32)
    print(f"Training feature windows: {X_train.shape}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    iso = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(Xs)

    # raw_score: higher = worse (invert score_samples where higher means more normal)
    raw_scores = (-iso.score_samples(Xs)).astype(np.float64)

    mapper = build_health_mapper(raw_scores)

    # health threshold default at 70 maps ~q95 by our mapping
    model_path = os.path.join(OUTPUT_FOLDER, "model_vibration_health.joblib")
    joblib.dump({"scaler": scaler, "iso": iso, "feature_names": feature_names, "mapper": mapper}, model_path)

    meta = {
        "sensor_type": "vibration_health",
        "table": TABLE_NAME,
        "tag_col": TAG_COL,
        "x_col": X_COL, "y_col": Y_COL, "z_col": Z_COL,
        "time_col": TIME_COL,
        "order_col": ORDER_COL,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "tags_used": kept,
        "feature_names": feature_names,
        "health_mapper": mapper,
        "note": "raw_score = -IsolationForest.score_samples(scaled_features)"
    }
    meta_path = os.path.join(OUTPUT_FOLDER, "vibration_health_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", model_path)
    print("Saved:", meta_path)

if __name__ == "__main__":
    main()
