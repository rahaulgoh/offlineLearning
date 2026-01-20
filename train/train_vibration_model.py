import os
import json
from typing import Dict, List, Tuple, Optional
from datetime import timezone, datetime

import numpy as np
import psycopg2
import torch
import torch.nn as nn

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

TABLE_NAME = "derived_vibration_record"
TAG_COL = "tag_id"
VALUE_COL = "value"
TIME_COL = "created_on"
ORDER_COL = "idx"

OUTPUT_FOLDER = "sensor_models/vibration"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Data pull
MAX_ROWS_PER_TAG = 10000
MIN_ROWS_PER_TAG = 300

# Windowing 
WINDOW_SIZE = 60
STRIDE = 1

# Training
EPOCHS = 200
LR = 3e-4
GRAD_CLIP_NORM = 1.0

# Thresholding
THRESH_PERCENTILE = 99.0

# Safety
ZSCORE_CLIP = 8.0

# ----------------------------
# DB HELPERS
# ----------------------------
def db_fetchall(query: str, params: Tuple = ()) -> List[Tuple]:
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return rows
    finally:
        if conn is not None:
            conn.close()

def get_all_tag_ids() -> List[str]:
    q = f"SELECT DISTINCT {TAG_COL} FROM {TABLE_NAME}"
    rows = db_fetchall(q)
    out = []
    for (tag,) in rows:
        if tag is None:
            continue
        out.append(str(tag))
    return out

def fetch_series_for_tag(tag_id: str, limit: int) -> Optional[np.ndarray]:
    query = f"""
        SELECT {VALUE_COL}
        FROM {TABLE_NAME}
        WHERE {TAG_COL} = %s
          AND {VALUE_COL} = {VALUE_COL}                  -- filters NaN
          AND {VALUE_COL} <> 'Infinity'::float8
          AND {VALUE_COL} <> '-Infinity'::float8
        ORDER BY {ORDER_COL} DESC
        LIMIT %s;
    """
    rows = db_fetchall(query, (tag_id, limit))
    if not rows:
        return None

    x = np.array([r[0] for r in rows], dtype=np.float64)
    x = x[np.isfinite(x)].astype(np.float32)

    if x.shape[0] < MIN_ROWS_PER_TAG:
        return None

    # Reverse to chronological order (since we pulled DESC)
    return x[::-1]

# ----------------------------
# PREPROCESSING
# ----------------------------
def compute_mean_std(x: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-8:
        sigma = 1.0
    return mu, sigma


def to_windows(z: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    z shape (N,)
    returns windows shape (M, window)
    """
    n = z.shape[0]
    if n < window:
        return np.empty((0, window), dtype=np.float32)

    windows = []
    for start in range(0, n - window + 1, stride):
        windows.append(z[start : start + window])

    if not windows:
        return np.empty((0, window), dtype=np.float32)

    return np.stack(windows).astype(np.float32)


# ----------------------------
# MODEL
# ----------------------------
class WindowAutoEncoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_generic_model(X: np.ndarray) -> WindowAutoEncoder:
    model = WindowAutoEncoder(in_dim=X.shape[1])
    model.train()

    x_tensor = torch.tensor(X, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for _ in range(EPOCHS):
        opt.zero_grad()
        out = model(x_tensor)
        loss = loss_fn(out, x_tensor)

        if not torch.isfinite(loss):
            raise RuntimeError("Loss became non-finite during training. Check input scaling/cleaning.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        opt.step()

    return model


def reconstruction_errors(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        out = model(x_tensor)
        err = torch.mean((out - x_tensor) ** 2, dim=1).cpu().numpy()
    return err.astype(np.float32)


# ----------------------------
# MAIN
# ----------------------------
def main() -> None:
    tag_ids = get_all_tag_ids()
    print(f"Found {len(tag_ids)} sensors. Building training set for generic vibration model.")

    per_tag_stats: Dict[str, Dict] = {}
    per_tag_windows: Dict[str, np.ndarray] = {}
    all_windows_list: List[np.ndarray] = []

    kept_tags = 0

    for tag_id in tag_ids:
        x = fetch_series_for_tag(tag_id, MAX_ROWS_PER_TAG)
        if x is None:
            continue

        mu, sigma = compute_mean_std(x)
        z = (x - mu) / sigma
        z = np.clip(z, -ZSCORE_CLIP, ZSCORE_CLIP).astype(np.float32)

        W = to_windows(z, WINDOW_SIZE, STRIDE)
        if W.shape[0] < 50:
            continue

        per_tag_stats[tag_id] = {
            "mean": mu,
            "std": sigma,
            "rows_used": int(x.shape[0]),
            "windows_used": int(W.shape[0]),
        }
        per_tag_windows[tag_id] = W
        all_windows_list.append(W)
        kept_tags += 1

    if kept_tags == 0:
        raise RuntimeError("No sensors had enough clean data to train.")

    X_train = np.vstack(all_windows_list).astype(np.float32)

    if not np.isfinite(X_train).all():
        raise RuntimeError("Non-finite values detected in training windows after cleaning.")

    print(f"Training windows: {X_train.shape[0]}  Window size: {X_train.shape[1]}  Tags used: {kept_tags}")

    model = train_generic_model(X_train)

    train_err = reconstruction_errors(model, X_train)
    final_loss = float(np.mean(train_err))
    print(f"Training complete. Mean reconstruction error: {final_loss:.6f}")

    # Compute per-tag thresholds using the trained generic model
    thresholds: Dict[str, float] = {}
    for tag_id, W in per_tag_windows.items():
        err = reconstruction_errors(model, W)
        thr = float(np.percentile(err, THRESH_PERCENTILE))
        thresholds[tag_id] = thr

    # Save TorchScript model
    model.eval()
    example = torch.zeros((1, WINDOW_SIZE), dtype=torch.float32)
    traced = torch.jit.trace(model, example)

    model_path = os.path.join(OUTPUT_FOLDER, "model_vibration.pt")
    traced.save(model_path)

    # Save metadata
    metadata = {
        "sensor_type": "vibration",
        "table": TABLE_NAME,
        "tag_col": TAG_COL,
        "value_col": VALUE_COL,
        "time_col": TIME_COL,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "zscore_clip": ZSCORE_CLIP,
        "threshold_percentile": THRESH_PERCENTILE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_windows": int(X_train.shape[0]),
        "mean_recon_error": final_loss,
        "per_tag": {},
    }

    for tag_id, stats in per_tag_stats.items():
        metadata["per_tag"][tag_id] = {
            "mean": stats["mean"],
            "std": stats["std"],
            "threshold": thresholds[tag_id],
            "rows_used": stats["rows_used"],
            "windows_used": stats["windows_used"],
        }

    meta_path = os.path.join(OUTPUT_FOLDER, "vibration_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()