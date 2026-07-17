import os
import sys
# Force UTF-8 console so emoji status prints don't crash on Windows cp1252 terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd
import warnings
from scipy.stats import kendalltau
from scipy.signal import savgol_filter  # 🔧 FIX: Import for smoothing
from core.config import WINDOW_SIZE, STEP_SIZE, EXPECTED_COLS_3D

# Joint Index Mapping
J = {
    'Midspain': 0, 'AnkleLeft': 3, 'AnkleRight': 6, 'ElbowLeft': 9, 'ElbowRight': 12,
    'FootLeft': 15, 'FootRight': 18, 'HandLeft': 21, 'HandRight': 24, 'HandTipLeft': 27,
    'HandTipRight': 30, 'Head': 33, 'HipLeft': 36, 'HipRight': 39, 'KneeLeft': 42,
    'KneeRight': 45, 'Neck': 48, 'ShoulderLeft': 51, 'ShoulderRight': 54, 'SpineBase': 57,
    'SpineShoulder': 60, 'ThumbLeft': 63, 'ThumbRight': 66, 'WristLeft': 69, 'WristRight': 72
}

_HERE = os.path.dirname(os.path.abspath(__file__))
_MW_PATH_CANDIDATES = [
    "mann_whitney_indices.npy",                                
    os.path.join(_HERE, "..", "mann_whitney_indices.npy"),     
    os.path.join(_HERE, "mann_whitney_indices.npy"),           
]

MW_INDICES = None
for _p in _MW_PATH_CANDIDATES:
    if os.path.exists(_p):
        MW_INDICES = np.load(_p).astype(np.int64)
        print(f"✅ Loaded Mann-Whitney indices ({len(MW_INDICES)} features) from {_p}")
        break

def get_vec(frame, joint_name): return frame[J[joint_name] : J[joint_name]+3]

def calc_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    v1_mag, v2_mag = np.linalg.norm(v1), np.linalg.norm(v2)
    if v1_mag == 0 or v2_mag == 0: return 0.0
    dot_prod = np.clip(np.dot(v1, v2) / (v1_mag * v2_mag), -1.0, 1.0)
    return np.degrees(np.arccos(dot_prod))

def calc_dist(p1, p2): return np.linalg.norm(p1 - p2)

def smooth_coordinates(coords_df: pd.DataFrame) -> pd.DataFrame:
    """🔧 FIX: Apply Savitzky-Golay smoothing to reduce MediaPipe jitter."""
    if len(coords_df) < 13:
        return coords_df  # Too short to smooth
    
    smoothed = coords_df.copy()
    for col in smoothed.columns:
        try:
            # Window=13, polyorder=3 — preserves real movement, kills jitter
            smoothed[col] = savgol_filter(smoothed[col].values, window_length=13, polyorder=3)
        except Exception:
            pass  # If a column can't be smoothed, leave it
    return smoothed

def extract_spatiotemporal_features(raw_frames):
    bio_frames = np.zeros((30, 12))
    limit = min(30, len(raw_frames))
    
    for i in range(limit):
        f = raw_frames[i]
        torso_len = calc_dist(get_vec(f, 'SpineShoulder'), get_vec(f, 'SpineBase'))
        if torso_len < 0.01: torso_len = 1.0

        bio_frames[i, 0] = calc_angle(get_vec(f, 'ShoulderLeft'), get_vec(f, 'ElbowLeft'), get_vec(f, 'WristLeft'))
        bio_frames[i, 1] = calc_angle(get_vec(f, 'ShoulderRight'), get_vec(f, 'ElbowRight'), get_vec(f, 'WristRight'))
        bio_frames[i, 2] = calc_angle(get_vec(f, 'HipLeft'), get_vec(f, 'KneeLeft'), get_vec(f, 'AnkleLeft'))
        bio_frames[i, 3] = calc_angle(get_vec(f, 'HipRight'), get_vec(f, 'KneeRight'), get_vec(f, 'AnkleRight'))
        bio_frames[i, 4] = calc_angle(get_vec(f, 'SpineShoulder'), get_vec(f, 'ShoulderLeft'), get_vec(f, 'ElbowLeft'))
        bio_frames[i, 5] = calc_angle(get_vec(f, 'SpineShoulder'), get_vec(f, 'ShoulderRight'), get_vec(f, 'ElbowRight'))
        bio_frames[i, 6] = calc_angle(get_vec(f, 'SpineBase'), get_vec(f, 'HipLeft'), get_vec(f, 'KneeLeft'))
        bio_frames[i, 7] = calc_angle(get_vec(f, 'SpineBase'), get_vec(f, 'HipRight'), get_vec(f, 'KneeRight'))
        bio_frames[i, 8] = calc_dist(get_vec(f, 'WristLeft'), get_vec(f, 'WristRight')) / torso_len
        bio_frames[i, 9] = calc_dist(get_vec(f, 'AnkleLeft'), get_vec(f, 'AnkleRight')) / torso_len
        bio_frames[i, 10] = calc_dist(get_vec(f, 'Head'), get_vec(f, 'WristLeft')) / torso_len
        bio_frames[i, 11] = calc_dist(get_vec(f, 'Head'), get_vec(f, 'WristRight')) / torso_len

    velocity = np.diff(bio_frames, axis=0) 
    acceleration = np.diff(velocity, axis=0) 

    time_sequence = np.arange(30)
    tau_time = np.zeros(12)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in range(12):
            tau_stat, _ = kendalltau(time_sequence, bio_frames[:, col])
            tau_time[col] = tau_stat if not np.isnan(tau_stat) else 0.0

        tau_sym = np.zeros(4)
        tau_sym[0], _ = kendalltau(bio_frames[:, 0], bio_frames[:, 1])
        tau_sym[1], _ = kendalltau(bio_frames[:, 2], bio_frames[:, 3])
        tau_sym[2], _ = kendalltau(bio_frames[:, 4], bio_frames[:, 5])
        tau_sym[3], _ = kendalltau(bio_frames[:, 6], bio_frames[:, 7])
        tau_sym = np.nan_to_num(tau_sym, nan=0.0)

    flattened_bio = bio_frames.flatten()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(bio_frames, axis=0)
        stds = np.nanstd(bio_frames, axis=0)
        mean_vel = np.nanmean(np.abs(velocity), axis=0) 
        max_vel = np.nanmax(np.abs(velocity), axis=0) 
        mean_acc = np.nanmean(np.abs(acceleration), axis=0) 
        max_acc = np.nanmax(np.abs(acceleration), axis=0) 

    spatiotemporal_features = np.concatenate([
        flattened_bio, means, stds, mean_vel, max_vel, mean_acc, max_acc, tau_time, tau_sym
    ])
    
    return np.nan_to_num(spatiotemporal_features, nan=0.0)

def prepare_inference_data(raw_df: pd.DataFrame, return_full: bool = False):
    if not all(col in raw_df.columns for col in EXPECTED_COLS_3D):
        print("Warning: Missing expected columns in raw data.")
        empty = np.empty((0, 0), dtype=np.float32)
        return (empty, [], empty) if return_full else (empty, [])

    coords_df = raw_df[EXPECTED_COLS_3D].copy()

    # 🔧 FIX: SMOOTH the coordinates before windowing to kill MediaPipe jitter
    coords_df = smooth_coordinates(coords_df)

    frames = coords_df.values
    total_frames = len(frames)

    if total_frames < WINDOW_SIZE:
        print(f"Video too short: {total_frames} frames. Minimum is {WINDOW_SIZE}.")
        empty = np.empty((0, 0), dtype=np.float32)
        return (empty, [], empty) if return_full else (empty, [])

    if MW_INDICES is None:
        raise ValueError("mann_whitney_indices.npy is missing. Cannot filter 448 features to 376 for XGBoost.")

    feats, feats_full, timestamps = [], [], []
    for start_idx in range(0, total_frames - WINDOW_SIZE + 1, STEP_SIZE):
        window = frames[start_idx : start_idx + WINDOW_SIZE]
        full_448 = extract_spatiotemporal_features(window)
        feats.append(full_448[MW_INDICES])
        feats_full.append(full_448)
        timestamps.append(round((start_idx + WINDOW_SIZE / 2.0) / 30.0, 2))

    X = np.asarray(feats, dtype=np.float32)
    if return_full:
        return X, timestamps, np.asarray(feats_full, dtype=np.float32)
    return X, timestamps


ENGINEERED_FEATURE_COUNT = 448  # full spatiotemporal vector before Mann-Whitney selection


def clean_kinect_coords(coords_df: pd.DataFrame) -> pd.DataFrame:
    """Training-faithful cleaning (matches the original parse_v5_skeleton_file):
    Kinect writes 0.0 for untracked joints, so treat 0.0 as missing, interpolate
    across the gap, then median-smooth. Without this, missing-joint zeros corrupt
    the angle/velocity features and skew the prediction."""
    c = coords_df.apply(pd.to_numeric, errors="coerce")
    c = c.replace(0.0, np.nan)
    c = c.interpolate(method="linear", limit_direction="both")
    c = c.rolling(window=3, min_periods=1, center=True).median()
    return c.ffill().bfill().fillna(0.0)


def _resample_df_to_fps(df: pd.DataFrame, src_fps: float, target_fps: float = 30.0) -> pd.DataFrame:
    """Linearly resample a per-frame coordinate dataframe to a constant fps (local copy
    of the pose-extraction resampler, kept here to avoid importing the MediaPipe module)."""
    if df is None or len(df) < 2 or not src_fps or src_fps <= 0 or np.isnan(src_fps) or abs(src_fps - target_fps) < 0.1:
        return df
    n = len(df)
    n2 = max(1, int(round(n / src_fps * target_fps)))
    si = np.linspace(0.0, n - 1, n)
    ti = np.linspace(0.0, n - 1, n2)
    return pd.DataFrame({c: np.interp(ti, si, df[c].values) for c in df.columns})


def fps_from_timestamps(ts_series) -> float:
    """Derive frame rate from an 'H:M:S:MS' timestamp column; default 30 if unparseable."""
    import re
    def to_sec(s):
        a = re.split("[:.]", str(s).replace(")", ""))
        try:
            return int(a[0]) * 3600 + int(a[1]) * 60 + int(a[2]) + int(a[3]) / 1000.0
        except Exception:
            return None
    t = [x for x in (to_sec(v) for v in ts_series) if x is not None]
    if len(t) < 2:
        return 30.0
    d = np.diff(t)
    d = d[d > 0]
    return float(1.0 / np.median(d)) if len(d) else 30.0


def prepare_csv_data(raw_df: pd.DataFrame, return_full: bool = False):
    """Handle the two CSV upload formats robustly, returning (X[:,376], timestamps, kind).

      * "coordinates": 75 raw Kinect 3D columns -> clean -> window -> 448 features -> MW-select.
      * "features":    >=448 pre-engineered feature columns -> MW-select directly.

    Raises ValueError with a clear message for unrecognized formats so the caller can
    return a 400 instead of silently scoring garbage.
    """
    if MW_INDICES is None:
        raise ValueError("mann_whitney_indices.npy is missing; cannot select model features.")

    # Derive frame rate from a timestamp column (before dropping it) so coordinate
    # CSVs get the same fps normalization the video path uses.
    time_cols = [c for c in raw_df.columns if "timestamp" in str(c).lower() or "h:m:s" in str(c).lower()]
    src_fps = fps_from_timestamps(raw_df[time_cols[0]]) if time_cols else 30.0

    work = raw_df.drop(columns=time_cols, errors="ignore")
    cols = set(work.columns)

    # --- Case 1: raw 3D coordinates ---
    if len(set(EXPECTED_COLS_3D) & cols) >= 70:
        missing = [c for c in EXPECTED_COLS_3D if c not in cols]
        if missing:
            raise ValueError(f"CSV is missing {len(missing)} required 3D coordinate columns (e.g. {missing[:3]}).")
        # Mirror the video path exactly: clean -> resample to 30 fps -> savgol smooth.
        # This is what makes keypoints CSVs (MediaPipe-convention) score like the video
        # path, and is a no-op-ish pass for genuine 30 fps Kinect data.
        coords = clean_kinect_coords(work[EXPECTED_COLS_3D])
        coords = _resample_df_to_fps(coords, src_fps)
        coords = smooth_coordinates(coords)
        frames = coords.values
        if len(frames) < WINDOW_SIZE:  # pad short clips, mirroring the training parser
            pad = np.repeat(frames[-1:], WINDOW_SIZE - len(frames), axis=0)
            frames = np.vstack([frames, pad])
        feats, feats_full, ts = [], [], []
        for s in range(0, len(frames) - WINDOW_SIZE + 1, STEP_SIZE):
            f448 = extract_spatiotemporal_features(frames[s:s + WINDOW_SIZE])
            feats.append(f448[MW_INDICES])
            feats_full.append(f448)
            ts.append(round((s + WINDOW_SIZE / 2.0) / 30.0, 2))
        X = np.asarray(feats, dtype=np.float32)
        if return_full:
            return X, ts, "coordinates", np.asarray(feats_full, dtype=np.float32)
        return X, ts, "coordinates"

    # --- Case 2: pre-engineered feature export ---
    numeric = work.select_dtypes(include=[np.number])
    if numeric.shape[1] >= ENGINEERED_FEATURE_COUNT:
        full = numeric.iloc[:, :ENGINEERED_FEATURE_COUNT].values.astype(np.float32)
        X = np.nan_to_num(full[:, MW_INDICES], nan=0.0)
        ts = [round(i, 2) for i in range(len(X))]
        if return_full:
            return X, ts, "features", np.nan_to_num(full, nan=0.0)
        return X, ts, "features"

    raise ValueError(
        "Unrecognized CSV format. Expected either the 75 Kinect 3D coordinate columns "
        f"(found {len(set(EXPECTED_COLS_3D) & cols)}/75) or >= {ENGINEERED_FEATURE_COUNT} "
        f"engineered feature columns (found {numeric.shape[1]})."
    )