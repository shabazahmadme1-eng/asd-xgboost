import numpy as np
import pandas as pd
import warnings
from scipy.stats import kendalltau
from core.config import WINDOW_SIZE, STEP_SIZE, EXPECTED_COLS_3D

# Joint Index Mapping
J = {
    'Midspain': 0, 'AnkleLeft': 3, 'AnkleRight': 6, 'ElbowLeft': 9, 'ElbowRight': 12,
    'FootLeft': 15, 'FootRight': 18, 'HandLeft': 21, 'HandRight': 24, 'HandTipLeft': 27,
    'HandTipRight': 30, 'Head': 33, 'HipLeft': 36, 'HipRight': 39, 'KneeLeft': 42,
    'KneeRight': 45, 'Neck': 48, 'ShoulderLeft': 51, 'ShoulderRight': 54, 'SpineBase': 57,
    'SpineShoulder': 60, 'ThumbLeft': 63, 'ThumbRight': 66, 'WristLeft': 69, 'WristRight': 72
}

def get_vec(frame, joint_name): return frame[J[joint_name] : J[joint_name]+3]

def calc_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    v1_mag, v2_mag = np.linalg.norm(v1), np.linalg.norm(v2)
    if v1_mag == 0 or v2_mag == 0: return 0.0
    dot_prod = np.clip(np.dot(v1, v2) / (v1_mag * v2_mag), -1.0, 1.0)
    return np.degrees(np.arccos(dot_prod))

def calc_dist(p1, p2): return np.linalg.norm(p1 - p2)

def extract_spatiotemporal_features(raw_frames):
    """The 448-Feature Engine (Angles, Velocity, Jerk, Kendall's Tau)"""
    bio_frames = np.zeros((30, 12))
    for i in range(30):
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

    # Temporal Math
    velocity = np.diff(bio_frames, axis=0)
    acceleration = np.diff(velocity, axis=0)

    # Kendall's Tau Coordination Math
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

def prepare_inference_data(raw_df: pd.DataFrame):
    """Slices the video into overlapping 30-frame windows and extracts the 448 features."""
    
    # Ensure columns match our expected 3D layout
    if not all(col in raw_df.columns for col in EXPECTED_COLS_3D):
        print("Warning: Missing expected columns in raw data.")
        return None
        
    coords_df = raw_df[EXPECTED_COLS_3D].copy()
    frames = coords_df.values
    total_frames = len(frames)
    
    if total_frames < WINDOW_SIZE:
        print(f"Video too short: {total_frames} frames. Minimum is {WINDOW_SIZE}.")
        return None

    windowed_data = []
    
    # Slide across the video, extracting 448 features for every 1-second window
    for start_idx in range(0, total_frames - WINDOW_SIZE + 1, STEP_SIZE):
        window_frames = frames[start_idx : start_idx + WINDOW_SIZE]
        
        # 🚀 Calling the new 448-feature Spatiotemporal Engine
        features = extract_spatiotemporal_features(window_frames)
        
        # Calculate a timestamp for the UI
        timestamp_sec = round((start_idx + (WINDOW_SIZE / 2)) / 30.0, 2)
        
        windowed_data.append({
            "timestamp": timestamp_sec,
            "data": features
        })
        
    return windowed_data