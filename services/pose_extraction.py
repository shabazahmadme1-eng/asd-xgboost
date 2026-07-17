import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# --- EXACT CLINICAL COLUMNS ---
EXPECTED_COLS_3D = [
    'Midspain-x', 'Midspain-y', 'Midspain-z', 'AnkleLeft-x', 'AnkleLeft-y', 'AnkleLeft-z', 
    'AnkleRight-x', 'AnkleRight-y', 'AnkleRight-z', 'ElbowLeft-x', 'ElbowLeft-y', 'ElbowLeft-z', 
    'ElbowRight-x', 'ElbowRight-y', 'ElbowRight-z', 'FootLeft-x', 'FootLeft-y', 'FootLeft-z', 
    'FootRight-x', 'FootRight-y', 'FootRight-z', 'HandLeft-x', 'HandLeft-y', 'HandLeft-z', 
    'HandRight-x', 'HandRight-y', 'HandRight-z', 'HandTipLeft-x', 'HandTipLeft-y', 'HandTipLeft-z', 
    'HandTipRight-x', 'HandTipRight-y', 'HandTipRight-z', 'Head-x', 'Head-y', 'Head-z', 
    'HipLeft-x', 'HipLeft-y', 'HipLeft-z', 'HipRight-x', 'HipRight-y', 'HipRight-z', 
    'KneeLeft-x', 'KneeLeft-y', 'KneeLeft-z', 'KneeRight-x', 'KneeRight-y', 'KneeRight-z', 
    'Neck-x', 'Neck-y', 'Neck-z', 'ShoulderLeft-x', 'ShoulderLeft-y', 'ShoulderLeft-z', 
    'ShoulderRight-x', 'ShoulderRight-y', 'ShoulderRight-z', 'SpineBase-x', 'SpineBase-y', 
    'SpineBase-z', 'SpineShoulder-x', 'SpineShoulder-y', 'SpineShoulder-z', 'ThumbLeft-x', 
    'ThumbLeft-y', 'ThumbLeft-z', 'ThumbRight-x', 'ThumbRight-y', 'ThumbRight-z', 
    'WristLeft-x', 'WristLeft-y', 'WristLeft-z', 'WristRight-x', 'WristRight-y', 'WristRight-z'
]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# The model's velocity/acceleration and window-length features assume a fixed
# frame rate. Source videos vary (25/30/60 fps), so we resample every clip to
# this rate before feature engineering. Training and inference MUST use the same
# value, otherwise per-frame derivatives are scaled inconsistently.
TARGET_FPS = 30.0


def resample_to_fps(df: pd.DataFrame, src_fps: float, target_fps: float = TARGET_FPS) -> pd.DataFrame:
    """Linearly resample a per-frame coordinate dataframe to a constant fps."""
    if df is None or len(df) < 2:
        return df
    if not src_fps or src_fps <= 0 or np.isnan(src_fps) or abs(src_fps - target_fps) < 0.1:
        return df
    n_src = len(df)
    n_tgt = max(1, int(round(n_src / src_fps * target_fps)))
    src_idx = np.linspace(0.0, n_src - 1, n_src)
    tgt_idx = np.linspace(0.0, n_src - 1, n_tgt)
    return pd.DataFrame({c: np.interp(tgt_idx, src_idx, df[c].values) for c in df.columns})

def mid_of_points(p1, p2):
    """Helper to find midpoint of two computed 3D points"""
    return [(p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0, (p1[2]+p2[2])/2.0]

def get_skeletal_data(video_path: str, output_video_path: str) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    detected_frames = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    if width == 0 or height == 0: width, height = 640, 480
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            out.write(frame)

            # 🔧 FIX 1: Use NaN instead of 0.0 for missing frames
            row_data = {col: np.nan for col in EXPECTED_COLS_3D}

            if results.pose_world_landmarks:
                detected_frames += 1
                lm = results.pose_world_landmarks.landmark
                def get_xyz(idx): return [lm[idx].x * -1.0, lm[idx].y * -1.0, lm[idx].z]
                def get_mid(idx1, idx2): return [((lm[idx1].x + lm[idx2].x) / 2.0) * -1.0, ((lm[idx1].y + lm[idx2].y) / 2.0) * -1.0, (lm[idx1].z + lm[idx2].z) / 2.0]

                # Precompute spine points for accurate Midspain
                spine_shoulder = get_mid(11, 12)
                spine_base = get_mid(23, 24)

                row_data.update({
                    # 🔧 FIX 2: Midspain is now the midpoint of SpineShoulder and SpineBase (true mid-torso)
                    'Midspain-x': mid_of_points(spine_shoulder, spine_base)[0], 
                    'Midspain-y': mid_of_points(spine_shoulder, spine_base)[1], 
                    'Midspain-z': mid_of_points(spine_shoulder, spine_base)[2],
                    'AnkleLeft-x': get_xyz(27)[0], 'AnkleLeft-y': get_xyz(27)[1], 'AnkleLeft-z': get_xyz(27)[2],
                    'AnkleRight-x': get_xyz(28)[0], 'AnkleRight-y': get_xyz(28)[1], 'AnkleRight-z': get_xyz(28)[2],
                    'ElbowLeft-x': get_xyz(13)[0], 'ElbowLeft-y': get_xyz(13)[1], 'ElbowLeft-z': get_xyz(13)[2],
                    'ElbowRight-x': get_xyz(14)[0], 'ElbowRight-y': get_xyz(14)[1], 'ElbowRight-z': get_xyz(14)[2],
                    'FootLeft-x': get_xyz(31)[0], 'FootLeft-y': get_xyz(31)[1], 'FootLeft-z': get_xyz(31)[2],
                    'FootRight-x': get_xyz(32)[0], 'FootRight-y': get_xyz(32)[1], 'FootRight-z': get_xyz(32)[2],
                    'HandLeft-x': get_xyz(15)[0], 'HandLeft-y': get_xyz(15)[1], 'HandLeft-z': get_xyz(15)[2],
                    'HandRight-x': get_xyz(16)[0], 'HandRight-y': get_xyz(16)[1], 'HandRight-z': get_xyz(16)[2],
                    'HandTipLeft-x': get_xyz(19)[0], 'HandTipLeft-y': get_xyz(19)[1], 'HandTipLeft-z': get_xyz(19)[2],
                    'HandTipRight-x': get_xyz(20)[0], 'HandTipRight-y': get_xyz(20)[1], 'HandTipRight-z': get_xyz(20)[2],
                    'Head-x': get_xyz(0)[0], 'Head-y': get_xyz(0)[1], 'Head-z': get_xyz(0)[2],
                    'HipLeft-x': get_xyz(23)[0], 'HipLeft-y': get_xyz(23)[1], 'HipLeft-z': get_xyz(23)[2],
                    'HipRight-x': get_xyz(24)[0], 'HipRight-y': get_xyz(24)[1], 'HipRight-z': get_xyz(24)[2],
                    'KneeLeft-x': get_xyz(25)[0], 'KneeLeft-y': get_xyz(25)[1], 'KneeLeft-z': get_xyz(25)[2],
                    'KneeRight-x': get_xyz(26)[0], 'KneeRight-y': get_xyz(26)[1], 'KneeRight-z': get_xyz(26)[2],
                    'Neck-x': get_mid(11, 12)[0], 'Neck-y': get_mid(11, 12)[1], 'Neck-z': get_mid(11, 12)[2],
                    'ShoulderLeft-x': get_xyz(11)[0], 'ShoulderLeft-y': get_xyz(11)[1], 'ShoulderLeft-z': get_xyz(11)[2],
                    'ShoulderRight-x': get_xyz(12)[0], 'ShoulderRight-y': get_xyz(12)[1], 'ShoulderRight-z': get_xyz(12)[2],
                    'SpineBase-x': spine_base[0], 'SpineBase-y': spine_base[1], 'SpineBase-z': spine_base[2],
                    'SpineShoulder-x': spine_shoulder[0], 'SpineShoulder-y': spine_shoulder[1], 'SpineShoulder-z': spine_shoulder[2],
                    'ThumbLeft-x': get_xyz(21)[0], 'ThumbLeft-y': get_xyz(21)[1], 'ThumbLeft-z': get_xyz(21)[2],
                    'ThumbRight-x': get_xyz(22)[0], 'ThumbRight-y': get_xyz(22)[1], 'ThumbRight-z': get_xyz(22)[2],
                    'WristLeft-x': get_xyz(15)[0], 'WristLeft-y': get_xyz(15)[1], 'WristLeft-z': get_xyz(15)[2],
                    'WristRight-x': get_xyz(16)[0], 'WristRight-y': get_xyz(16)[1], 'WristRight-z': get_xyz(16)[2],
                })

            frames_data.append(row_data)

    cap.release()
    out.release()
    
    df = pd.DataFrame(frames_data)

    # 🔧 FIX 3: Interpolate missing frames instead of leaving NaN/0
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.ffill().bfill()
    df = df.fillna(0.0) # Nuclear fallback

    # Normalize to a constant frame rate so velocity/acceleration features are
    # comparable across videos and consistent with the training set.
    total_frames = len(df)
    df = resample_to_fps(df[EXPECTED_COLS_3D], fps)

    # Expose pose-detection quality for inference-time gating.
    df.attrs["detection_rate"] = (detected_frames / total_frames) if total_frames else 0.0
    df.attrs["source_frames"] = total_frames
    return df