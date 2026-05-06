import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from core.config import EXPECTED_COLS_3D

mp_pose = mp.solutions.pose
# NEW: Import drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_skeletal_data(video_path: str, output_video_path: str) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    frames_data = []

    # --- NEW: Setup Video Writer ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    if width == 0 or height == 0: width, height = 640, 480
    
    # VP8 codec (.webm) is highly compatible with React/HTML5 video players
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # --- NEW: Draw the skeleton on the frame ---
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Write the drawn frame to the new video file
            out.write(frame)

            # 1. Initialize row with 0.0s (Safety net for .avi missed frames)
            row_data = {col: 0.0 for col in EXPECTED_COLS_3D}

            # 2. Fill with real data if detected
            if results.pose_world_landmarks:
                lm = results.pose_world_landmarks.landmark
                def get_xyz(idx): return [lm[idx].x * -1.0, lm[idx].y * -1.0, lm[idx].z]
                def get_midpoint(idx1, idx2): return [((lm[idx1].x + lm[idx2].x) / 2.0) * -1.0, ((lm[idx1].y + lm[idx2].y) / 2.0) * -1.0, (lm[idx1].z + lm[idx2].z) / 2.0]

                row_data.update({
                    'Midspain-x': get_midpoint(23, 24)[0], 'Midspain-y': get_midpoint(23, 24)[1], 'Midspain-z': get_midpoint(23, 24)[2],
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
                    'Neck-x': get_midpoint(11, 12)[0], 'Neck-y': get_midpoint(11, 12)[1], 'Neck-z': get_midpoint(11, 12)[2],
                    'ShoulderLeft-x': get_xyz(11)[0], 'ShoulderLeft-y': get_xyz(11)[1], 'ShoulderLeft-z': get_xyz(11)[2],
                    'ShoulderRight-x': get_xyz(12)[0], 'ShoulderRight-y': get_xyz(12)[1], 'ShoulderRight-z': get_xyz(12)[2],
                    'SpineBase-x': get_midpoint(23, 24)[0], 'SpineBase-y': get_midpoint(23, 24)[1], 'SpineBase-z': get_midpoint(23, 24)[2],
                    'SpineShoulder-x': get_midpoint(11, 12)[0], 'SpineShoulder-y': get_midpoint(11, 12)[1], 'SpineShoulder-z': get_midpoint(11, 12)[2],
                    'ThumbLeft-x': get_xyz(21)[0], 'ThumbLeft-y': get_xyz(21)[1], 'ThumbLeft-z': get_xyz(21)[2],
                    'ThumbRight-x': get_xyz(22)[0], 'ThumbRight-y': get_xyz(22)[1], 'ThumbRight-z': get_xyz(22)[2],
                    'WristLeft-x': get_xyz(15)[0], 'WristLeft-y': get_xyz(15)[1], 'WristLeft-z': get_xyz(15)[2],
                    'WristRight-x': get_xyz(16)[0], 'WristRight-y': get_xyz(16)[1], 'WristRight-z': get_xyz(16)[2],
                })

            frames_data.append(row_data)

    cap.release()
    out.release() # NEW: Make sure to release the video writer!
    return pd.DataFrame(frames_data)