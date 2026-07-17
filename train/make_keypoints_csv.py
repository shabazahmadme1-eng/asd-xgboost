"""
Generate a *_keypoints.csv from a video using the SAME MediaPipe extraction the
video inference path uses. Output matches the keypoints CSV format:
    H:M:S:MS) , <75 EXPECTED_COLS_3D coordinate columns>

This lets a video and its generated CSV produce identical predictions, and gives
labelled keypoints CSVs for calibrating the CSV path.

Usage:
  python -m train.make_keypoints_csv C:/path/video.mp4 [out.csv]
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import cv2

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mediapipe as mp
from services.pose_extraction import EXPECTED_COLS_3D, mid_of_points

mp_pose = mp.solutions.pose


def ts_string(frame_idx, fps):
    t = frame_idx / (fps if fps else 30.0)
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h}:{m}:{s}:{ms})"


def generate(video_path, out_csv):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    idx = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            row = {c: 0.0 for c in EXPECTED_COLS_3D}
            row["H:M:S:MS)"] = ts_string(idx, fps)
            if res.pose_world_landmarks:
                lm = res.pose_world_landmarks.landmark
                def g(i): return [lm[i].x * -1.0, lm[i].y * -1.0, lm[i].z]
                def gm(a, b): return [((lm[a].x + lm[b].x) / 2) * -1.0, ((lm[a].y + lm[b].y) / 2) * -1.0, (lm[a].z + lm[b].z) / 2]
                ss, sb = gm(11, 12), gm(23, 24)
                mid = mid_of_points(ss, sb)
                vals = {
                    'Midspain': mid, 'AnkleLeft': g(27), 'AnkleRight': g(28), 'ElbowLeft': g(13),
                    'ElbowRight': g(14), 'FootLeft': g(31), 'FootRight': g(32), 'HandLeft': g(15),
                    'HandRight': g(16), 'HandTipLeft': g(19), 'HandTipRight': g(20), 'Head': g(0),
                    'HipLeft': g(23), 'HipRight': g(24), 'KneeLeft': g(25), 'KneeRight': g(26),
                    'Neck': gm(11, 12), 'ShoulderLeft': g(11), 'ShoulderRight': g(12),
                    'SpineBase': sb, 'SpineShoulder': ss, 'ThumbLeft': g(21), 'ThumbRight': g(22),
                    'WristLeft': g(15), 'WristRight': g(16),
                }
                for joint, xyz in vals.items():
                    row[f"{joint}-x"], row[f"{joint}-y"], row[f"{joint}-z"] = xyz
            rows.append(row)
            idx += 1
    cap.release()
    df = pd.DataFrame(rows)[["H:M:S:MS)"] + EXPECTED_COLS_3D]
    df.to_csv(out_csv, index=False)
    print(f"{os.path.basename(video_path)} -> {out_csv}  ({len(df)} frames @ {fps:.1f} fps)")
    return out_csv


if __name__ == "__main__":
    vid = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(vid))[0] + "_keypoints.csv"
    generate(vid, out)
