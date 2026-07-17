"""
Build a MediaPipe typical-population reference (full 448-feature mean/std) from the
dataset's TD videos, run through the EXACT video inference pipeline. Used to express a
patient's metrics as deviations from a normative baseline IN THE SAME (MediaPipe) feature
space - so a typical MediaPipe clip doesn't look 'abnormal' against a Kinect baseline.

Run:  python -m train.build_mp_reference
"""
import os
import sys
import glob
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.pose_extraction import get_skeletal_data
from services.feature_engineering import prepare_inference_data

DATASET = r"C:/Users/shaba/Desktop/Dataset/Typical/*/video/video.avi"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediapipe_td_reference_448.npz")
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mpref_tmp.webm")


def main():
    vids = sorted(glob.glob(DATASET))
    print(f"Building MediaPipe TD reference from {len(vids)} typical videos...")
    all_full = []
    for i, v in enumerate(vids, 1):
        try:
            raw = get_skeletal_data(v, TMP)
            _, _, X_full = prepare_inference_data(raw, return_full=True)
            if len(X_full):
                all_full.append(np.asarray(X_full))
            print(f"[{i}/{len(vids)}] {os.path.basename(os.path.dirname(os.path.dirname(v)))}: {len(X_full)} windows")
        except Exception as e:
            print(f"[{i}/{len(vids)}] ERROR {e}")
        finally:
            if os.path.exists(TMP):
                os.remove(TMP)
    X = np.concatenate(all_full, axis=0)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    np.savez(OUT, mu=mu, sd=sd, n=len(X))
    print(f"\nSaved {OUT} from {len(X)} typical MediaPipe windows (dim {X.shape[1]}).")


if __name__ == "__main__":
    main()
