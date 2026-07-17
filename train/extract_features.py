"""
Build a MediaPipe-based training set from the Kinect autism gait dataset.

Each raw `video.avi` is run through the EXACT server inference pipeline
(get_skeletal_data -> prepare_inference_data) so that the features used for
training match the features produced at serving time. Every sliding window
becomes one labelled sample; the subject id is recorded so train/val splits can
be made by subject (no window-level leakage).

Label convention: ASD = 1, Typical = 0  (positive class = ASD).

Resumable: per-video feature arrays are cached as .npy under train/cache/.
Run:  python -m train.extract_features
"""
import os
import sys
import glob
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.pose_extraction import get_skeletal_data
from services.feature_engineering import prepare_inference_data

DATASET_ROOT = r"C:/Users/shaba/Desktop/Dataset"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
TMP_VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_extract_tmp.webm")
os.makedirs(CACHE_DIR, exist_ok=True)

# (glob pattern, label, subject-prefix) for each class folder.
SOURCES = [
    (os.path.join(DATASET_ROOT, "Autism", "children with ASD", "*", "video", "video.avi"), 1, "asd"),
    (os.path.join(DATASET_ROOT, "Autism", "Severe level of ASD", "*", "video", "video.avi"), 1, "severe"),
    (os.path.join(DATASET_ROOT, "Typical", "*", "video", "video.avi"), 0, "td"),
]


def subject_id(path: str, prefix: str) -> str:
    # .../<subject>/video/video.avi  -> subject is two levels up
    subject = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return f"{prefix}_{subject}"


def main():
    jobs = []
    for pattern, label, prefix in SOURCES:
        for path in sorted(glob.glob(pattern)):
            jobs.append((path, label, subject_id(path, prefix)))
    print(f"Found {len(jobs)} raw videos to process.")

    manifest = []  # (subject, label, n_windows, status)
    for i, (path, label, subj) in enumerate(jobs, 1):
        cache_x = os.path.join(CACHE_DIR, f"{subj}.npy")
        if os.path.exists(cache_x):
            X = np.load(cache_x)
            manifest.append((subj, label, len(X), "cached"))
            print(f"[{i}/{len(jobs)}] {subj}: cached ({len(X)} windows)")
            continue
        try:
            raw = get_skeletal_data(path, TMP_VIDEO)
            X, _ = prepare_inference_data(raw)
            X = np.asarray(X, dtype=np.float32)
            if X.ndim != 2 or len(X) == 0:
                manifest.append((subj, label, 0, "no_windows"))
                print(f"[{i}/{len(jobs)}] {subj}: 0 windows (skipped)")
                continue
            np.save(cache_x, X)
            manifest.append((subj, label, len(X), "ok"))
            print(f"[{i}/{len(jobs)}] {subj}: {len(X)} windows, dim={X.shape[1]}")
        except Exception as e:
            manifest.append((subj, label, 0, f"error:{e}"))
            print(f"[{i}/{len(jobs)}] {subj}: ERROR {e}")
        finally:
            if os.path.exists(TMP_VIDEO):
                os.remove(TMP_VIDEO)

    # Assemble the full matrix from cache.
    Xs, ys, groups = [], [], []
    for subj, label, n, status in manifest:
        cache_x = os.path.join(CACHE_DIR, f"{subj}.npy")
        if not os.path.exists(cache_x):
            continue
        X = np.load(cache_x)
        if len(X) == 0:
            continue
        Xs.append(X)
        ys.append(np.full(len(X), label, dtype=np.int64))
        groups.append(np.array([subj] * len(X)))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    g = np.concatenate(groups, axis=0)
    np.save(os.path.join(CACHE_DIR, "_X.npy"), X)
    np.save(os.path.join(CACHE_DIR, "_y.npy"), y)
    np.save(os.path.join(CACHE_DIR, "_groups.npy"), g)

    n_subj = len(set(g))
    print("\n==== EXTRACTION SUMMARY ====")
    print(f"Videos with usable windows: {sum(1 for _,_,n,_ in manifest if n>0)} / {len(jobs)}")
    print(f"Total window samples: {len(X)}  (feature dim {X.shape[1]})")
    print(f"ASD windows: {int((y==1).sum())}  Typical windows: {int((y==0).sum())}")
    print(f"Distinct subjects: {n_subj}")
    print(f"Saved matrices to {CACHE_DIR}/_X.npy, _y.npy, _groups.npy")


if __name__ == "__main__":
    main()
