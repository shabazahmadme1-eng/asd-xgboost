import os
import sys
# Force UTF-8 console so emoji status prints don't crash on Windows cp1252 terminals
# (lets the server start with a plain `uvicorn app:app`, no PYTHONIOENCODING needed).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import uuid
import shutil
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.pose_extraction import get_skeletal_data
from services.feature_engineering import prepare_inference_data, prepare_csv_data, MW_INDICES

app = FastAPI(title="ASD Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wildcard origin + credentials is rejected by browsers; pick one
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public base URL used to build the processed-video link. Override in deployment.
BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000")

os.makedirs("processed_skeletons", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="processed_skeletons"), name="static")

# Single spatiotemporal model (label ASD=1, see original model1_xgboost.py). It was
# trained on Kinect 3D coordinates but generalizes to MediaPipe-from-video features for
# both the CSV (hardware) and video upload paths, given the sustained-spike decision
# rule below. Risk = predict_proba[:, 1], verified on folder-labeled dataset files.
KINECT_MODEL_PATH = "xgboost_asd_spatiotemporal_optimized.json"

def _load_model(path):
    try:
        m = xgb.XGBClassifier()
        m.load_model(path)
        print(f"✅ Loaded model: {path}")
        return m
    except Exception as e:
        print(f"⚠️ Warning: Could not load model {path}. Error: {e}")
        return None

print("🚀 Loading XGBoost model...")
model = _load_model(KINECT_MODEL_PATH)

# Decision logic mirrors the validated backend: a clip is flagged atypical when a
# sufficient number of 1-second windows individually exceed the per-window risk
# threshold (a sustained-spike rule), NOT when the mean risk is high. This is what
# separates a typical low-motion clip (no spikes) from an atypical one (many spikes),
# and avoids a single noisy window or a low mean hiding real signal.
WINDOW_RISK_THRESHOLD = 0.30  # a window counts as "atypical" above this probability
MIN_ATYPICAL_FRACTION = 0.10  # flag the clip if >=10% of windows (and at least MIN_ATYPICAL_WINDOWS) are atypical
MIN_ATYPICAL_WINDOWS = 2
MIN_DETECTION_RATE = 0.60     # Reject videos where MediaPipe found a pose in <60% of frames

REGION_NAMES = ["Upper Body", "Lower Body", "Symmetry"]

# Map each of the 448 raw spatiotemporal features to a body region so the
# clinical card reflects real biomechanics instead of dumping everything into
# one bucket. Feature layout (see services/feature_engineering.py):
#   [0:360]   flattened_bio  -> 30 frames x 12 metrics  (metric = idx % 12)
#   [360:444] 7 stat blocks  -> means/stds/mean_vel/max_vel/mean_acc/max_acc/tau_time, 12 metrics each
#   [444:448] tau_sym        -> 4 bilateral-symmetry features
_METRIC_REGION = {
    0: "Upper Body", 1: "Upper Body",    # elbow angle L/R
    2: "Lower Body", 3: "Lower Body",    # knee angle L/R
    4: "Upper Body", 5: "Upper Body",    # shoulder angle L/R
    6: "Lower Body", 7: "Lower Body",    # hip angle L/R
    8: "Upper Body",                     # wrist-to-wrist distance
    9: "Lower Body",                     # ankle-to-ankle distance
    10: "Upper Body", 11: "Upper Body",  # head-to-wrist distance L/R
}

def _region_for_raw_index(k: int) -> str:
    if k >= 444:
        return "Symmetry"                       # tau_sym block
    if k >= 360:
        return _METRIC_REGION[(k - 360) % 12]   # stat blocks
    return _METRIC_REGION[k % 12]               # flattened per-frame bio

# Region aligned to the model's 376 selected features (model column j <- MW_INDICES[j]).
FEATURE_REGIONS = [_region_for_raw_index(int(k)) for k in MW_INDICES] if MW_INDICES is not None else []

REGION_DESCRIPTIONS = {
    "Upper Body": "Arm/shoulder kinematics - elbow & shoulder angles, wrist excursion and inter-wrist distance. Sensitive to repetitive arm movements (stereotypies) and reduced/asymmetric arm swing.",
    "Lower Body": "Gait & lower-limb kinematics - hip & knee flexion, ankle base-of-support. Sensitive to atypical gait, toe-walking and posture.",
    "Symmetry": "Left-right coordination - Kendall's-tau correlation between paired limbs. Sensitive to breakdowns in bilateral motor symmetry.",
}

# Out-of-distribution reference (training feature mean/std) for a confidence flag.
OOD_STATS = None
try:
    _s = np.load(os.path.join("train", "kinect_feature_stats.npz"))
    OOD_STATS = (_s["mu"], _s["sd"], float(_s["baseline"]))
    print(f"✅ Loaded OOD reference (baseline z={OOD_STATS[2]:.2f})")
except Exception as e:
    print(f"⚠️ OOD reference not loaded ({e}); confidence flag disabled.")

DISCLAIMER = ("This is an automated screening aid, NOT a diagnosis. Results quantify motor "
              "kinematics only and must be interpreted by a qualified clinician alongside history, "
              "observation and validated diagnostic instruments.")

# Typical-population references (full 448-feature mean/std from the dataset's TD subjects),
# used to express a patient's metrics as deviations from a normative baseline. We keep one
# per sensor modality because raw feature VALUES differ between Kinect and MediaPipe, so a
# MediaPipe clip must be compared to a MediaPipe baseline (and vice versa) to avoid
# sensor-artifact "deviations".
def _load_ref(path):
    try:
        r = np.load(path)
        print(f"✅ Loaded reference {os.path.basename(path)} ({int(r['n'])} windows)")
        return (r["mu"], r["sd"])
    except Exception as e:
        print(f"⚠️ Reference {os.path.basename(path)} not loaded ({e}).")
        return None

TD_REF_KINECT = _load_ref(os.path.join("train", "td_reference_448.npz"))
TD_REF_MP = _load_ref(os.path.join("train", "mediapipe_td_reference_448.npz"))

def pick_reference(source_kind):
    """MediaPipe-derived inputs (video, keypoints CSV, engineered-feature CSV) compare
    against the MediaPipe baseline; genuine Kinect coordinate exports use the Kinect one."""
    mp_like = source_kind in ("video", "csv-coordinates", "csv-features")
    return (TD_REF_MP or TD_REF_KINECT) if mp_like else (TD_REF_KINECT or TD_REF_MP)

# The 12 biomechanical metrics (order matches services/feature_engineering.py) and the
# 448-vector block layout, so SHAP drivers and deviations can be named for clinicians.
BIO_METRICS = [
    "Left Arm Angle", "Right Arm Angle", "Left Leg Angle", "Right Leg Angle",
    "Left Shoulder Posture", "Right Shoulder Posture", "Left Hip Posture", "Right Hip Posture",
    "Wrist-to-Wrist Distance", "Ankle-to-Ankle Distance", "Head-to-Left-Wrist", "Head-to-Right-Wrist",
]
SYM_METRICS = ["Arms", "Legs", "Shoulders", "Hips"]
_ASPECTS = ["average", "variability", "velocity", "peak velocity",
            "acceleration (jerk)", "peak acceleration", "temporal trend"]


def _metric_region(metric: str) -> str:
    m = metric.lower()
    if any(w in m for w in ["arm", "shoulder", "wrist", "head"]):
        return "Upper Body"
    if any(w in m for w in ["leg", "hip", "ankle"]):
        return "Lower Body"
    return "Symmetry"


def _feature_label(k: int):
    """Map a raw 448-feature index to (metric_name, aspect)."""
    if k >= 444:
        return SYM_METRICS[k - 444] + " symmetry", "bilateral symmetry"
    if k >= 360:
        return BIO_METRICS[(k - 360) % 12], _ASPECTS[(k - 360) // 12]
    return BIO_METRICS[k % 12], "instantaneous"


# Human-readable (metric, aspect) label for each of the model's 376 selected features.
FEATURE_LABELS = [_feature_label(int(k)) for k in MW_INDICES] if MW_INDICES is not None else []


def kinematic_markers(X, top=6):
    """Top specific kinematic drivers of the ASD score for THIS recording, via SHAP,
    aggregated to the (metric, aspect) level and named for clinicians."""
    if not FEATURE_LABELS:
        return []
    try:
        contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    except Exception:
        return []
    pos = np.clip(contribs, 0, None).sum(axis=0)
    agg = {}
    for j, (metric, aspect) in enumerate(FEATURE_LABELS):
        agg[(metric, aspect)] = agg.get((metric, aspect), 0.0) + float(pos[j])
    total = sum(agg.values()) or 1.0
    ordered = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return [{
        "marker": f"{metric} - {aspect}",
        "metric": metric, "aspect": aspect, "region": _metric_region(metric),
        "contribution_pct": round(v / total * 100, 1),
    } for (metric, aspect), v in ordered if v > 0]


def kinematic_findings(X_full, ref):
    """Per-metric deviations from the typical population (z-score on posture / variability /
    movement-speed). Returns the most notable abnormal findings."""
    if ref is None or X_full is None or len(X_full) == 0:
        return []
    mu, sd = ref
    pmean = np.asarray(X_full).mean(axis=0)
    blocks = {"posture": 360, "variability": 372, "movement speed": 384}
    found = []
    for label, base in blocks.items():
        for mi in range(12):
            k = base + mi
            z = float((pmean[k] - mu[k]) / sd[k])
            if abs(z) >= 1.5:
                found.append({
                    "marker": BIO_METRICS[mi], "aspect": label,
                    "region": _metric_region(BIO_METRICS[mi]),
                    "patient_value": round(float(pmean[k]), 2),
                    "typical_value": round(float(mu[k]), 2),
                    "z_score": round(z, 1),
                    "direction": "above typical" if z > 0 else "below typical",
                })
    found.sort(key=lambda d: abs(d["z_score"]), reverse=True)
    return found[:8]


def symmetry_indices(X_full, ref):
    """Left-right coordination per limb pair (Kendall's-tau), vs the typical baseline."""
    if ref is None or X_full is None or len(X_full) == 0:
        return []
    mu, sd = ref
    pmean = np.asarray(X_full).mean(axis=0)
    out = []
    for i, pair in enumerate(SYM_METRICS):
        k = 444 + i
        z = float((pmean[k] - mu[k]) / sd[k])
        out.append({
            "pair": pair,
            "coordination": round(float(pmean[k]), 2),
            "typical": round(float(mu[k]), 2),
            "status": "reduced" if z < -1.5 else "typical",
        })
    return out


def ood_score(X):
    """Mean per-feature z-distance of the input vs the model's training distribution."""
    if OOD_STATS is None:
        return None
    mu, sd, _ = OOD_STATS
    return float(np.abs((X - mu) / sd).mean())


def regional_drivers(X):
    """Patient-specific contribution of each body region toward the ASD score, via SHAP.
    Unlike global feature importance, this reflects what drove THIS recording."""
    out = {k: 0.0 for k in REGION_NAMES}
    if not FEATURE_REGIONS:
        return out
    try:
        contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    except Exception as e:
        print(f"⚠️ SHAP unavailable, falling back to global importance: {e}")
        imps = getattr(model, "feature_importances_", np.ones(len(FEATURE_REGIONS)))
        for j, r in enumerate(FEATURE_REGIONS):
            out[r] += float(imps[j])
        total = sum(out.values()) or 1.0
        return {k: v / total * 100 for k, v in out.items()}
    pos = np.clip(contribs, 0, None).sum(axis=0)  # contributions pushing toward ASD
    for j, r in enumerate(FEATURE_REGIONS):
        out[r] += float(pos[j])
    total = sum(out.values())
    if total > 0:
        out = {k: v / total * 100 for k, v in out.items()}
    return out


def build_clinical_summary(is_atypical, predictions, atypical_count, min_windows,
                           regions_pct, window_threshold):
    n = len(predictions)
    peak = float(np.max(predictions)) if n else 0.0
    mean = float(np.mean(predictions)) if n else 0.0
    frac = atypical_count / max(n, 1)
    primary = max(regions_pct, key=regions_pct.get) if regions_pct else "Upper Body"
    if not is_atypical:
        severity = "Within typical range"
        narrative = (f"No sustained atypical kinematics detected: {atypical_count} of {n} "
                     f"one-second windows crossed the {int(window_threshold*100)}% per-window "
                     f"risk threshold (flag requires >= {min_windows}). The motor profile "
                     f"aligns with neurotypical development.")
    else:
        severity = ("Pronounced" if peak >= 0.80 else "Moderate" if peak >= 0.55
                    else "Mild / low-confidence flag")
        narrative = (f"{atypical_count} of {n} one-second windows ({frac*100:.0f}%) showed "
                     f"atypical kinematics - most strongly in {primary.lower()}. "
                     f"Peak window risk {peak*100:.0f}%, average {mean*100:.0f}%. "
                     f"Clinical correlation recommended.")
    return {
        "classification": "Atypical Kinematics Flagged" if is_atypical else "Typical Motor Development",
        "severity": severity,
        "peak_risk": round(peak, 4),
        "mean_risk": round(mean, 4),
        "atypical_windows": int(atypical_count),
        "total_windows": int(n),
        "atypical_fraction_pct": round(frac * 100, 1),
        "flag_threshold_windows": int(min_windows),
        "window_risk_threshold_pct": int(window_threshold * 100),
        "primary_driver": primary if is_atypical else None,
        "decision_rule": (f"Flagged: >= {min_windows} of {n} windows exceeded "
                          f"{int(window_threshold*100)}% risk." if is_atypical else
                          f"Not flagged: only {atypical_count} of {n} windows exceeded "
                          f"{int(window_threshold*100)}% (need >= {min_windows})."),
        "narrative": narrative,
    }


@app.post("/api/analyze")
async def analyze_patient(
    file: UploadFile = File(...),
    file_type: str = Form("video") 
):
    if model is None:
        raise HTTPException(status_code=500, detail="XGBoost model is not loaded.")

    # Sanitize the client-supplied filename to prevent path traversal and collisions.
    safe_name = os.path.basename(file.filename or "upload")
    temp_input = os.path.join("temp_uploads", f"temp_{uuid.uuid4().hex}_{safe_name}")
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        raw_df = None
        video_url = None
        raw_timestamps = []
        source_kind = "unknown"
        detection_rate = None

        if file_type == "csv" or safe_name.endswith(".csv"):
            print(f"🚀 Processing CSV Hardware Path...")
            raw_df = pd.read_csv(temp_input)

            time_col = next((col for col in raw_df.columns if 'H:M:S:MS' in col), None)
            if time_col:
                raw_timestamps = raw_df[time_col].tolist()

            # prepare_csv_data auto-detects raw-coordinate vs pre-engineered-feature
            # CSVs, applies training-faithful cleaning, and rejects unknown formats.
            try:
                X_windows, window_timestamps, csv_kind, X_full = prepare_csv_data(raw_df, return_full=True)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            source_kind = f"csv-{csv_kind}"
            print(f"   Detected CSV format: {csv_kind} ({len(X_windows)} windows)")
        else:
            print(f"🎬 Processing Video via MediaPipe...")
            stem = os.path.splitext(safe_name)[0] or "video"
            output_filename = f"skeletal_{uuid.uuid4().hex}_{stem}.webm"
            output_path = os.path.join("processed_skeletons", output_filename)

            raw_df = get_skeletal_data(temp_input, output_path)
            video_url = f"{BASE_URL}/static/{output_filename}"
            source_kind = "video"

            # Quality gate: refuse to score videos where MediaPipe rarely found a
            # full-body pose. Such clips (occlusion, wrong framing, no person) would
            # otherwise yield an unreliable score presented with false confidence.
            detection_rate = float(raw_df.attrs.get("detection_rate", 1.0))
            if detection_rate < MIN_DETECTION_RATE:
                raise HTTPException(
                    status_code=422,
                    detail=(f"Low pose-detection quality ({detection_rate:.0%} of frames). "
                            f"Use a clearer, full-body, single-person video."),
                )
            if raw_df is None or raw_df.empty:
                raise HTTPException(status_code=400, detail="No skeletal data extracted.")
            X_windows, window_timestamps, X_full = prepare_inference_data(raw_df, return_full=True)

        if len(X_windows) == 0:
            raise HTTPException(status_code=400, detail="Failed to engineer spatiotemporal features.")

        X_matrix = np.asarray(X_windows, dtype=np.float32)
        if X_matrix.ndim == 1:
            X_matrix = X_matrix.reshape(1, -1)
        elif X_matrix.ndim == 3:
            X_matrix = X_matrix.reshape(X_matrix.shape[0], -1)

        # 🔧 FIX: SANITY CHECK - Crash loudly instead of silently producing garbage
        assert X_matrix.shape[1] == model.n_features_in_, \
            f"FATAL: Model expects {model.n_features_in_} features, got {X_matrix.shape[1]}. " \
            f"Your MW_INDICES is misaligned with the trained model!"

        # Class 1 is ASD (verified on folder-labeled dataset files and the original
        # training code). Column 1 is the per-window atypical-risk probability.
        predictions = model.predict_proba(X_matrix)[:, 1]

        # Sustained-spike aggregation (matches the validated backend): flag the clip
        # atypical when enough individual windows exceed the per-window risk threshold,
        # rather than thresholding the mean. A typical low-motion clip produces no
        # spikes; an atypical one produces several.
        n_windows = len(predictions)
        peak_risk = float(np.max(predictions))
        mean_risk = float(np.mean(predictions))
        atypical_count = int(np.sum(predictions >= WINDOW_RISK_THRESHOLD))
        min_windows = max(MIN_ATYPICAL_WINDOWS, int(n_windows * MIN_ATYPICAL_FRACTION))

        if n_windows < MIN_ATYPICAL_WINDOWS:
            # Too few windows for the spike rule (e.g. a single short clip);
            # fall back to a direct probability decision on what we have.
            is_atypical = bool(mean_risk >= 0.50)
        else:
            is_atypical = bool(atypical_count >= min_windows)

        # Headline score = peak window risk, so the UI gauge matches the verdict
        # (the mean stays low even for a flagged clip and would mislead clinicians).
        final_risk_score = peak_risk

        # BUILD THE REACT TIMELINE
        if raw_timestamps and len(raw_timestamps) > 0:
            step = max(1, len(raw_timestamps) // n_windows)
            window_times = raw_timestamps[::step][:n_windows]
            timeline = [{"timestamp": str(ts), "risk_score": float(pred)} for ts, pred in zip(window_times, predictions)]
        else:
            timeline = [{"timestamp": f"{ts}s", "risk_score": float(pred)} for ts, pred in zip(window_timestamps, predictions)]

        # PATIENT-SPECIFIC regional drivers (SHAP) + clinical narrative
        regions_pct = regional_drivers(X_matrix)
        regions_list = [{"name": k, "value": round(v, 1)} for k, v in regions_pct.items()]
        clinical_summary = build_clinical_summary(
            is_atypical, predictions, atypical_count, min_windows, regions_pct, WINDOW_RISK_THRESHOLD)

        # Per-region cards with patient-specific share + description + flag status
        flagged_regions = {r for r in REGION_NAMES if regions_pct.get(r, 0) >= 100.0 / len(REGION_NAMES)}
        regional_breakdown = [{
            "name": r,
            "contribution_pct": round(regions_pct.get(r, 0.0), 1),
            "status": "Elevated" if (is_atypical and r in flagged_regions) else "Typical",
            "description": REGION_DESCRIPTIONS[r],
        } for r in REGION_NAMES]

        # Quality / confidence flags
        ood = ood_score(X_matrix)
        warnings_list = []
        if ood is not None and OOD_STATS and ood > OOD_STATS[2] * 1.35:
            warnings_list.append("Input differs notably from the model's reference data "
                                 "(possible non-standard recording or capture convention); interpret with caution.")
        if n_windows < 5:
            warnings_list.append(f"Short recording ({n_windows} analysis window(s)); the result is less stable. "
                                 "A 10s+ clip of the child moving is recommended.")
        if source_kind == "video" and detection_rate is not None and detection_rate < 0.90:
            warnings_list.append(f"Full-body pose detected in only {detection_rate*100:.0f}% of frames; "
                                 "ensure the whole body stays in frame.")

        input_meta = {
            "kind": source_kind,
            "windows_analyzed": n_windows,
            "window_seconds": 1.0,
            "detection_rate_pct": round(detection_rate * 100, 1) if detection_rate is not None else None,
            "ood_score": round(ood, 2) if ood is not None else None,
            "ood_baseline": round(OOD_STATS[2], 2) if OOD_STATS else None,
        }

        # --- Deeper clinical detail ---
        # Specific kinematic drivers (named metric + aspect), per-metric deviations vs the
        # typical population, bilateral symmetry, the exact atypical timestamps, and stability.
        ref = pick_reference(source_kind)
        markers = kinematic_markers(X_matrix)
        findings = kinematic_findings(X_full, ref)
        symmetry = symmetry_indices(X_full, ref)
        flagged_moments = [
            {"timestamp": window_timestamps[i] if i < len(window_timestamps) else i,
             "risk": round(float(predictions[i]), 3)}
            for i in range(n_windows) if predictions[i] >= WINDOW_RISK_THRESHOLD
        ][:25]
        result_stability = {
            "window_risk_std": round(float(np.std(predictions)), 3),
            "consistency": ("high" if float(np.std(predictions)) < 0.12
                            else "moderate" if float(np.std(predictions)) < 0.22 else "low"),
        }

        return {
            "status": "success",
            "processed_video_url": video_url,
            # --- compatibility fields (existing frontend) ---
            "final_risk_score": final_risk_score,
            "is_atypical": is_atypical,
            "regions": regions_list,
            "timeline": timeline,
            # --- enriched clinical detail ---
            "clinical_summary": clinical_summary,
            "regional_breakdown": regional_breakdown,
            "kinematic_markers": markers,
            "kinematic_findings": findings,
            "symmetry": symmetry,
            "flagged_moments": flagged_moments,
            "result_stability": result_stability,
            "input_meta": input_meta,
            "quality": {"reliable": len(warnings_list) == 0, "warnings": warnings_list},
            "disclaimer": DISCLAIMER,
        }

    except Exception as e:
        print(f"❌ Processing Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)