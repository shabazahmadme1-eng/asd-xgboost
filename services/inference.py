import numpy as np
import pandas as pd
import os
from core.config import STATIC_DIR, MANN_WHITNEY_PATH

# Load the clinical filter globally
try:
    MANN_WHITNEY_INDICES = np.load(MANN_WHITNEY_PATH)
except FileNotFoundError:
    print("⚠️ WARNING: mann_whitney_indices.npy not found! Put it in the main backend folder.")
    MANN_WHITNEY_INDICES = np.arange(448) 

BIO_METRICS = [
    "Left Arm Angle", "Right Arm Angle", "Left Leg Angle", "Right Leg Angle",
    "Left Shoulder Posture", "Right Shoulder Posture", "Left Hip Posture", "Right Hip Posture",
    "Wrist-to-Wrist Distance", "Ankle-to-Ankle Distance", "Head-to-Left-Wrist", "Head-to-Right-Wrist"
]

REGIONS = {
    "upper_body": [0, 1, 4, 5],
    "lower_body": [2, 3, 6, 7],
    "symmetry": [8, 9, 10, 11]
}

def get_feature_description(idx: int) -> str:
    if idx < 360: return f"Instantaneous {BIO_METRICS[idx % 12]} at Frame {idx // 12}"
    idx -= 360
    if idx < 12: return f"Average {BIO_METRICS[idx]}"
    idx -= 12
    if idx < 12: return f"High Variance in {BIO_METRICS[idx]}"
    idx -= 12
    if idx < 12: return f"High Velocity in {BIO_METRICS[idx]}"
    idx -= 12
    if idx < 12: return f"Peak Velocity in {BIO_METRICS[idx]}"
    idx -= 12
    if idx < 12: return f"High Acceleration (Jerk) in {BIO_METRICS[idx]}"
    idx -= 12
    if idx < 12: return f"Peak Acceleration (Jerk) in {BIO_METRICS[idx]}"
    idx -= 12
    if idx < 12: return f"Atypical Time-Trend in {BIO_METRICS[idx]}"
    idx -= 12
    sym_labels = ["Arm Symmetry", "Leg Symmetry", "Shoulder Symmetry", "Hip Symmetry"]
    return f"Atypical Bilateral {sym_labels[idx]}"

def generate_excel_report(windowed_data, base_filename):
    """Generates the 448-column downloadable Excel file for the React UI."""
    column_names = []
    
    for frame_num in range(30):
        for metric in BIO_METRICS:
            column_names.append(f"Frame_{frame_num}_{metric.replace(' ', '_')}")
            
    for stat in ["MEAN", "STD_DEV", "MEAN_VEL", "MAX_VEL", "MEAN_ACCEL", "MAX_ACCEL", "TAU_TREND"]:
        for metric in BIO_METRICS:
            column_names.append(f"{stat}_{metric.replace(' ', '_')}")
            
    column_names.extend(["TAU_SYM_Arms", "TAU_SYM_Legs", "TAU_SYM_Shoulders", "TAU_SYM_Hips"])

    all_rows = []
    for window in windowed_data:
        all_rows.append(window["data"].flatten())

    df = pd.DataFrame(all_rows, columns=column_names)
    df.insert(0, "Window_Timestamp", [w["timestamp"] for w in windowed_data])

    # 🛠️ FIXED: Save to STATIC_DIR so React can download it
    excel_filename = f"{base_filename}_raw_features.xlsx"
    excel_path = os.path.join(STATIC_DIR, excel_filename)
    df.to_excel(excel_path, index=False, engine='openpyxl')
    
    return excel_filename

def run_diagnosis(windowed_data, model, video_filename):
    timeline = []
    importances = model.feature_importances_
    
    # 🎯 The mathematically optimized sweet spot
    RISK_THRESHOLD = 0.30
    MIN_TOTAL_WINDOWS = max(2, int(len(windowed_data) * 0.10)) 
    
    peak_risk = 0.0
    total_high_risk_windows = 0
    regional_accumulators = {"upper_body": 0, "lower_body": 0, "symmetry": 0}
    regional_top_markers = {"upper_body": set(), "lower_body": set(), "symmetry": set()}

    for window in windowed_data:
        X_input_full = window["data"].flatten()
        X_filtered = X_input_full[MANN_WHITNEY_INDICES].reshape(1, -1)
        
        overall_risk = float(model.predict_proba(X_filtered)[0][1]) 
        
        live_frames = X_input_full[:360].reshape(30, 12)
        window_means = np.mean(live_frames, axis=0)
        deviations = np.abs(live_frames - window_means)
        max_dev_idx = np.argmax(deviations)
        peak_frame = max_dev_idx // 12
        peak_metric = max_dev_idx % 12
        frame_trigger = f"Instantaneous {BIO_METRICS[peak_metric]} at Frame {peak_frame}"

        regional_impacts = {"upper_body": 0.0, "lower_body": 0.0, "symmetry": 0.0}
        total_impact = np.sum(importances) + 1e-9 
        
        for i, original_idx in enumerate(MANN_WHITNEY_INDICES):
            if original_idx < 360:
                base_idx = original_idx % 12
            elif original_idx < 444:
                base_idx = (original_idx - 360) % 12
            else:
                base_idx = [0, 2, 4, 6][original_idx - 444]

            for region_name, region_idxs in REGIONS.items():
                if base_idx in region_idxs:
                    regional_impacts[region_name] += importances[i]
                    if importances[i] > (total_impact / 50): 
                        regional_top_markers[region_name].add(get_feature_description(original_idx))

        regional_spikes = {
            r: min(round(overall_risk * (val / total_impact) * 2.5, 3), 1.0) 
            for r, val in regional_impacts.items()
        }

        timeline.append({
            "timestamp": window["timestamp"], 
            "overall_risk_score": round(overall_risk, 3), 
            "regional_spikes": regional_spikes,
            "frame_trigger": frame_trigger
        })

        if overall_risk > peak_risk: peak_risk = overall_risk
        if overall_risk >= RISK_THRESHOLD:
            total_high_risk_windows += 1
            for r, score in regional_spikes.items():
                if score >= RISK_THRESHOLD: regional_accumulators[r] += 1

    is_asd = total_high_risk_windows >= MIN_TOTAL_WINDOWS
    
    if not is_asd:
        diagnosis = "Neurotypical"
        severity = "Baseline"
        narrative = "Kinematic patterns align with neurotypical baseline. No sustained atypical movements detected."
    else:
        diagnosis = "Autism Spectrum Disorder"
        primary_driver = max(regional_accumulators, key=regional_accumulators.get).replace("_", " ") if sum(regional_accumulators.values()) > 0 else "kinematic variance"
        
        if peak_risk < 0.65:
            severity = "Mild"
            narrative = f"Mild atypical kinematics detected ({total_high_risk_windows} overlapping windows flagged), primarily driven by {primary_driver}."
        elif peak_risk < 0.85:
            severity = "Moderate"
            narrative = f"Moderate atypical kinematics detected ({total_high_risk_windows} overlapping windows flagged), primarily driven by {primary_driver}."
        else:
            severity = "Severe"
            narrative = f"Highly pronounced atypical kinematics detected ({total_high_risk_windows} overlapping windows flagged), primarily driven by {primary_driver}."

    def format_region(region_name):
        is_elevated = regional_accumulators[region_name] >= 2
        markers = list(regional_top_markers[region_name])[:2] if is_elevated else ["None detected"]
        return {
            "risk_status": "Elevated" if is_elevated else "Baseline",
            "markers": markers if markers else ["None detected"]
        }

    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    excel_filename = generate_excel_report(windowed_data, base_name)

    return {
        "status": "success",
        "clinical_summary": {
            "diagnosis": "Autism Spectrum Disorder" if is_asd else "Neurotypical",
            "severity_level": severity,
            "overall_confidence": f"{peak_risk * 100:.2f}%",
            "total_atypical_windows": total_high_risk_windows,
            "narrative": narrative
        },
        "regional_breakdown": {
            "upper_body_kinematics": format_region("upper_body"),
            "lower_body_kinematics": format_region("lower_body"),
            "spatial_symmetry": format_region("symmetry")
        },
        "analysis_timeline": timeline,
        "excel_url": f"http://127.0.0.1:8000/static/{excel_filename}"
    }