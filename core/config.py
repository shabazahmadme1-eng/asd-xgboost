import os

# Dynamically route to the root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 🚀 THE OPTIMIZED SPATIOTEMPORAL PIPELINE ---
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_asd_spatiotemporal_optimized.json")
MANN_WHITNEY_PATH = os.path.join(BASE_DIR, "mann_whitney_indices.npy")

# Directories
TEMP_DIR = os.path.join(BASE_DIR, "temp_uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure they exist immediately
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Extraction Settings
WINDOW_SIZE = 30  
STEP_SIZE = 10    

# Joint Index Mapping
J = {
    'Midspain': 0, 'AnkleLeft': 3, 'AnkleRight': 6, 'ElbowLeft': 9, 'ElbowRight': 12,
    'FootLeft': 15, 'FootRight': 18, 'HandLeft': 21, 'HandRight': 24, 'HandTipLeft': 27,
    'HandTipRight': 30, 'Head': 33, 'HipLeft': 36, 'HipRight': 39, 'KneeLeft': 42,
    'KneeRight': 45, 'Neck': 48, 'ShoulderLeft': 51, 'ShoulderRight': 54, 'SpineBase': 57,
    'SpineShoulder': 60, 'ThumbLeft': 63, 'ThumbRight': 66, 'WristLeft': 69, 'WristRight': 72
}

# --- THE MISSING PIECE ---
# The 75 strict 3D columns from your raw Excel/CSV tracking files
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
    'ShoulderRight-x', 'ShoulderRight-y', 'ShoulderRight-z', 'SpineBase-x', 'SpineBase-y', 'SpineBase-z', 
    'SpineShoulder-x', 'SpineShoulder-y', 'SpineShoulder-z', 'ThumbLeft-x', 'ThumbLeft-y', 'ThumbLeft-z', 
    'ThumbRight-x', 'ThumbRight-y', 'ThumbRight-z', 'WristLeft-x', 'WristLeft-y', 'WristLeft-z', 
    'WristRight-x', 'WristRight-y', 'WristRight-z'
]