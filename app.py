import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import xgboost as xgb

# Import STATIC_DIR directly from config
from core.config import MODEL_PATH, TEMP_DIR, BASE_DIR, STATIC_DIR
from services.pose_extraction import get_skeletal_data
from services.feature_engineering import prepare_inference_data
from services.inference import run_diagnosis

app = FastAPI(title="ASD Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static folder so React can access BOTH videos and Excel files via URL
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

print("🚀 Loading Spatiotemporal XGBoost Model (91% Accuracy)...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print("✅ Server is ready!")

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    temp_input = os.path.join(TEMP_DIR, f"temp_{file.filename}")
    
    output_filename = f"skeletal_{file.filename.split('.')[0]}.webm"
    output_path = os.path.join(STATIC_DIR, output_filename)
    
    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Extract Raw Frames AND Save the Output Video
        raw_df = get_skeletal_data(temp_input, output_path)
        
        if raw_df is None or raw_df.empty:
            return {"status": "error", "message": "Backend returned empty data. No person detected."}

        # 2. Apply Outlier Shield & Extract Features
        windowed_data = prepare_inference_data(raw_df)
        if not windowed_data:
            return {"status": "error", "message": "Video could not be processed."}

        # 3. Predict (FIXED: Added file.filename so the Excel file can be named properly)
        result = run_diagnosis(windowed_data, model, file.filename)
        
        # 4. Inject the Video URL into the JSON
        result["video_url"] = f"http://127.0.0.1:8000/static/{output_filename}"
        
        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}
        
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)