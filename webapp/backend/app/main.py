"""
FastAPI Backend Application for Skin Lesion Detection and Classification.
Provides REST API endpoints for image upload, inference, Grad-CAM, PDF reporting, and health checks.
"""

import os
import io
import glob
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from .model_loader import get_model, find_checkpoint_path
from .preprocessing import validate_and_load_image
from .inference import run_inference
from .class_info import CLASS_DETAILS, MEDICAL_DISCLAIMER, MODEL_EVALUATION_METRICS
from .report_generator import generate_pdf_report
from .gradcam import image_to_base64
from PIL import Image

app = FastAPI(
    title="Skin Lesion AI API",
    description="Deep-Learning-based Multiclass Skin Lesion Classification API using ConvNeXt-Tiny and HAM10000",
    version="1.0.0"
)

# Enable CORS for frontend development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup to ensure fast responses."""
    try:
        print("Pre-loading ConvNeXt-Tiny model...")
        get_model()
        print("Model pre-loaded successfully.")
    except Exception as e:
        print(f"Warning: Model could not be pre-loaded at startup: {e}")


@app.get("/")
def root():
    return {
        "app": "Skin Lesion AI API",
        "status": "online",
        "docs": "/docs",
        "model": "ConvNeXt-Tiny (HAM10000)"
    }


@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend status and model availability."""
    model_loaded = False
    model_path = None
    error = None
    
    try:
        model_path = find_checkpoint_path()
        model, meta = get_model()
        model_loaded = model is not None
    except Exception as e:
        error = str(e)
        
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "checkpoint_path": model_path,
        "device": "cpu",
        "error": error
    }


@app.get("/api/classes")
def get_classes():
    """Returns the seven supported HAM10000 classes, risk tiers, and model evaluation metrics."""
    return {
        "classes": list(CLASS_DETAILS.values()),
        "evaluation_metrics": MODEL_EVALUATION_METRICS,
        "medical_disclaimer": MEDICAL_DISCLAIMER
    }


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...), gradcam: bool = Query(True)):
    """
    Accepts an uploaded skin lesion image (JPG, JPEG, PNG).
    Runs real inference through the trained ConvNeXt-Tiny model.
    Returns predicted class, top 3, full distribution, and Grad-CAM visualization.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided.")
        
    try:
        image_bytes = await file.read()
        pil_img = validate_and_load_image(image_bytes)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")
        
    try:
        result = run_inference(pil_img, generate_cam=gradcam)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/report")
async def generate_report_endpoint(inference_data: Dict[str, Any]):
    """
    Generates a downloadable PDF medical summary report from inference results.
    """
    if not inference_data or "predicted_class" not in inference_data:
        raise HTTPException(status_code=400, detail="Invalid inference payload for PDF generation.")
        
    try:
        pdf_bytes = generate_pdf_report(inference_data)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=Skin_Lesion_AI_Report.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@app.get("/api/sample-images")
def get_sample_images():
    """
    Returns available sample test images stored in the project for instant demo testing.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parent_root = os.path.abspath(os.path.join(project_root, ".."))
    sample_dirs = [os.path.join(project_root, "sample_images"), os.path.join(parent_root, "sample_images")]
    
    samples = []
    seen = set()
    for s_dir in sample_dirs:
        if os.path.isdir(s_dir):
            for img_path in glob.glob(os.path.join(s_dir, "*.jpg")):
                fname = os.path.basename(img_path)
                if fname in seen:
                    continue
                seen.add(fname)
                try:
                    with Image.open(img_path) as im:
                        b64 = image_to_base64(im, format="JPEG")
                    samples.append({
                        "filename": fname,
                        "title": f"Sample: {fname}",
                        "image_base64": b64
                    })
                except Exception:
                    pass
                
    return {"samples": samples}
