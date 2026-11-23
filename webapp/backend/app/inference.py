"""
Inference Engine for ConvNeXt-Tiny Skin Lesion Classification.
Coordinates preprocessing, model forward pass, score calculation, and Grad-CAM generation.
"""

from typing import Dict, Any
from PIL import Image
import torch
import numpy as np

from .model_loader import get_model
from .preprocessing import preprocess_image
from .class_info import CLASS_DETAILS, IDX_TO_CLASS, MEDICAL_DISCLAIMER, MODEL_EVALUATION_METRICS
from .gradcam import generate_heatmap, overlay_gradcam, image_to_base64


def run_inference(pil_img: Image.Image, generate_cam: bool = True) -> Dict[str, Any]:
    """
    Executes full inference on an input PIL image:
    1. Preprocesses image to (1, 3, 224, 224)
    2. Runs model inference (eval mode, disabled gradients)
    3. Calculates softmax confidence scores
    4. Extracts Top 1 and Top 3 predictions
    5. Optionally generates Grad-CAM visualization
    6. Attaches clinical context and medical disclaimer
    """
    model, metadata = get_model()
    idx_to_class = metadata.get("idx_to_class", IDX_TO_CLASS)
    
    # Preprocess
    input_tensor = preprocess_image(pil_img)
    
    # Forward pass
    with torch.no_grad():
        features = model.features(input_tensor)
        logits = model.classifier(model.avgpool(features))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    top_idx = int(np.argmax(probabilities))
    top_code = idx_to_class[top_idx]
    top_confidence = float(probabilities[top_idx])
    
    # Build complete distribution (sorted by confidence descending)
    all_predictions = []
    for idx, prob in enumerate(probabilities):
        code = idx_to_class[idx]
        details = CLASS_DETAILS.get(code, {})
        all_predictions.append({
            "code": code,
            "name": details.get("name", code),
            "short_name": details.get("short_name", code),
            "confidence_percent": round(float(prob) * 100, 2),
            "score": float(prob),
            "risk_level": details.get("risk_level", "Unknown"),
            "risk_badge": details.get("risk_badge", "default"),
            "type": details.get("type", "Unknown"),
            "color": details.get("color", "#3B82F6")
        })
        
    all_predictions.sort(key=lambda x: x["confidence_percent"], reverse=True)
    top_3 = all_predictions[:3]
    
    top_class_info = CLASS_DETAILS.get(top_code, {})
    
    # Generate Grad-CAM if requested
    gradcam_base64 = None
    gradcam_available = False
    gradcam_note = None
    
    if generate_cam:
        try:
            heatmap = generate_heatmap(model, features, top_idx)
            
            blended_img = overlay_gradcam(pil_img, heatmap, alpha=0.55)
            gradcam_base64 = image_to_base64(blended_img, format="JPEG")
            gradcam_available = True
            gradcam_note = (
                "Grad-CAM visual overlay highlights salient regions attended by the "
                "neural network's final stage. It is an explanatory visualization, not histological proof."
            )
        except Exception as e:
            print(f"Grad-CAM generation error: {e}")
            gradcam_note = f"Grad-CAM visualization unavailable: {str(e)}"
            
    # Base64 for original image
    original_base64 = image_to_base64(pil_img, format="JPEG")
    
    return {
        "predicted_class": {
            "code": top_code,
            "name": top_class_info.get("name", top_code),
            "short_name": top_class_info.get("short_name", top_code),
            "confidence_percent": round(top_confidence * 100, 2),
            "risk_level": top_class_info.get("risk_level", "Unknown"),
            "risk_badge": top_class_info.get("risk_badge", "default"),
            "type": top_class_info.get("type", "Unknown"),
            "description": top_class_info.get("description", ""),
            "clinical_guidance": top_class_info.get("clinical_guidance", ""),
            "color": top_class_info.get("color", "#3B82F6")
        },
        "top_3_predictions": top_3,
        "confidence_distribution": all_predictions,
        "gradcam": {
            "available": gradcam_available,
            "image_base64": gradcam_base64,
            "note": gradcam_note
        },
        "original_image_base64": original_base64,
        "evaluation_context": MODEL_EVALUATION_METRICS,
        "medical_disclaimer": MEDICAL_DISCLAIMER
    }
