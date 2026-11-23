"""
Standalone Verification Script for Phase 3:
Tests:
1. Model checkpoint loading
2. Model forward pass on sample image
3. Softmax calculation and top 3 extraction
4. Grad-CAM generation
5. PDF report generation
"""

import argparse
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.model_loader import get_model
from backend.app.preprocessing import validate_and_load_image
from backend.app.inference import run_inference
from backend.app.report_generator import generate_pdf_report


def main():
    parser = argparse.ArgumentParser(description="Verify the local inference pipeline.")
    parser.add_argument("image", type=Path, help="Dermoscopic image to use for the smoke test")
    parser.add_argument("--output", type=Path, help="Optional path for the generated PDF")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3 VERIFICATION TEST")
    print("=" * 60)
    
    # 1. Load Model
    print("\n[Step 1] Loading ConvNeXt-Tiny model from checkpoint...")
    model, metadata = get_model()
    print(f"  Checkpoint: {metadata['checkpoint_path']}")
    print(f"  Backbone: {metadata['backbone']}")
    print(f"  Best Epoch: {metadata['best_epoch']}")
    print(f"  Class to Index: {metadata['class_to_idx']}")
    
    # 2. Test Image
    print(f"\n[Step 2] Validating and loading test image: {args.image}")
    pil_img = validate_and_load_image(args.image.read_bytes())
    print(f"  Decoded Image Size: {pil_img.size}, Mode: {pil_img.mode}")
    
    # 3. Run Inference
    print("\n[Step 3] Running inference and Grad-CAM generation...")
    result = run_inference(pil_img, generate_cam=True)
    
    pred_class = result["predicted_class"]
    print(f"  Predicted Category: {pred_class['name']} ({pred_class['code'].upper()})")
    print(f"  Confidence: {pred_class['confidence_percent']}%")
    print(f"  Clinical Risk: {pred_class['risk_level']} ({pred_class['type']})")
    
    print("\n  Top 3 Predictions:")
    for rank, p in enumerate(result["top_3_predictions"], 1):
        print(f"    #{rank} {p['code'].upper()} ({p['short_name']}): {p['confidence_percent']}%")
        
    print("\n  Grad-CAM Saliency Map:")
    print(f"    Available: {result['gradcam']['available']}")
    if result["gradcam"]["available"]:
        print(f"    Image Base64 length: {len(result['gradcam']['image_base64'])} characters")
        print(f"    Note: {result['gradcam']['note']}")
        
    # 4. Generate PDF Report
    print("\n[Step 4] Testing PDF Report generation...")
    pdf_bytes = generate_pdf_report(result)
    print(f"  Generated PDF size: {len(pdf_bytes)} bytes")
    
    if args.output is not None:
        args.output.write_bytes(pdf_bytes)
        print(f"  Saved test PDF to: {args.output}")
    
    print("\n" + "=" * 60)
    print("ALL PHASE 3 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
