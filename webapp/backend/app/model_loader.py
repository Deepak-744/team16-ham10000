"""
Model Loader for ConvNeXt-Tiny trained on HAM10000.
Loads the checkpoint verified from the training pipeline without dummy substitutions.
"""

import os
import glob
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from .class_info import CLASS_TO_IDX, IDX_TO_CLASS

# Global model cache
_LOADED_MODEL = None
_LOADED_METADATA = None


def find_checkpoint_path() -> str:
    """
    Uses the configured checkpoint path or searches the local project directories.
    """
    configured_path = os.environ.get("MODEL_CHECKPOINT_PATH")
    if configured_path:
        if not os.path.isfile(configured_path):
            raise FileNotFoundError(f"MODEL_CHECKPOINT_PATH does not point to a file: {configured_path}")
        return configured_path

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parent_root = os.path.abspath(os.path.join(project_root, ".."))
    models_dir = os.path.join(project_root, "backend", "models")
    
    # Priority paths:
    candidates = [
        # Direct file or zip placed by user in project root or parent
        os.path.join(project_root, "best_v5_convnext_tiny224_rs_inverse_auto_cap1200_floor500_strong_ls0.1_sch-ema-tta_ep25_best_seed42.pth.zip"),
        os.path.join(parent_root, "best_v5_convnext_tiny224_rs_inverse_auto_cap1200_floor500_strong_ls0.1_sch-ema-tta_ep25_best_seed42.pth.zip"),
        os.path.join(project_root, "best_v5_convnext_tiny224_rs_inverse_auto_cap1200_floor500_strong_ls0.1_sch-ema-tta_ep25_best_seed42.pth"),
        os.path.join(parent_root, "best_v5_convnext_tiny224_rs_inverse_auto_cap1200_floor500_strong_ls0.1_sch-ema-tta_ep25_best_seed42.pth"),
        os.path.join(models_dir, "best_model.pth"),
        os.path.join(models_dir, "model.pth")
    ]
    
    # Also glob for any convnext .pth or .pth.zip in root, parent or models dir
    candidates.extend(glob.glob(os.path.join(project_root, "*convnext*.pth*")))
    candidates.extend(glob.glob(os.path.join(parent_root, "*convnext*.pth*")))
    candidates.extend(glob.glob(os.path.join(models_dir, "*convnext*.pth*")))
    
    for path in candidates:
        if os.path.isfile(path):
            return path
            
    raise FileNotFoundError(
        f"ConvNeXt-Tiny checkpoint not found. Looked in project root: {project_root} and {models_dir}"
    )


def build_convnext_tiny(num_classes: int = 7) -> nn.Module:
    """
    Constructs the exact ConvNeXt-Tiny model used in training:
    torchvision convnext_tiny with classifier[2] replaced with nn.Linear(768, 7).
    """
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features  # 768
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def get_model(device: torch.device = None) -> tuple[nn.Module, dict]:
    """
    Loads and returns the cached PyTorch model and metadata.
    Loads on CPU by default (fast and robust for web inference).
    """
    global _LOADED_MODEL, _LOADED_METADATA
    
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL, _LOADED_METADATA
        
    if device is None:
        device = torch.device("cpu")
        
    ckpt_path = find_checkpoint_path()
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Load PyTorch checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Check if dict with model_state_dict or direct state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_to_idx = checkpoint.get("class_to_idx", CLASS_TO_IDX)
        best_epoch = checkpoint.get("best_epoch", 20)
        backbone = checkpoint.get("backbone", "convnext_tiny")
        bias = checkpoint.get("bias", None)
    else:
        state_dict = checkpoint
        class_to_idx = CLASS_TO_IDX
        best_epoch = 20
        backbone = "convnext_tiny"
        bias = None
        
    model = build_convnext_tiny(num_classes=len(class_to_idx))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    _LOADED_MODEL = model
    _LOADED_METADATA = {
        "checkpoint_path": ckpt_path,
        "class_to_idx": class_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()},
        "best_epoch": best_epoch,
        "backbone": backbone,
        "bias": bias
    }
    
    print(f"Successfully loaded {backbone} model (best epoch {best_epoch}) onto {device}.")
    return _LOADED_MODEL, _LOADED_METADATA
