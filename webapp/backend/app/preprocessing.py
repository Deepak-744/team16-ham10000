"""
Image Preprocessing and Validation Pipeline for Skin Lesion Classification.
Conforms strictly to the ConvNeXt-Tiny training specifications (224x224 RGB, ImageNet normalization).
"""

import io
from PIL import Image
import torch
from torchvision import transforms

# Exact parameters from training notebook
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Evaluation transform
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_FORMATS = {'JPEG', 'JPG', 'PNG'}


def validate_and_load_image(image_bytes: bytes) -> Image.Image:
    """
    Validates file size, format, and decodes the image safely into RGB PIL Image.
    Raises ValueError on validation failure.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Empty image file received.")
    
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.")
    
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()  # Verify integrity
    except Exception as e:
        raise ValueError(f"Corrupted or invalid image file: {str(e)}")
    
    # Re-open because verify() closes or invalidates the stream
    pil_img = Image.open(io.BytesIO(image_bytes))
    
    # Check format
    fmt = (pil_img.format or "").upper()
    if fmt not in ALLOWED_FORMATS and fmt != 'MPO':
        raise ValueError(f"Unsupported image format: '{fmt}'. Supported formats are JPG, JPEG, and PNG.")
    
    # Convert safely to RGB
    try:
        rgb_img = pil_img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to convert image to RGB: {str(e)}")
    
    return rgb_img


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Applies the exact 224x224 resizing and ImageNet normalization from training.
    Returns a batch tensor of shape (1, 3, 224, 224).
    """
    tensor = eval_transform(pil_img)
    return tensor.unsqueeze(0)  # Add batch dimension: (1, 3, 224, 224)
