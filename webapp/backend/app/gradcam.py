"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ConvNeXt-Tiny.
Targets the output of the final feature stage.
Produces an explanatory visual overlay highlighting regions influencing model classification.
"""

import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm
import io
import base64


def generate_heatmap(model: torch.nn.Module, features: torch.Tensor, class_idx: int) -> np.ndarray:
    """Compute Grad-CAM from features already produced by the prediction pass."""
    with torch.enable_grad():
        activations = features.detach().requires_grad_()
        logits = model.classifier(model.avgpool(activations))
        (gradients,) = torch.autograd.grad(logits[0, class_idx], activations)

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features.detach()).sum(dim=1).relu()[0].cpu().numpy()
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        return (cam - cam_min) / (cam_max - cam_min)
    return np.zeros_like(cam)


def overlay_gradcam(original_img: Image.Image, heatmap_2d: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Overlays a 2D heatmap on the original PIL image using the 'jet' colormap.
    Returns composite PIL Image in RGB format.
    """
    orig_w, orig_h = original_img.size
    
    # Resize heatmap to match image dimensions
    cam_pil = Image.fromarray((heatmap_2d * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    # Use the 256 grayscale values as indices to avoid a full-size float RGBA array.
    try:
        import matplotlib
        colormap = matplotlib.colormaps["jet"]
    except Exception:
        colormap = cm.get_cmap("jet")
    colors = (colormap(np.arange(256) / 255.0)[:, :3] * 255).astype(np.uint8)
    colored_heatmap = Image.fromarray(colors[np.asarray(cam_pil)])
    return Image.blend(original_img.convert("RGB"), colored_heatmap, alpha)


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Converts a PIL Image to a data-URI base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=format, quality=90)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format.upper() in ("JPG", "JPEG") else "image/png"
    return f"data:{mime};base64,{b64_str}"
