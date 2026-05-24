"""Explainability for the CLIP ViT-L/14 AI-image detector.
 
Provides two complementary saliency methods:
  - Grad-CAM (via pytorch-grad-cam): gradient-based, class-specific.
  - Attention Rollout (Abnar & Zuidema, 2020): gradient-free, model-intrinsic.
 
Both produce a 16x16 heatmap (the patch grid for ViT-L/14 @ 224 input) that
gets upsampled to 224x224 for overlay on the original image.
"""

from __future__ import annotations
 
import contextlib
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
 
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
 
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
 
 
# ---------------------------------------------------------------------------
# Model: CLIP ViT-L/14 visual encoder + classifier head
# ---------------------------------------------------------------------------
 
def build_classifier_head(embed_dim: int = 768) -> nn.Sequential:
    """Reproduce classifier architecture exactly (so the .pth loads cleanly).
 
    Matches the Sequential in CLIP.py:
        Linear(embed_dim, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 2)
    """
    return nn.Sequential(
        nn.Linear(embed_dim, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, 2),
    )
 
 
class CLIPDetector(nn.Module):
    """Combined CLIP visual encoder + classifier head.
 
    For inference and interpretability only — not for training. All parameters
    are kept frozen (requires_grad=False) to match how the classifier was
    trained (on top of frozen CLIP features). Gradients still flow through
    *activations* during a no_grad-free forward pass, which is what Grad-CAM
    needs to hook.
 
    Note: The training script L2-normalizes the CLIP feature before the
    head. We replicate that here — otherwise the head sees unnormalized
    feature magnitudes and predictions become meaningless.
    """
 
    def __init__(self, clip_model: nn.Module, classifier: nn.Module,
                 normalize_features: bool = True):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = classifier
        self.normalize_features = normalize_features
 
        for p in self.parameters():
            p.requires_grad_(False)
 
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # encode_image runs the visual transformer + final projection (1024 -> 768).
        # We do NOT wrap this in torch.no_grad — Grad-CAM needs the autograd graph.
        features = self.clip_model.encode_image(images)
        if self.normalize_features:
            features = features / features.norm(dim=-1, keepdim=True)
        return self.classifier(features.float())
 
 
# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
 
def reshape_transform_clip_vit(tensor: torch.Tensor,
                               height: int = 16, width: int = 16) -> torch.Tensor:
    """Convert ViT token sequence -> spatial feature map for pytorch-grad-cam.
 
    open_clip's transformer runs in sequence-first layout (L, N, D), where
    L = 1 (CLS) + 256 (16x16 patches) = 257 for ViT-L/14 @ 224.
 
    Steps:
      (L, N, D) -> (N, L, D)        permute back to batch-first
      drop the CLS token at index 0
      (N, 256, D) -> (N, 16, 16, D) reshape to spatial grid
      (N, 16, 16, D) -> (N, D, 16, 16)   channels-first for Grad-CAM
    """
    if tensor.shape[0] == 257:
        # (L, N, D) -> (N, L, D)
        tensor = tensor.permute(1, 0, 2)
    elif tensor.shape[1] == 257:
        # already batch-first
        pass
    else:
        raise ValueError(
            f"Unexpected tensor shape {tensor.shape}; expected L=257 on dim 0 or 1. "
            "Check that you're using ViT-L/14 @ 224 input."
        )
 
    tensor = tensor[:, 1:, :]  # drop CLS
    n, _, d = tensor.shape
    tensor = tensor.reshape(n, height, width, d)
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor
 
 
def gradcam_heatmap(detector: CLIPDetector,
                    image_tensor: torch.Tensor,
                    target_class: int = 1,
                    method: str = "gradcam") -> np.ndarray:
    """Compute a Grad-CAM heatmap for one image.
 
    Args:
        detector: CLIPDetector wrapping CLIP + classifier head.
        image_tensor: (1, 3, 224, 224) preprocessed with CLIP's transform.
        target_class: 0 = real, 1 = AI. Determines whose gradient drives the CAM.
        method: 'gradcam' or 'gradcam++'.
 
    Returns:
        Grayscale heatmap of shape (224, 224), values in [0, 1].
    """
    # ln_1 of the LAST transformer block is the standard ViT Grad-CAM target.
    # This is the layer norm immediately before the final attention layer —
    # capturing activations here gives the cleanest spatial structure
    # (post-attention representations get mixed across the sequence).
    target_layer = detector.clip_model.visual.transformer.resblocks[-1].ln_1
    cam_cls = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus}[method]
 
    with cam_cls(
        model=detector,
        target_layers=[target_layer],
        reshape_transform=reshape_transform_clip_vit,
    ) as cam:
        image_tensor = image_tensor.clone().requires_grad_(True)
        grayscale = cam(
            input_tensor=image_tensor,
            targets=[ClassifierOutputTarget(target_class)],
        )
    # Shape: (B, 224, 224)
    return grayscale[0]
 
 
# ---------------------------------------------------------------------------
# Attention Rollout (Abnar & Zuidema, 2020)
# ---------------------------------------------------------------------------
 
@contextlib.contextmanager
def _capture_attention_weights(detector: CLIPDetector):
    """Monkey-patch each ResidualAttentionBlock.attention to also stash weights.
 
    open_clip's MHA call uses need_weights=False (fast path), so we have to
    replace the bound `attention()` method on each block with a version that
    sets need_weights=True and caches the result. We restore on exit.
 
    Yields the list of blocks; after the wrapped forward pass, each block has
    a ._attn_weights attribute of shape (B, n_heads, L, L).
    """
    blocks = list(detector.clip_model.visual.transformer.resblocks)
 
    def make_capturing_attention(blk):
        def attention(q_x, k_x=None, v_x=None, attn_mask=None):
            k_x = k_x if k_x is not None else q_x
            v_x = v_x if v_x is not None else q_x
            if attn_mask is not None:
                attn_mask = attn_mask.to(q_x.dtype)
            out, weights = blk.attn(
                q_x, k_x, v_x,
                need_weights=True,
                average_attn_weights=False,  # keep per-head; we fuse manually
                attn_mask=attn_mask,
            )
            blk._attn_weights = weights.detach()
            return out
        return attention
 
    for block in blocks:
        block.attention = make_capturing_attention(block)
    try:
        yield blocks
    finally:
        for block in blocks:
            # Remove instance attr so the original class method is restored.
            block.__dict__.pop("attention", None)
            block.__dict__.pop("_attn_weights", None)
 
 
def attention_rollout(detector: CLIPDetector,
                      image_tensor: torch.Tensor,
                      head_fusion: str = "mean",
                      discard_ratio: float = 0.0) -> np.ndarray:
    """Compute Attention Rollout — accumulated CLS-to-patch attention.
 
    Args:
        detector: CLIPDetector.
        image_tensor: (1, 3, 224, 224) preprocessed.
        head_fusion: 'mean' | 'max' | 'min' — how to combine per-head attention.
        discard_ratio: zero out the lowest fraction of attention values
            per layer before normalization (recommended: 0.0 - 0.9; common
            choice is 0.9 to suppress diffuse low-attention noise).
 
    Returns:
        Heatmap of shape (224, 224), values in [0, 1] after min-max scaling.
    """
    with _capture_attention_weights(detector) as blocks:
        with torch.no_grad():
            _ = detector(image_tensor)
 
        # Collect per-layer attention, fuse heads
        attentions = []
        for block in blocks:
            a = block._attn_weights  # (B, H, L, L)
            if head_fusion == "mean":
                a = a.mean(dim=1)
            elif head_fusion == "max":
                a = a.max(dim=1).values
            elif head_fusion == "min":
                a = a.min(dim=1).values
            else:
                raise ValueError(f"Unknown head_fusion {head_fusion!r}")
            attentions.append(a[0])  # assume batch=1; -> (L, L)
 
    # Rollout: R_k = A_k_hat @ A_{k-1}_hat @ ... @ A_1_hat,
    # where A_l_hat = normalize(A_l + I) accounts for residual connections.
    L = attentions[0].shape[-1]
    device = attentions[0].device
    eye = torch.eye(L, device=device)
    result = eye.clone()
 
    for a in attentions:
        if discard_ratio > 0:
            # Zero the smallest `discard_ratio` fraction of entries
            # (Chefer et al. style noise suppression). Don't discard the
            # CLS row of itself.
            flat = a.flatten()
            k = int(flat.numel() * discard_ratio)
            if k > 0:
                threshold = flat.kthvalue(k).values
                a = torch.where(a < threshold, torch.zeros_like(a), a)
 
        a_hat = a + eye
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)
        result = a_hat @ result
 
    # CLS row, drop CLS column -> 256-D vector of CLS attention to each patch
    cls_attn = result[0, 1:].float().cpu().numpy()
 
    grid = int(round(cls_attn.size ** 0.5))
    assert grid * grid == cls_attn.size, f"Non-square patch grid: {cls_attn.size}"
    heatmap = cls_attn.reshape(grid, grid)
 
    # Min-max scale and upsample to 224
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = cv2.resize(heatmap.astype(np.float32), (224, 224),
                         interpolation=cv2.INTER_CUBIC)
    return heatmap
 
 
# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
 
def overlay_heatmap(pil_image: Image.Image,
                    heatmap: np.ndarray,
                    alpha: float = 0.45,
                    colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Overlay a [0,1] heatmap on a PIL image. Returns (224, 224, 3) uint8 RGB."""
    img = np.array(pil_image.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
 
    if heatmap.shape != (224, 224):
        heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.clip(heatmap, 0.0, 1.0)
 
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
 
    out = alpha * colored + (1 - alpha) * img
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
 
 
def save_comparison_figure(pil_image: Image.Image,
                           gradcam_map: np.ndarray,
                           rollout_map: np.ndarray,
                           output_path: Path,
                           title: str = "") -> None:
    """Save a side-by-side: original | Grad-CAM overlay | rollout overlay."""
    import matplotlib.pyplot as plt
 
    gc_overlay = overlay_heatmap(pil_image, gradcam_map)
    ro_overlay = overlay_heatmap(pil_image, rollout_map)
    original = np.array(pil_image.convert("RGB").resize((224, 224)))
 
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, img, label in zip(
        axes,
        [original, gc_overlay, ro_overlay],
        ["original", "Grad-CAM", "Attention Rollout"],
    ):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
 
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Convenience: build the whole pipeline from disk
# ---------------------------------------------------------------------------
 
def get_device() -> torch.device:
    """Prefer MPS on Apple Silicon, CUDA on NVIDIA, CPU otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
 
 
def load_detector(classifier_ckpt: str | Path,
                  device: Optional[torch.device] = None,
                  model_name: str = "ViT-L-14",
                  pretrained: str = "laion2b_s32b_b82k",
                  cache_dir: str = "./clip_cache"):
    """Load CLIP + trained classifier into a CLIPDetector.
 
    Returns (detector, preprocess_transform, device).
    """
    import open_clip
 
    device = device or get_device()
 
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, cache_dir=cache_dir,
    )
    clip_model = clip_model.to(device).eval()
    embed_dim = clip_model.text_projection.shape[-1]  # 768 for ViT-L/14
 
    classifier = build_classifier_head(embed_dim)
    state = torch.load(str(classifier_ckpt), map_location=device)
    classifier.load_state_dict(state)
    classifier = classifier.to(device).eval()
 
    detector = CLIPDetector(clip_model, classifier).to(device).eval()
    return detector, preprocess, device