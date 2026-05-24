"""Explainability for the CLIP AI-image detector."""
 
from .cam import (
    CLIPDetector,
    attention_rollout,
    build_classifier_head,
    get_device,
    gradcam_heatmap,
    load_detector,
    overlay_heatmap,
    save_comparison_figure,
)
 
__all__ = [
    "CLIPDetector",
    "attention_rollout",
    "build_classifier_head",
    "get_device",
    "gradcam_heatmap",
    "load_detector",
    "overlay_heatmap",
    "save_comparison_figure",
]
 