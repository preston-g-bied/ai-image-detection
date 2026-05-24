"""Generate Grad-CAM + Attention Rollout visualizations for sample val images.

Usage:
    python scripts/explain_samples.py \\
        --data-root data/tiny_genimage \\
        --classifier-ckpt best_classifier_large.pth \\
        --output-dir outputs/explainability \\
        --n-per-class 3

Default: 3 AI + 3 real per generator across all 7 generators = 42 figures.

Each figure shows: original | Grad-CAM (predicted class) | Attention Rollout,
plus a title noting ground-truth label, predicted label, and AI-class probability.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Make 'src' importable when running this script directly from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explainability import (
    attention_rollout,
    get_device,
    gradcam_heatmap,
    load_detector,
    save_comparison_figure,
)


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_DIRS = {"ai": 1, "nature": 0}
CLASS_NAMES = {0: "real", 1: "AI"}


def collect_samples(data_root: Path, generators, n_per_class: int):
    """Yield (path, label, generator_name) for the first n images in each
    val/<class> directory of each generator."""
    samples = []
    available_generators = sorted(
        d.name for d in data_root.iterdir() if d.is_dir()
    )
    generators = generators or available_generators

    for gen in generators:
        gen_dir = data_root / gen
        if not gen_dir.is_dir():
            print(f"  [warn] generator dir not found: {gen_dir}", file=sys.stderr)
            continue
        for class_name, label in CLASS_DIRS.items():
            img_dir = gen_dir / "val" / class_name
            if not img_dir.is_dir():
                continue
            paths = sorted(
                p for p in img_dir.iterdir()
                if p.suffix.lower() in VALID_EXTS
            )[:n_per_class]
            for p in paths:
                samples.append((p, label, gen))
    return samples


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/tiny_genimage", type=Path)
    p.add_argument("--classifier-ckpt", default="best_classifier_large.pth", type=Path)
    p.add_argument("--output-dir", default="outputs/explainability", type=Path)
    p.add_argument("--n-per-class", type=int, default=3,
                   help="How many images to sample per (generator, class).")
    p.add_argument("--generators", nargs="+", default=None,
                   help="Which generators to process. Default: all found under data-root.")
    p.add_argument("--rollout-discard", type=float, default=0.9,
                   help="Attention rollout: fraction of lowest-attention entries to "
                        "zero per layer (0.0 = no suppression, 0.9 = strong suppression).")
    p.add_argument("--cam-method", default="gradcam", choices=["gradcam", "gradcam++"])
    p.add_argument("--device", default=None,
                   help="Override device (cuda/mps/cpu). Default: auto-detect.")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    if not args.classifier_ckpt.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {args.classifier_ckpt}\n"
            "Run CLIP.py first (option 2) to produce best_classifier_large.pth."
        )

    print(f"Loading detector from {args.classifier_ckpt}...")
    detector, preprocess, device = load_detector(args.classifier_ckpt, device=device)

    samples = collect_samples(args.data_root, args.generators, args.n_per_class)
    print(f"Found {len(samples)} samples to process.")
    if not samples:
        print("Nothing to do. Check --data-root.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for img_path, label, gen in tqdm(samples, desc="Explaining"):
        pil = Image.open(img_path).convert("RGB")
        x = preprocess(pil).unsqueeze(0).to(device)

        # Predict (for figure caption and to pick Grad-CAM target class)
        with torch.no_grad():
            logits = detector(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(probs.argmax())

        # Grad-CAM: saliency for the PREDICTED class (i.e. what convinced the
        # model). Swap to target_class=label to show "what *should* have
        # mattered" if you want to study failure cases.
        gc_map = gradcam_heatmap(detector, x, target_class=pred,
                                 method=args.cam_method)
        ro_map = attention_rollout(detector, x, discard_ratio=args.rollout_discard)

        title = (
            f"{gen} | gt={CLASS_NAMES[label]} | "
            f"pred={CLASS_NAMES[pred]} (P(AI)={probs[1]:.2f})"
        )
        out_path = (
            args.output_dir
            / f"{gen}_gt-{CLASS_NAMES[label]}_pred-{CLASS_NAMES[pred]}_{img_path.stem}.png"
        )
        save_comparison_figure(pil, gc_map, ro_map, out_path, title=title)

    print(f"\nDone. {len(samples)} figures in {args.output_dir}/")


if __name__ == "__main__":
    main()