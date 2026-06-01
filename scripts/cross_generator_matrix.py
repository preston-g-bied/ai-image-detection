"""Cross-generator generalization matrix.

The headline experiment of the project: train a classifier on each generator's
training set individually, then evaluate on all 7 generators' validation sets.
Produces a 7x7 matrix of accuracy and AUC values — the diagonal is in-distribution
(train and eval on the same generator) and off-diagonal cells measure how well
the detector generalizes to unseen generators.

Outputs (all in results/):
    cross_generator_results.json        — full per-cell metrics
    cross_generator_acc_matrix.png      — 7x7 accuracy heatmap
    cross_generator_auc_matrix.png      — 7x7 AUC heatmap
    cross_generator_features.pt         — cached CLIP features (re-runs are fast)

Workflow:
    1. Extract CLIP ViT-L/14 features for the entire train+val sets ONCE.
       Cached to disk so subsequent runs skip this step.
    2. For each generator G_train, train an MLP head on G_train's training
       features only (~4000 samples, ~5 seconds per training run).
    3. For each (G_train, G_eval) pair, evaluate on G_eval's val set.

Runtime: ~15-20 min the first time (feature extraction dominates).
         ~1-2 min on subsequent runs (features cached, only training reruns).

Reuses TinyGenImagePerGeneratorDataset from CLIP.py to stay in sync with
the rest of the pipeline.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make the repo root importable so we can pull the dataset class from CLIP.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.CLIP import TinyGenImagePerGeneratorDataset  # noqa: E402


# tiny-genimage directory names -> short labels used in figures/tables.
# Anything not in this map will fall back to its raw directory name.
GEN_NAME_MAP = {
    'imagenet_ai_0419_biggan':   'biggan',
    'imagenet_ai_0419_vqdm':     'vqdm',
    'imagenet_ai_0424_sdv5':     'sdv5',
    'imagenet_ai_0424_wukong':   'wukong',
    'imagenet_ai_0508_adm':      'adm',
    'imagenet_glide':            'glide',
    'imagenet_midjourney':       'midjourney',
}

def clean_name(dir_name):
    return GEN_NAME_MAP.get(dir_name, dir_name)


DATA_ROOT = Path('data/tiny_genimage')
RESULTS_DIR = Path('results')
FEATURES_CACHE = RESULTS_DIR / 'cross_generator_features.pt'


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_head(embed_dim=768):
    """Matches CLIP.py's classifier head exactly."""
    return nn.Sequential(
        nn.Linear(embed_dim, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, 2),
    )


# ---------------------------------------------------------------------------
# Feature extraction (one-time, cached)
# ---------------------------------------------------------------------------

def extract_all_features(device):
    """Extract CLIP features for train + val across all 7 generators.
    Returns dict with tensors keyed by '<split>_features' / '<split>_labels' /
    '<split>_gen_indices'. Caches to disk."""

    if FEATURES_CACHE.exists():
        print(f"Loading cached features from {FEATURES_CACHE}...")
        return torch.load(FEATURES_CACHE)

    print("\nLoading CLIP ViT-L/14 (LAION-2B)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='laion2b_s32b_b82k',
        cache_dir='./clip_cache',
    )
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    out = {}
    discovered = None  # actual on-disk directory names, locked after train
    for split in ['train', 'val']:
        print(f"\n=== Extracting {split} features ===")
        dataset = TinyGenImagePerGeneratorDataset(
            root_dir=str(DATA_ROOT), split=split, transform=preprocess,
            generators=discovered,  # None first pass (auto-discover); locked for val
        )
        if discovered is None:
            discovered = dataset.generators
            print(f"\nDiscovered generators: {discovered}")
        # num_workers=2 for image loading is fine; the MPS-tensor-sharing
        # issue only affects DataLoaders over already-on-device tensors.
        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            num_workers=2, pin_memory=True)

        feats, labels = [], []
        for batch in tqdm(loader, desc=f'{split}'):
            # Dataset returns (image, label) or (image, label, gen_idx);
            # only the first two matter here.
            images, lbls = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                f = clip_model.encode_image(images)
                # L2-normalize — matches the training pipeline. Skipping this
                # would put the head out of distribution.
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu())
            labels.append(lbls)

        out[f'{split}_features'] = torch.cat(feats)
        out[f'{split}_labels'] = torch.cat(labels)
        out[f'{split}_gen_indices'] = torch.from_numpy(dataset.gen_indices).long()

    out['generators'] = discovered  # stash so reruns from cache get the same order

    FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, FEATURES_CACHE)
    print(f"\nFeatures cached to {FEATURES_CACHE} "
          f"({FEATURES_CACHE.stat().st_size / 1e6:.1f} MB)")
    return out


# ---------------------------------------------------------------------------
# Per-generator training and evaluation
# ---------------------------------------------------------------------------

def train_head_on_subset(features, labels, device,
                         epochs=20, batch_size=512, lr=1e-3):
    """Train a fresh MLP head on the given feature/label subset."""
    head = build_head(embed_dim=features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    features = features.to(device)
    labels = labels.to(device)
    n = len(features)

    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            logits = head(features[idx])
            loss = criterion(logits, labels[idx])
            loss.backward()
            optimizer.step()
    return head


def evaluate_head(head, features, labels, device):
    """Return (accuracy, AUC) on the given subset."""
    head.eval()
    features = features.to(device)
    with torch.no_grad():
        logits = head(features)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    acc = accuracy_score(labels_np, preds)
    auc = float('nan')
    if len(np.unique(labels_np)) > 1:
        try:
            auc = roc_auc_score(labels_np, probs)
        except Exception:
            pass
    return float(acc), float(auc)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_heatmap(matrix, labels, metric_name, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='viridis', vmin=0.4, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Evaluation generator')
    ax.set_ylabel('Training generator')
    ax.set_title(f'Cross-generator {metric_name}')

    # Annotate each cell with its value; flip text color for readability.
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if matrix[i, j] < 0.7 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.2f}',
                    ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label=metric_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cache = extract_all_features(device)
    train_features = cache['train_features']
    train_labels = cache['train_labels']
    train_gen = cache['train_gen_indices']
    val_features = cache['val_features']
    val_labels = cache['val_labels']
    val_gen = cache['val_gen_indices']

    GENERATORS_DIR = cache['generators']                    # on-disk names
    LABELS = [clean_name(g) for g in GENERATORS_DIR]        # short display names
    n = len(GENERATORS_DIR)
    print(f"\nGenerators (n={n}): {LABELS}")

    acc_matrix = np.zeros((n, n))
    auc_matrix = np.zeros((n, n))
    results = {'generators': LABELS,
               'generator_dirs': list(GENERATORS_DIR),
               'cells': {}}

    for i, gen_train in enumerate(LABELS):
        print(f"\n{'=' * 60}")
        print(f"Training on {gen_train} (row {i + 1}/{n})")
        print(f"{'=' * 60}")

        mask = (train_gen == i)
        gen_features = train_features[mask]
        gen_labels = train_labels[mask]
        print(f"  Train subset: {len(gen_features)} samples "
              f"(AI: {(gen_labels == 1).sum().item()}, "
              f"Real: {(gen_labels == 0).sum().item()})")

        t0 = time.time()
        head = train_head_on_subset(gen_features, gen_labels, device)
        print(f"  Trained in {time.time() - t0:.1f}s")

        for j, gen_eval in enumerate(LABELS):
            eval_mask = (val_gen == j)
            acc, auc = evaluate_head(
                head, val_features[eval_mask], val_labels[eval_mask], device,
            )
            acc_matrix[i, j] = acc
            auc_matrix[i, j] = auc
            results['cells'][f'{gen_train}->{gen_eval}'] = {
                'accuracy': acc, 'auc': auc,
                'n_eval': int(eval_mask.sum().item()),
            }
            marker = '*' if i == j else ' '
            print(f"  {marker} eval on {gen_eval:12s}  "
                  f"acc={acc:.4f}  auc={auc:.4f}")

    # Pretty-print the matrix to stdout
    print(f"\n{'=' * 70}")
    print("CROSS-GENERATOR ACCURACY MATRIX")
    print('=' * 70)
    print(f"{'train \\ eval':14s}", end='')
    for g in LABELS:
        print(f"{g[:10]:>10s}", end='')
    print()
    for i, gen_train in enumerate(LABELS):
        print(f"{gen_train:14s}", end='')
        for j in range(n):
            print(f"{acc_matrix[i, j]:10.3f}", end='')
        print()

    # Headline numbers for the paper: in-distribution vs out-of-distribution.
    diag = acc_matrix.diagonal()
    off_diag = acc_matrix[~np.eye(n, dtype=bool)].reshape(n, n - 1)

    print(f"\nIn-distribution mean acc (diagonal):       {diag.mean():.4f}")
    print(f"Out-of-distribution mean acc (off-diag):   {off_diag.mean():.4f}")
    print(f"Generalization gap:                        "
          f"{diag.mean() - off_diag.mean():.4f}")

    # Per-generator generalization quality.
    print("\nPer-generator out-of-distribution mean accuracy:")
    for i, g in enumerate(LABELS):
        ood = np.delete(acc_matrix[i], i).mean()
        print(f"  Trained on {g:12s} -> avg acc on others: {ood:.4f}")

    results['summary'] = {
        'in_distribution_mean_acc': float(diag.mean()),
        'out_of_distribution_mean_acc': float(off_diag.mean()),
        'generalization_gap': float(diag.mean() - off_diag.mean()),
        'per_generator_ood_mean_acc': {
            g: float(np.delete(acc_matrix[i], i).mean())
            for i, g in enumerate(LABELS)
        },
    }

    json_path = RESULTS_DIR / 'cross_generator_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults JSON: {json_path}")

    plot_heatmap(acc_matrix, LABELS, 'Accuracy',
                 RESULTS_DIR / 'cross_generator_acc_matrix.png')
    plot_heatmap(auc_matrix, LABELS, 'AUC',
                 RESULTS_DIR / 'cross_generator_auc_matrix.png')
    print(f"Heatmaps: {RESULTS_DIR}/cross_generator_*_matrix.png")


if __name__ == '__main__':
    main()