# A Comparative Study of AI-Generated Image Detection Across Generative Models

**CS 610 — Advanced Artificial Intelligence**
**Team:** Preston Bied, Luke Ricciardi, Rahul Ukkalam

This project investigates cross-generator generalization for AI-generated image detection using **CLIP (ViT-L/14)** as a frozen feature extractor. We train and evaluate on the [GenImage benchmark](https://genimage-dataset.github.io/) via the [tiny-genimage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) Kaggle subset, which covers 7 generators spanning both diffusion models and GANs, with perfectly balanced train/val splits totaling 35,000 images.

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd ai-image-detection

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your Kaggle API token (one-time, per machine)
#    a. Go to https://www.kaggle.com/settings/account
#    b. Scroll to "API" and click "Create New Token" — downloads kaggle.json
#    c. Save it:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 5. Download the dataset
./scripts/download_tiny_genimage.sh
```

The script is idempotent — if `data/tiny_genimage/` already exists and is non-empty, it skips the download.

---

## Dataset

**Source:** [yangsangtai/tiny-genimage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) on Kaggle (CC-BY-NC-SA-4.0)

**Generators (7):**

| Generator | Family | Native Resolution |
|-----------|--------|-------------------|
| adm | Diffusion | 256×256 |
| biggan | GAN | 128×128 |
| glide | Diffusion | 256×256 |
| midjourney | Diffusion | 1024×1024 |
| sdv15 | Diffusion (Stable Diffusion v1.5) | 512×512 |
| vqdm | Diffusion | 256×256 |
| wukong | Diffusion | 512×512 |

> **Note:** `sdv14` (Stable Diffusion v1.4) is not included in this Kaggle subset and is absent from this study.

**Counts per generator:**
- Train: 2,000 AI + 2,000 real = 4,000 images
- Val: 500 AI + 500 real = 1,000 images
- **Total: 35,000 images** (~7.8 GB extracted)

**Directory structure:**
```
data/tiny_genimage/
└── <generator>/
    ├── train/
    │   ├── ai/
    │   └── nature/
    └── val/
        ├── ai/
        └── nature/
```

Real images (`nature/`) are from ImageNet. AI images are generator-specific.