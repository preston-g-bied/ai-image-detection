#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/data/tiny_genimage"
ZIP_DIR="$REPO_ROOT/data/tiny_genimage_zip"
ZIP_FILE="$ZIP_DIR/tiny-genimage.zip"

# ── Idempotency check ──────────────────────────────────────────────────────────
if [[ -d "$DEST" && -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
    echo "Already downloaded, skipping."
    echo "Dataset is at: $DEST"
    du -sh "$DEST"
    exit 0
fi

# ── Dependency: kaggle CLI ─────────────────────────────────────────────────────
if ! command -v kaggle &>/dev/null; then
    echo ""
    echo "ERROR: The 'kaggle' CLI is not installed."
    echo ""
    echo "Install it with:"
    echo "  pip install kaggle"
    echo ""
    exit 1
fi

# ── Dependency: kaggle.json API token ─────────────────────────────────────────
if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo ""
    echo "ERROR: Kaggle API token not found at ~/.kaggle/kaggle.json"
    echo ""
    echo "To create one:"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Scroll to the 'API' section and click 'Create New Token'"
    echo "  3. A file named kaggle.json will be downloaded"
    echo "  4. Move it into place:"
    echo "       mkdir -p ~/.kaggle"
    echo "       mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json"
    echo "       chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# ── Download ───────────────────────────────────────────────────────────────────
mkdir -p "$ZIP_DIR"
echo "Downloading tiny-genimage from Kaggle (~7.8 GB)..."
kaggle datasets download -d yangsangtai/tiny-genimage -p "$ZIP_DIR"

# ── Extract ────────────────────────────────────────────────────────────────────
mkdir -p "$DEST"
echo "Extracting to $DEST ..."
python3 -c "
import zipfile, sys
with zipfile.ZipFile('$ZIP_FILE', 'r') as zf:
    zf.extractall('$DEST')
print('Extraction complete.')
"

# ── Cleanup zip ────────────────────────────────────────────────────────────────
rm -rf "$ZIP_DIR"
echo "Zip deleted."

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "Done! Dataset ready at: $DEST"
du -sh "$DEST"
