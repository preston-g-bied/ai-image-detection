# scripts/train_clip_classifier.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from pathlib import Path
import open_clip
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import warnings
import os
import json
import csv

warnings.filterwarnings('ignore')

# Set environment variables for Hugging Face
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_TOKEN'] = ''  # Add token here (if available)

# Get the repo root directory (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / 'data' / 'tiny_genimage'

print(f"Script directory: {SCRIPT_DIR}")
print(f"Repo root: {REPO_ROOT}")
print(f"Data directory: {DATA_DIR}")
print(f"Data directory exists: {DATA_DIR.exists()}")


class TinyGenImagePerGeneratorDataset(Dataset):
    """Dataset for per-generator layout:
    
    data/tiny-genimage/<generator>/<split>/{ai,nature}/*.jpg
    
    Records which generator each sample came from for per-generator metrics.
    """
    
    def __init__(self, root_dir=None, split='train', transform=None, generators=None):
        """
        Args:
            root_dir: Root directory containing generator subdirectories (default: ../data/tiny-genimage)
            split: 'train' or 'val'
            transform: CLIP transforms to apply
            generators: List of generator names to include (None = all generators)
        """
        if root_dir is None:
            self.root_dir = DATA_DIR
        else:
            self.root_dir = Path(root_dir)
        
        self.split = split
        self.transform = transform
        self.samples = []
        self.gen_indices = []
        self.gen_names = []
        
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Get list of generators
        if generators is None:
            # Get all subdirectories in root_dir
            self.generators = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        else:
            self.generators = generators
        
        print(f"\nFound {len(self.generators)} generators:")
        for gen in self.generators:
            print(f"  - {gen}")
        
        # Load samples from each generator
        for gen_idx, gen in enumerate(self.generators):
            split_path = self.root_dir / gen / self.split
            if not split_path.exists():
                print(f"  Warning: {split_path} not found, skipping {gen}")
                continue
            
            gen_sample_count = 0
            for label_dir, label in [('ai', 1), ('nature', 0)]:
                img_dir = split_path / label_dir
                if img_dir.exists():
                    for img_path in img_dir.rglob('*'):
                        if img_path.suffix.lower() in self.valid_extensions:
                            self.samples.append((str(img_path), label))
                            self.gen_indices.append(gen_idx)
                            gen_sample_count += 1
            
            print(f"  {gen}: {gen_sample_count} samples")
        
        self.gen_indices = np.array(self.gen_indices, dtype=np.int32)
        
        print(f"\nTotal: Loaded {len(self.samples)} samples from {split} split")
        print(f"  AI samples: {sum(1 for _, label in self.samples if label == 1)}")
        print(f"  Nature samples: {sum(1 for _, label in self.samples if label == 0)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), self.gen_indices[idx]


class FeatureDataset(Dataset):
    """Dataset for pre-extracted features"""
    def __init__(self, features, targets, gen_indices=None):
        self.features = features
        self.targets = targets
        self.gen_indices = gen_indices
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        if self.gen_indices is not None:
            return self.features[idx], self.targets[idx], self.gen_indices[idx]
        return self.features[idx], self.targets[idx]


def compute_metrics_per_generator(predictions, probs, targets, gen_indices, generators):
    """Compute metrics with per-generator breakdown"""
    
    predictions = np.array(predictions) if isinstance(predictions, list) else predictions
    probs = np.array(probs) if isinstance(probs, list) else probs
    targets = np.array(targets) if isinstance(targets, list) else targets
    gen_indices = np.array(gen_indices) if isinstance(gen_indices, list) else gen_indices
    
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # Overall metrics
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'auc': float('nan'),
        'per_generator': {}
    }
    
    try:
        if len(np.unique(targets)) > 1:
            metrics['auc'] = roc_auc_score(targets, probs)
    except:
        pass
    
    # Per-generator metrics
    unique_gens = np.unique(gen_indices)
    for gen_idx in unique_gens:
        mask = gen_indices == gen_idx
        gen_targets = targets[mask]
        gen_preds = predictions[mask]
        gen_probs = probs[mask]
        
        gen_metrics = {
            'accuracy': accuracy_score(gen_targets, gen_preds),
            'samples': len(gen_targets),
            'generator_name': generators[gen_idx]
        }
        
        try:
            if len(np.unique(gen_targets)) > 1:
                gen_metrics['auc'] = roc_auc_score(gen_targets, gen_probs)
            else:
                gen_metrics['auc'] = float('nan')
        except:
            gen_metrics['auc'] = float('nan')
        
        metrics['per_generator'][generators[gen_idx]] = gen_metrics
    
    return metrics


def extract_features_with_generators(model, dataloader, device):
    """Extract features from CLIP model and track generator indices"""
    all_features = []
    all_targets = []
    all_gen_indices = []
    
    for images, targets, gen_indices in tqdm(dataloader, desc="Feature extraction"):
        images = images.to(device)
        with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
        
        all_features.append(features.cpu())
        all_targets.append(targets)
        all_gen_indices.append(gen_indices)
    
    return (torch.cat(all_features), 
            torch.cat(all_targets), 
            torch.cat(all_gen_indices))


def train_with_precomputed_features(model_type='large'):
    """Train using pre-extracted features with mixed precision and gradient accumulation"""
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Set model configuration based on type
    if model_type == 'small':
        model_name = 'ViT-B-32'
        pretrained = 'laion2b_s34b_b79k'
        print("\n" + "="*50)
        print("TRAINING SMALLER MODEL (ViT-B-32)")
        print("="*50)
    else:
        model_name = 'ViT-L-14'
        pretrained = 'laion2b_s32b_b82k'
        print("\n" + "="*50)
        print("TRAINING LARGER MODEL (ViT-L-14)")
        print("="*50)
    
    # Load transforms
    _, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=pretrained,
        cache_dir=str(REPO_ROOT / 'clip_cache')
    )
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = TinyGenImagePerGeneratorDataset(
        root_dir=DATA_DIR, 
        split='train', 
        transform=preprocess
    )
    val_dataset = TinyGenImagePerGeneratorDataset(
        root_dir=DATA_DIR, 
        split='val', 
        transform=preprocess
    )
    
    # Store generator names for later
    generators = train_dataset.generators
    
    # Create data loaders for feature extraction
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=False,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=False,
        prefetch_factor=2
    )
    
    # Load CLIP model for feature extraction
    print(f"\nLoading {model_name} model...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision='fp16' if torch.cuda.is_available() else 'fp32',
        cache_dir=str(REPO_ROOT / 'clip_cache')
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # Freeze CLIP backbone
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # Get embedding dimension
    embed_dim = clip_model.text_projection.shape[-1]
    print(f"Embedding dimension: {embed_dim}")
    
    # Pre-extract features
    print("\nExtracting training features...")
    train_features, train_targets, train_gen_indices = extract_features_with_generators(
        clip_model, train_loader, device
    )
    print("Extracting validation features...")
    val_features, val_targets, val_gen_indices = extract_features_with_generators(
        clip_model, val_loader, device
    )
    
    print(f"\nFeatures extracted - Train: {train_features.shape}, Val: {val_features.shape}")
    
    # Move features to GPU for faster training (if available)
    train_features = train_features.to(device)
    val_features = val_features.to(device)
    train_targets = train_targets.to(device)
    val_targets = val_targets.to(device)
    train_gen_indices = train_gen_indices.to(device)
    val_gen_indices = val_gen_indices.to(device)
    
    # Create feature datasets
    train_feat_dataset = FeatureDataset(train_features, train_targets, train_gen_indices)
    val_feat_dataset = FeatureDataset(val_features, val_targets, val_gen_indices)
    
    # Create loaders for training with larger batch size
    train_feat_loader = DataLoader(
        train_feat_dataset, 
        batch_size=512,
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    val_feat_loader = DataLoader(
        val_feat_dataset, 
        batch_size=512, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    # Train classifier
    classifier = nn.Sequential(
        nn.Linear(embed_dim, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, 2)
    ).to(device)
    
    # AdamW for better training
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Gradient accumulation settings
    accumulation_steps = 4
    effective_batch_size = 512 * accumulation_steps
    
    print(f"\nTraining classifier on pre-extracted features...")
    print(f"Effective batch size: {effective_batch_size}")
    
    best_accuracy = 0
    results_history = []
    
    for epoch in range(20):
        # Training phase
        classifier.train()
        train_loss = 0
        train_preds, train_targets_list = [], []
        optimizer.zero_grad()
        
        for batch_idx, (features, targets, _) in enumerate(tqdm(train_feat_loader, desc=f"Epoch {epoch+1}")):
            with autocast(enabled=torch.cuda.is_available()):
                logits = classifier(features)
                loss = criterion(logits, targets)
                loss = loss / accumulation_steps  # Normalize loss for gradient accumulation
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps  # Denormalize loss
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets_list.extend(targets.cpu().numpy())
        
        # Handle remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        train_acc = accuracy_score(train_targets_list, train_preds)
        
        # Validation phase with per-generator metrics
        classifier.eval()
        val_preds, val_probs, val_targets_list, val_gen_indices_list = [], [], [], []
        
        with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
            for features, targets, gen_indices in val_feat_loader:
                logits = classifier(features)
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(AI)
                val_probs.extend(probs.cpu().numpy())
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())
                val_gen_indices_list.extend(gen_indices.cpu().numpy())
        
        val_metrics = compute_metrics_per_generator(
            val_preds, val_probs, val_targets_list, val_gen_indices_list, generators
        )
        
        # Store results
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_loss/len(train_feat_loader),
            'train_acc': train_acc,
            'val_acc': val_metrics['accuracy'],
            'val_auc': val_metrics['auc'],
            'per_generator': val_metrics['per_generator']
        }
        results_history.append(epoch_results)
        
        print(f"\nEpoch {epoch+1}/20")
        print(f"Train Loss: {epoch_results['train_loss']:.4f}, Train Acc: {epoch_results['train_acc']:.4f}")
        print(f"Val Acc: {epoch_results['val_acc']:.4f}, Val AUC: {epoch_results['val_auc']:.4f}")
        
        # Print per-generator accuracy
        print("\nPer-generator validation accuracy:")
        for gen_name, gen_metrics in val_metrics['per_generator'].items():
            print(f"  {gen_name}: Acc={gen_metrics['accuracy']:.4f}, AUC={gen_metrics['auc']:.4f}, n={gen_metrics['samples']}")
        
        # Save best model
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            model_save_path = REPO_ROOT / 'models' / f'best_classifier_{model_type}.pth'
            model_save_path.parent.mkdir(exist_ok=True)
            torch.save(classifier.state_dict(), model_save_path)
            print(f"  New best model saved! (Acc: {best_accuracy:.4f}) -> {model_save_path}")
        
        print()
    
    # Save training results
    save_training_results(results_history, model_type, generators)
    
    return results_history, best_accuracy


def save_training_results(results_history, model_type, generators):
    """Save training results to file"""
    
    results_dir = REPO_ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    json_file = results_dir / f'training_results_{model_type}.json'
    with open(json_file, 'w') as f:
        json.dump(results_history, f, indent=2)
    print(f"\nResults saved to {json_file}")
    
    # Save as CSV for easy viewing (overall metrics)
    csv_file = results_dir / f'training_results_{model_type}.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_acc', 'val_acc', 'val_auc'])
        writer.writeheader()
        for epoch_results in results_history:
            writer.writerow({
                'epoch': epoch_results['epoch'],
                'train_loss': epoch_results['train_loss'],
                'train_acc': epoch_results['train_acc'],
                'val_acc': epoch_results['val_acc'],
                'val_auc': epoch_results['val_auc']
            })
    print(f"Results saved to {csv_file}")
    
    # Save per-generator results
    per_gen_file = results_dir / f'per_generator_results_{model_type}.csv'
    with open(per_gen_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'generator', 'accuracy', 'auc', 'samples'])
        for epoch_results in results_history:
            epoch = epoch_results['epoch']
            for gen_name, gen_metrics in epoch_results['per_generator'].items():
                writer.writerow([
                    epoch, gen_name, 
                    gen_metrics['accuracy'], 
                    gen_metrics['auc'], 
                    gen_metrics['samples']
                ])
    print(f"Per-generator results saved to {per_gen_file}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY FOR {model_type.upper()} MODEL")
    print(f"{'='*50}")
    best_epoch = max(results_history, key=lambda x: x['val_acc'])
    print(f"Best validation accuracy: {best_epoch['val_acc']:.4f} at epoch {best_epoch['epoch']}")
    print(f"Best validation AUC: {best_epoch['val_auc']:.4f}")
    print(f"Final validation accuracy: {results_history[-1]['val_acc']:.4f}")
    print(f"Final validation AUC: {results_history[-1]['val_auc']:.4f}")


def compare_models():
    """Compare the results of both models"""
    print("\n" + "="*60)
    print("LOADING RESULTS FOR COMPARISON")
    print("="*60)
    
    results_dir = REPO_ROOT / 'results'
    
    # Load results if they exist
    small_results = None
    large_results = None
    
    try:
        with open(results_dir / 'training_results_small.json', 'r') as f:
            small_results = json.load(f)
        print("✓ Loaded small model results")
    except FileNotFoundError:
        print("✗ Small model results not found. Run small model first.")
    
    try:
        with open(results_dir / 'training_results_large.json', 'r') as f:
            large_results = json.load(f)
        print("✓ Loaded large model results")
    except FileNotFoundError:
        print("✗ Large model results not found. Run large model first.")
    
    if small_results and large_results:
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        
        small_best = max(small_results, key=lambda x: x['val_acc'])
        large_best = max(large_results, key=lambda x: x['val_acc'])
        
        print(f"\nLarge Model (ViT-L-14) Best Accuracy: {large_best['val_acc']:.4f}")
        print(f"Small Model (ViT-B-32) Best Accuracy: {small_best['val_acc']:.4f}")
        print(f"Difference: {abs(large_best['val_acc'] - small_best['val_acc']):.4f}")
        
        # Create comparison plot
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 5))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot([r['epoch'] for r in large_results], [r['val_acc'] for r in large_results], 
                    label='ViT-L-14 (Large)', marker='o', linewidth=2)
            plt.plot([r['epoch'] for r in small_results], [r['val_acc'] for r in small_results], 
                    label='ViT-B-32 (Small)', marker='s', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Validation Accuracy', fontsize=12)
            plt.title('Model Comparison - Validation Accuracy', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Plot AUC
            plt.subplot(1, 2, 2)
            plt.plot([r['epoch'] for r in large_results], [r['val_auc'] for r in large_results], 
                    label='ViT-L-14 (Large)', marker='o', linewidth=2)
            plt.plot([r['epoch'] for r in small_results], [r['val_auc'] for r in small_results], 
                    label='ViT-B-32 (Small)', marker='s', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Validation AUC', fontsize=12)
            plt.title('Model Comparison - Validation AUC', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(results_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
            print("\n✓ Comparison plot saved as 'results/model_comparison.png'")
            
        except ImportError:
            print("\n⚠ Matplotlib not installed. Install with: pip install matplotlib")
    else:
        print("\n⚠ Cannot compare models. Make sure both models have been trained.")


if __name__ == "__main__":
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU detected. Training will be slow on CPU.")
    
    # Verify data directory exists
    if not DATA_DIR.exists():
        print(f"\n ERROR: Data directory not found: {DATA_DIR}")
        print(f"Please ensure the data directory exists at: {DATA_DIR}")
        exit(1)
    
    # Choose which model(s) to run
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    print("Options:")
    print("1. Train both models (large and small)")
    print("2. Train only large model (ViT-L-14)")
    print("3. Train only small model (ViT-B-32)")
    print("4. Compare existing results (without training)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        print("\n" + "="*60)
        print("TRAINING BOTH MODELS")
        print("="*60)
        
        large_results, large_best_acc = train_with_precomputed_features(model_type='large')
        small_results, small_best_acc = train_with_precomputed_features(model_type='small')
        compare_models()
        
    elif choice == '2':
        print("\n" + "="*60)
        print("TRAINING LARGE MODEL ONLY")
        print("="*60)
        results, best_acc = train_with_precomputed_features(model_type='large')
        print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")
        
    elif choice == '3':
        print("\n" + "="*60)
        print("TRAINING SMALL MODEL ONLY")
        print("="*60)
        results, best_acc = train_with_precomputed_features(model_type='small')
        print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")
        
    elif choice == '4':
        compare_models()
        
    else:
        print("Invalid choice. Please run again and select 1-4.")