# Phase 4 — Online Data Augmentation Pipeline

```
Module: src/augment.py
Priority: HIGH — online augmentation is the primary regularizer
GPU Required: No (CPU-based transforms)
Estimated Time: Implementation only, no training
Dependencies: Phase 2 (dataset class), Phase 3B (offline-augmented dataset), Phase 0 (config)
```

---

## 4.1 Objective

> [!NOTE]
> **This is ONLINE augmentation** — random transforms applied on-the-fly during each training epoch.
> It works **on top of** Phase 3B's offline-augmented dataset (~5,720 images).
> Each offline-augmented copy gets further randomized by online augmentation every epoch, creating **massive effective diversity**.
>
> | Layer | Phase | What happens | Effective multiplier |
> |-------|-------|-------------|---------------------|
> | **Offline** (Phase 3B) | Before training | Creates permanent new images on disk | ~2.1× (2,680 → 5,720) |
> | **Online** (this phase) | During each epoch | Applies random transforms per batch | ∞ (different every epoch) |
> | **CutMix/MixUp** (batch-level) | During each step | Mixes images within a batch | Further diversity |

Build a comprehensive, configurable augmentation pipeline using **Albumentations** that:
1. Prevents overfitting on the ~5,720 expanded training images
2. Applies **heavier augmentation to minority classes** (0, 3) — additional to offline aug
3. Includes advanced batch-level techniques: CutMix, MixUp, Random Erasing
4. Has three strength presets: light, medium, heavy
5. Respects the configured resolution (192×192 fast mode or 224×224 quality mode)

---

## 4.2 Augmentation Strategy by Strength

### Standard Transforms (Applied to All Samples)

| Transform | Light | Medium | Heavy |
|-----------|-------|--------|-------|
| **Resize** | 224×224 | 224×224 | 256×256 → RandomCrop(224) |
| **HorizontalFlip** | p=0.5 | p=0.5 | p=0.5 |
| **Rotation** | ±10° | ±15° | ±20° |
| **ShiftScaleRotate** | — | shift=0.0625, scale=0.1, rotate=15 | shift=0.1, scale=0.15, rotate=20 |
| **RandomBrightnessContrast** | ±0.1 | ±0.2 | ±0.3 |
| **HueSaturationValue** | ±10 | ±20 | ±30 |
| **GaussNoise** | — | var=10–50, p=0.3 | var=10–50, p=0.5 |
| **GaussianBlur** | — | blur_limit=3, p=0.2 | blur_limit=5, p=0.3 |
| **CoarseDropout** | — | max_holes=4, size=16 | max_holes=8, size=32 |
| **RandomResizedCrop** | scale=(0.9,1.0) | scale=(0.8,1.0) | scale=(0.7,1.0) |
| **Normalize** | ImageNet stats | ImageNet stats | ImageNet stats |
| **ToTensorV2** | ✓ | ✓ | ✓ |

---

## 4.3 Functions to Implement

### 4.3.1 `get_train_transform(cfg) -> A.Compose`

**Purpose:** Build the training augmentation pipeline based on config strength.

**Implementation:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(cfg):
    strength = cfg.augmentation.strength  # "light" | "medium" | "heavy"
    img_size = cfg.data.image_size
    
    transforms = []
    
    if strength == "heavy":
        transforms.append(A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)))
    else:
        transforms.append(A.Resize(img_size, img_size))
    
    transforms.append(A.HorizontalFlip(p=0.5))
    
    if strength in ("medium", "heavy"):
        rotate_limit = 15 if strength == "medium" else 20
        transforms.append(A.ShiftScaleRotate(
            shift_limit=0.0625 if strength == "medium" else 0.1,
            scale_limit=0.1 if strength == "medium" else 0.15,
            rotate_limit=rotate_limit,
            p=0.5
        ))
    else:
        transforms.append(A.Rotate(limit=10, p=0.3))
    
    bc_limit = {"light": 0.1, "medium": 0.2, "heavy": 0.3}[strength]
    transforms.append(A.RandomBrightnessContrast(
        brightness_limit=bc_limit, contrast_limit=bc_limit, p=0.5
    ))
    
    hsv_limit = {"light": 10, "medium": 20, "heavy": 30}[strength]
    transforms.append(A.HueSaturationValue(
        hue_shift_limit=hsv_limit, sat_shift_limit=hsv_limit, val_shift_limit=hsv_limit, p=0.4
    ))
    
    if strength in ("medium", "heavy"):
        transforms.append(A.GaussNoise(var_limit=(10, 50), p=0.3))
        transforms.append(A.GaussianBlur(blur_limit=3 if strength == "medium" else 5, p=0.2))
    
    if strength == "heavy":
        transforms.append(A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, p=0.4
        ))
    elif strength == "medium":
        transforms.append(A.CoarseDropout(
            max_holes=4, max_height=16, max_width=16,
            fill_value=0, p=0.3
        ))
    
    # Always normalize and convert to tensor
    transforms.append(A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)
```

---

### 4.3.2 `get_val_transform(cfg) -> A.Compose`

**Purpose:** Minimal transforms for validation (deterministic).

```python
def get_val_transform(cfg):
    return A.Compose([
        A.Resize(cfg.data.image_size, cfg.data.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

---

### 4.3.3 `get_minority_transform(cfg) -> A.Compose`

**Purpose:** Extra-heavy augmentation applied **only to minority class samples** (classes 0 and 3).

**Rationale:** Since classes 0 and 3 have far fewer samples, applying more aggressive augmentation to them creates more diverse training examples, helping the model generalize better on these classes.

**Implementation:**
- Same as `get_train_transform(cfg)` with `strength="heavy"`, **plus:**
  - `A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5)`
  - `A.RandomGamma(gamma_limit=(80, 120), p=0.3)`
  - `A.CLAHE(clip_limit=2.0, p=0.3)`
  - `A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2)`

**Integration with Dataset:**
```python
class TomJerryDataset(Dataset):
    def __init__(self, df, image_dir, transform, minority_transform=None, minority_classes=[0, 3]):
        self.minority_transform = minority_transform
        self.minority_classes = minority_classes
        # ... rest of init
    
    def __getitem__(self, idx):
        # ... load image
        label = int(row['appearance'])
        
        # Apply heavier aug to minority classes
        if self.minority_transform and label in self.minority_classes:
            augmented = self.minority_transform(image=image)
        else:
            augmented = self.transform(image=image)
        
        image = augmented['image']
        return image, label
```

---

### 4.3.4 `CutMixUp` — Batch-Level Augmentation

**Purpose:** Implement CutMix and MixUp as batch-level augmentations applied during training.

> [!NOTE]
> CutMix and MixUp operate on **batches**, not individual images. They're applied inside the training loop, not in the albumentations pipeline.

#### MixUp Implementation

```python
def mixup_data(x, y, alpha=0.4):
    """
    Apply MixUp augmentation to a batch.
    
    Args:
        x: Input images (B, C, H, W)
        y: Labels (B,)
        alpha: Beta distribution parameter
    
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

#### CutMix Implementation

```python
def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation to a batch.
    
    Args:
        x: Input images (B, C, H, W)
        y: Labels (B,)
        alpha: Beta distribution parameter
    
    Returns:
        mixed_x, y_a, y_b, lam
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    _, _, H, W = x.shape
    
    # Generate random bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda by actual area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return mixed_x, y, y[index], lam
```

#### Usage in Training Loop (Phase 7)

```python
for images, labels in train_loader:
    # Randomly choose CutMix or MixUp (50/50)
    r = np.random.rand()
    if cfg.augmentation.use_cutmix and r < 0.25:
        images, labels_a, labels_b, lam = cutmix_data(images, labels, cfg.augmentation.cutmix_alpha)
        use_mix = True
    elif cfg.augmentation.use_mixup and r < 0.5:
        images, labels_a, labels_b, lam = mixup_data(images, labels, cfg.augmentation.mixup_alpha)
        use_mix = True
    else:
        use_mix = False
    
    outputs = model(images)
    
    if use_mix:
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
    else:
        loss = criterion(outputs, labels)
```

---

### 4.3.5 `visualize_augmentations(image_path, transforms_dict, save_path)`

**Purpose:** Debug tool — visualize what augmentation does to a single image.

**Implementation:**
- Load one image
- Apply each transform preset 8 times
- Create a grid showing original + 8 augmented versions
- Repeat for light/medium/heavy
- Save to `outputs/augmentation_preview.png`

---

## 4.4 Configuration Knobs

```yaml
augmentation:
  strength: "medium"          # "light" | "medium" | "heavy"
  use_cutmix: true
  cutmix_alpha: 1.0
  use_mixup: true
  mixup_alpha: 0.4
  random_erasing_prob: 0.25
  minority_extra_aug: true    # Apply heavier aug to classes 0 and 3
  minority_classes: [0, 3]
```

---

## 4.5 Design Rationale

| Decision | Rationale |
|----------|-----------|
| Albumentations over torchvision | ~3× faster (OpenCV backend), richer transform library |
| ImageNet normalization | Required for pretrained backbone compatibility |
| CutMix + MixUp combined | Complimentary: CutMix is spatial, MixUp is global blending |
| Heavier minority aug | Creates more diverse samples for underrepresented classes |
| Three strength presets | Quick experimentation: start medium, try heavy if overfitting |
| Batch-level CutMix/MixUp | Can't do in albumentations pipeline — needs batch mixing |

---

## 4.6 Outputs

| Output | Description |
|--------|-------------|
| `outputs/augmentation_preview.png` | Visual grid of augmented samples per strength |
| No persistent files | Transforms are created on-the-fly during training |
