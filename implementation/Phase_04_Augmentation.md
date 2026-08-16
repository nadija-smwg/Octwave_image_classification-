# Phase 4 — Online Data Augmentation Pipeline (PRIMARY)

```
Module: src/augment.py
Priority: CRITICAL — online augmentation is the PRIMARY regularizer and data diversity strategy
GPU Required: No (CPU-based transforms)
Estimated Time: Implementation only, no training
Dependencies: Phase 2 (dataset class), Phase 0 (config)
Key Changes (Optimization Update):
  - Promoted to CRITICAL/PRIMARY (§1 — preferred over offline augmentation)
  - Removed Perspective and aggressive rotation (§3 — cartoon frames are upright)
  - RandomResizedCrop now default for ALL presets (§3)
  - CutMix/MixUp are individually toggleable, not both by default (§3, §14)
  - Removed minority_extra_aug (§2 — use WeightedRandomSampler instead)
  - Removed random_erasing_prob (redundant with CoarseDropout)
  - Emphasized Color/Hue/Saturation jitter to prevent color shortcuts (§3)
```

---

## 4.1 Objective

> [!IMPORTANT]
> **This is the PRIMARY augmentation strategy.** The model should see a different transformed version of each image during every epoch, creating effectively unlimited diversity from the 2,680 original images.
>
> **Why online augmentation is preferred over offline (§1):**
> - No unnecessary disk usage
> - Effectively unlimited variations
> - Reduces overfitting
> - Allows more experimentation
> - Works well with `WeightedRandomSampler` (minority classes appear more frequently, each time with a different transform)

> [!NOTE]
> If offline augmentation (Phase 3B) is also enabled, online augmentation applies **on top of** the offline-augmented dataset. Each copy gets further randomized every epoch.

Build a comprehensive, configurable augmentation pipeline using **Albumentations** that:
1. Prevents overfitting on the training images
2. Uses **cartoon-specific transforms** (§3) — horizontal flip, color jitter, random crop+resize
3. **Avoids unrealistic transforms** — no aggressive rotation, no perspective distortion
4. Includes advanced batch-level techniques: CutMix OR MixUp (tested individually)
5. Has three strength presets: light, medium, heavy
6. Respects the configured resolution (128–256 progressive resizing)

---

## 4.2 Cartoon-Specific Augmentation Strategy (§3)

> [!IMPORTANT]
> The augmentation strategy must match the actual test distribution. Cartoon frames are **always upright**, characters have **distinctive colors** (Tom=grey, Jerry=brown), and they appear at **varying positions and scales**.

### What to Use (Strongly Consider)

| Transform | Why | Notes |
|-----------|-----|-------|
| **HorizontalFlip** | Flipping preserves whether Tom/Jerry/both are present | Verify against dataset |
| **Color/Hue/Saturation Jitter** | Prevents color shortcuts (grey=Tom) | Moderate intensity |
| **RandomResizedCrop** | Characters at different positions/scales/distances | Primary spatial augmentation |

### What to Avoid

| Transform | Why Avoid |
|-----------|-----------|
| **Aggressive Rotation (≥10°)** | Cartoon frames are always upright |
| **Perspective** | Unrealistic for 2D cartoon frames |
| **Extreme Affine** | Introduces noise that doesn't match test distribution |

### Standard Transforms (Applied to All Samples)

| Transform | Light | Medium | Heavy |
|-----------|-------|--------|-------|
| **RandomResizedCrop** | scale=(0.85,1.0) | scale=(0.75,1.0) | scale=(0.65,1.0) |
| **HorizontalFlip** | p=0.5 | p=0.5 | p=0.5 |
| **ShiftScaleRotate** (rotate ≤8°) | — | shift=0.06, scale=0.1, rot=5°, p=0.4 | shift=0.1, scale=0.15, rot=8°, p=0.5 |
| **RandomBrightnessContrast** | ±0.15 | ±0.25 | ±0.3 |
| **HueSaturationValue** | hue=10, sat=15 | hue=15, sat=20 | hue=20, sat=25 |
| **ColorJitter** | — | brightness=0.2, hue=0.05, p=0.3 | brightness=0.3, hue=0.08, p=0.4 |
| **GaussNoise** | — | var=10–40, p=0.2 | var=10–50, p=0.3 |
| **GaussianBlur** | — | blur_limit=3, p=0.15 | blur_limit=5, p=0.2 |
| **CoarseDropout** | — | max_holes=4, size=16, p=0.2 | max_holes=6, size=24, p=0.3 |
| **Normalize** | ImageNet stats | ImageNet stats | ImageNet stats |
| **ToTensorV2** | ✓ | ✓ | ✓ |

---

## 4.3 Functions to Implement

### 4.3.1 `get_train_transform(cfg) -> A.Compose`

**Purpose:** Build the training augmentation pipeline based on config strength.

> [!NOTE]
> **Key changes from previous version (§3):**
> - `RandomResizedCrop` is now the **default spatial transform** for all presets (not just heavy)
> - No standalone `Rotate` transform — cartoon frames are upright
> - `ShiftScaleRotate` rotation limited to ≤8°
> - Color/Hue jitter is emphasized to prevent color shortcuts

**Implementation:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(cfg):
    strength = cfg.augmentation.strength  # "light" | "medium" | "heavy"
    img_size = cfg.resolution.current     # Uses progressive resizing
    
    transforms = []
    
    # --- PRIMARY: RandomResizedCrop (§3) ---
    # Used for ALL presets: characters appear at varying positions/scales
    scale_min = {"light": 0.85, "medium": 0.75, "heavy": 0.65}[strength]
    transforms.append(A.RandomResizedCrop(
        height=img_size, width=img_size,
        scale=(scale_min, 1.0), ratio=(0.85, 1.15)
    ))
    
    # --- HorizontalFlip (§3: strongly recommended for cartoons) ---
    transforms.append(A.HorizontalFlip(p=0.5))
    
    # --- ShiftScaleRotate (MILD rotation only, §3) ---
    if strength in ("medium", "heavy"):
        rotate_limit = 5 if strength == "medium" else 8  # §3: NO aggressive rotation
        transforms.append(A.ShiftScaleRotate(
            shift_limit=0.06 if strength == "medium" else 0.1,
            scale_limit=0.1 if strength == "medium" else 0.15,
            rotate_limit=rotate_limit,  # §3: cartoon frames are upright
            p=0.4 if strength == "medium" else 0.5
        ))
    # NOTE: No Rotate or Perspective for any preset (§3)
    
    # --- Color/Hue/Saturation Jitter (§3: prevent color shortcuts) ---
    bc_limit = {"light": 0.15, "medium": 0.25, "heavy": 0.3}[strength]
    transforms.append(A.RandomBrightnessContrast(
        brightness_limit=bc_limit, contrast_limit=bc_limit, p=0.5
    ))
    
    hsv_hue = {"light": 10, "medium": 15, "heavy": 20}[strength]
    hsv_sat = {"light": 15, "medium": 20, "heavy": 25}[strength]
    transforms.append(A.HueSaturationValue(
        hue_shift_limit=hsv_hue, sat_shift_limit=hsv_sat, val_shift_limit=hsv_sat, p=0.4
    ))
    
    # Additional color jitter for medium/heavy (§3: encourages shape/structure learning)
    if strength in ("medium", "heavy"):
        transforms.append(A.ColorJitter(
            brightness=0.2 if strength == "medium" else 0.3,
            contrast=0.2 if strength == "medium" else 0.3,
            saturation=0.2 if strength == "medium" else 0.3,
            hue=0.05 if strength == "medium" else 0.08,
            p=0.3 if strength == "medium" else 0.4
        ))
    
    # --- Noise/Blur (medium, heavy only) ---
    if strength in ("medium", "heavy"):
        transforms.append(A.GaussNoise(var_limit=(10, 40 if strength == "medium" else 50), p=0.2 if strength == "medium" else 0.3))
        transforms.append(A.GaussianBlur(blur_limit=3 if strength == "medium" else 5, p=0.15 if strength == "medium" else 0.2))
    
    # --- CoarseDropout (medium, heavy) ---
    if strength == "heavy":
        transforms.append(A.CoarseDropout(
            max_holes=6, max_height=24, max_width=24,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, p=0.3
        ))
    elif strength == "medium":
        transforms.append(A.CoarseDropout(
            max_holes=4, max_height=16, max_width=16,
            fill_value=0, p=0.2
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
    img_size = cfg.resolution.current  # Uses progressive resizing
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

---

### 4.3.3 Batch-Level Augmentation: MixUp and CutMix

> [!IMPORTANT]
> **Optimization Update (§3, §14):** MixUp and CutMix should be tested **individually**, not simultaneously by default. Run controlled experiments:
> - Experiment 6: Add MixUp only
> - Experiment 7: Add CutMix only (replace MixUp)
> - Keep whichever provides measurable improvement on validation Macro F1.
> - Do NOT automatically use both.

MixUp and CutMix operate on **batches**, not individual images. They're applied inside the training loop, not in the albumentations pipeline.

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
    # Decide augmentation type for this batch
    r = np.random.rand()
    if cfg.augmentation.use_cutmix and r < 0.5:
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
  strength: "medium"           # "light" | "medium" | "heavy"
  # Cartoon-specific: NO aggressive rotation/perspective (§3)
  # Emphasize: HFlip, Color/Hue jitter, RandomResizedCrop
  use_cutmix: false            # Test individually (§3, §14)
  cutmix_alpha: 1.0
  use_mixup: false             # Test individually (§3, §14)
  mixup_alpha: 0.4
  # REMOVED: random_erasing_prob (redundant with CoarseDropout)
  # REMOVED: minority_extra_aug (use WeightedRandomSampler instead, §2)
```

---

## 4.5 Design Rationale

| Decision | Rationale |
|----------|-----------|
| Albumentations over torchvision | ~3× faster (OpenCV backend), richer transform library |
| ImageNet normalization | Required for pretrained backbone compatibility |
| CutMix OR MixUp (not both default) | Test individually, keep only what helps (§3, §14) |
| `RandomResizedCrop` for ALL presets | Cartoon characters appear at varying positions/scales (§3) |
| NO `Rotate` or `Perspective` | Cartoon frames are always upright (§3) |
| Color/Hue jitter emphasized | Prevents color shortcuts (grey=Tom, brown=Jerry) (§3) |
| Three strength presets | Quick experimentation: start medium, try heavy if overfitting |
| `WeightedRandomSampler` over `minority_extra_aug` | Simpler, same effect (§2) |

---

## 4.6 Outputs

| Output | Description |
|--------|-------------|
| `outputs/augmentation_preview.png` | Visual grid of augmented samples per strength |
| No persistent files | Transforms are created on-the-fly during training |
