# Phase 9 — Test-Time Augmentation (TTA) & Inference

```
Module: src/infer.py
Priority: HIGH — TTA is a free performance boost at inference
GPU Required: Yes
Estimated Time: 10–15 minutes for full ensemble + TTA
Dependencies: Phase 8 (trained models)
```

---

## 9.1 Objective

Run inference on the test set with:
1. **Test-Time Augmentation (TTA)** — multiple augmented views per image, average predictions
2. **Multi-model ensemble** — average across all fold/backbone models
3. Combine TTA + ensemble for maximum performance

---

## 9.2 TTA Strategy

For each test image, create multiple augmented versions and average their predictions:

| TTA Transform | Description |
|---------------|-------------|
| **Original** | Standard resize + normalize (always included) |
| **Horizontal flip** | Mirror image left-right |
| **Top-left crop** | Crop 90% from top-left, resize |
| **Top-right crop** | Crop 90% from top-right, resize |
| **Bottom-left crop** | Crop 90% from bottom-left, resize |
| **Bottom-right crop** | Crop 90% from bottom-right, resize |

**Total predictions per image per model:** 6 (1 original + 5 augmented)

---

## 9.3 Functions to Implement

### 9.3.1 `get_tta_transforms(cfg) -> list[A.Compose]`

**Purpose:** Create list of TTA transforms.

```python
def get_tta_transforms(cfg):
    """
    Create list of TTA transforms. Each produces a differently-augmented
    view of the same image.
    """
    img_size = cfg.data.image_size
    norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    tta_list = []
    
    # 1. Original (always included)
    tta_list.append(A.Compose([
        A.Resize(img_size, img_size),
        norm,
        ToTensorV2(),
    ]))
    
    if not cfg.tta.enabled:
        return tta_list
    
    # 2. Horizontal flip
    if "hflip" in cfg.tta.transforms:
        tta_list.append(A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),  # Always flip for TTA
            norm,
            ToTensorV2(),
        ]))
    
    # 3. Corner crops
    crop_size = int(img_size * 0.9)
    crop_positions = {
        "crop_tl": (0, 0),                           # top-left
        "crop_tr": (0, img_size - crop_size),         # top-right
        "crop_bl": (img_size - crop_size, 0),         # bottom-left
        "crop_br": (img_size - crop_size, img_size - crop_size),  # bottom-right
    }
    
    for crop_name, (y, x) in crop_positions.items():
        if crop_name in cfg.tta.transforms:
            tta_list.append(A.Compose([
                A.Resize(img_size, img_size),
                A.Crop(x_min=x, y_min=y, x_max=x + crop_size, y_max=y + crop_size),
                A.Resize(img_size, img_size),
                norm,
                ToTensorV2(),
            ]))
    
    print(f"TTA: {len(tta_list)} transforms configured")
    return tta_list
```

---

### 9.3.2 `predict_with_tta(model, test_df, cfg) -> np.array`

**Purpose:** Run TTA inference for a single model.

```python
@torch.no_grad()
def predict_with_tta(model, test_df, cfg):
    """
    Predict test set with TTA.
    
    For each TTA transform, run full inference on the test set,
    then average the softmax probabilities.
    
    Returns:
        np.array: (N_test, num_classes) averaged softmax probabilities
    """
    model.eval()
    tta_transforms = get_tta_transforms(cfg)
    
    all_tta_probs = []
    
    for tta_idx, transform in enumerate(tta_transforms):
        print(f"  TTA {tta_idx + 1}/{len(tta_transforms)}...")
        
        dataset = TomJerryDataset(test_df, cfg.paths.image_dir, transform, is_test=True)
        loader = DataLoader(
            dataset, batch_size=cfg.training.batch_size * 2,
            shuffle=False, num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory
        )
        
        batch_probs = []
        for images, filenames in loader:
            images = images.to(cfg.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=cfg.training.use_amp):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            batch_probs.append(probs.cpu().numpy())
        
        tta_probs = np.concatenate(batch_probs)  # (N_test, 4)
        all_tta_probs.append(tta_probs)
    
    # Average across TTA transforms
    avg_probs = np.mean(all_tta_probs, axis=0)  # (N_test, 4)
    
    return avg_probs
```

---

### 9.3.3 `ensemble_predict(model_paths, test_df, cfg) -> np.array`

**Purpose:** Ensemble predictions from multiple models + TTA.

```python
def ensemble_predict(model_paths, test_df, cfg, weights=None):
    """
    Run ensemble inference: load each model, predict with TTA, average.
    
    Args:
        model_paths: List of checkpoint paths
        test_df: Test DataFrame
        cfg: Config
        weights: Optional model weights (from CV F1). None = equal weights.
    
    Returns:
        np.array: (N_test, num_classes) final ensemble probabilities
    """
    if weights is None:
        weights = np.ones(len(model_paths)) / len(model_paths)
    else:
        weights = np.array(weights) / np.sum(weights)
    
    print(f"\nEnsemble inference with {len(model_paths)} models...")
    
    all_model_probs = []
    
    for i, ckpt_path in enumerate(model_paths):
        print(f"\nModel {i + 1}/{len(model_paths)}: {os.path.basename(ckpt_path)}")
        
        # Load model (auto-detect backbone from checkpoint)
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        model_cfg = checkpoint.get('cfg', cfg)
        model = TomJerryClassifier(model_cfg).to(cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Predict with TTA
        model_probs = predict_with_tta(model, test_df, cfg)
        all_model_probs.append(model_probs)
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Weighted average across models
    ensemble_probs = np.zeros_like(all_model_probs[0])
    for probs, w in zip(all_model_probs, weights):
        ensemble_probs += w * probs
    
    return ensemble_probs
```

---

### 9.3.4 `analyze_predictions(probs, test_df, cfg)`

**Purpose:** Post-inference analysis before generating submission.

```python
def analyze_predictions(probs, test_df, cfg):
    """
    Analyze prediction confidence and distribution.
    Helps decide if the model is well-calibrated.
    """
    predictions = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)
    
    print(f"\n{'='*50}")
    print("PREDICTION ANALYSIS")
    print(f"{'='*50}")
    
    # 1. Predicted class distribution
    print("\nPredicted class distribution (test set):")
    for cls in range(cfg.num_classes):
        count = (predictions == cls).sum()
        pct = count / len(predictions) * 100
        print(f"  {cfg.class_names[cls]}: {count} ({pct:.1f}%)")
    
    # 2. Confidence distribution
    print(f"\nConfidence statistics:")
    print(f"  Mean: {max_probs.mean():.4f}")
    print(f"  Median: {np.median(max_probs):.4f}")
    print(f"  Min: {max_probs.min():.4f}")
    print(f"  < 0.5: {(max_probs < 0.5).sum()} samples")
    print(f"  < 0.7: {(max_probs < 0.7).sum()} samples")
    print(f"  > 0.9: {(max_probs > 0.9).sum()} samples")
    
    # 3. Save confidence histogram
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=50, edgecolor='black')
    plt.xlabel('Max Softmax Probability')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.savefig(f"{cfg.paths.output_dir}/predictions/confidence_dist.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return predictions, max_probs
```

---

## 9.4 Inference Pipeline

```mermaid
graph TD
    A["Load test.csv"] --> B["For each model checkpoint"]
    B --> C["Load model weights"]
    C --> D["For each TTA transform"]
    D --> E["Predict probabilities"]
    E --> F["Average across TTA"]
    F --> G["Weight by model quality"]
    G --> H["Sum weighted probabilities"]
    H --> I["Argmax → final predictions"]
    I --> J["Analyze & validate"]
    J --> K["Generate submission CSV"]
```

---

## 9.5 Configuration Knobs

```yaml
tta:
  enabled: true
  transforms:
    - "hflip"
    - "crop_tl"
    - "crop_tr"
    - "crop_bl"
    - "crop_br"
  n_augments: 5               # For reference (auto-computed from list)
```

---

## 9.6 Performance Impact

| Method | Expected Improvement |
|--------|---------------------|
| Single model, no TTA | Baseline |
| Single model + TTA (6 views) | +0.5–1.5% Macro F1 |
| 5-fold ensemble, no TTA | +1–3% Macro F1 |
| 5-fold ensemble + TTA | +1.5–4% Macro F1 |
| 15-model multi-backbone + TTA | +2–5% Macro F1 |

---

## 9.7 Time Estimates

| Setup | Inference Time (T4 GPU) |
|-------|------------------------|
| 1 model, no TTA | ~30 seconds |
| 1 model, 6× TTA | ~3 minutes |
| 5 models, 6× TTA | ~15 minutes |
| 15 models, 6× TTA | ~45 minutes |

> [!TIP]
> If time-constrained, use TTA with just 2 transforms (original + hflip). This gives ~80% of the TTA benefit in ~33% of the time.
