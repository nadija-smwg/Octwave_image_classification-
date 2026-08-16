# Phase 8 — Cross-Validation & Ensembling

```
Module: src/crossval.py
Priority: HIGH — ensembling is the single biggest leaderboard boost
GPU Required: Yes (trains multiple models)
Estimated Time: 2–10 hours depending on configuration
Dependencies: Phases 2–7 (everything up to training)
```

---

## 8.1 Objective

1. Train across **K folds** to get robust local CV estimate
2. **Average softmax probabilities** across fold models for prediction
3. Support **multi-architecture ensembling** (different backbones, not just different folds)
4. Provide a reliable local Macro F1 score to decide which submissions to pick

---

## 8.2 Ensembling Strategies

### Strategy 1: Single-Backbone K-Fold (Default)

```
EfficientNetV2-S × 5 folds = 5 models
Final prediction = average of 5 softmax vectors → argmax
```

**Pros:** Fast, strong baseline
**Time:** ~2–3.5 hours

### Strategy 2: Multi-Backbone K-Fold (Best Performance)

```
EfficientNetV2-S × 5 folds = 5 models
ConvNeXt-Tiny    × 5 folds = 5 models
Swin-Tiny        × 5 folds = 5 models
                              --------
                              15 models total

Final prediction = average of 15 softmax vectors → argmax
```

**Pros:** Architecture diversity reduces correlated errors
**Time:** ~6–10 hours

### Strategy 3: Multi-Backbone 3-Fold (Time-Constrained)

```
EfficientNetV2-S × 3 folds = 3 models
ConvNeXt-Tiny    × 3 folds = 3 models
Swin-Tiny        × 3 folds = 3 models
                              --------
                              9 models total
```

**Pros:** Good diversity, manageable time
**Time:** ~4–6 hours

> [!TIP]
> **Recommendation:** Start with Strategy 1 (single backbone, 5 folds) to validate your pipeline and get a CV score. Then run Strategy 2 or 3 for the final submission.

---

## 8.3 Functions to Implement

### 8.3.1 `run_cross_validation(cfg) -> dict`

**Purpose:** Train all folds and collect results.

```python
def run_cross_validation(cfg):
    """
    Run K-fold cross-validation for a single backbone.
    
    Returns:
        dict: {
            'fold_results': list of fold result dicts,
            'oof_preds': np.array (N, num_classes) — out-of-fold predictions,
            'oof_labels': np.array (N,) — true labels,
            'cv_macro_f1': float — aggregate CV Macro F1,
            'cv_per_class_f1': dict — per-class F1 from OOF
        }
    """
    # Load and prepare data
    train_df = pd.read_csv(cfg.paths.train_csv)
    
    # Optional: apply noise cleaning
    if cfg.noise.enabled and cfg.noise.action != "keep":
        train_df = create_clean_training_set(train_df, ...)
    
    # Create scene-aware folds
    folds = create_scene_aware_folds(train_df, cfg.crossval.n_folds)
    
    # Initialize OOF prediction array
    oof_preds = np.zeros((len(train_df), cfg.num_classes))
    oof_labels = train_df['appearance'].values
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_train_df = train_df.iloc[train_idx]
        fold_val_df = train_df.iloc[val_idx]
        
        # Train this fold
        result = train_fold(fold_idx, fold_train_df, fold_val_df, cfg)
        fold_results.append(result)
        
        # Collect OOF predictions
        # Load best model for this fold and predict on val set
        model = load_model(result['best_model_path'], cfg)
        val_probs = predict_probs(model, fold_val_df, cfg)
        oof_preds[val_idx] = val_probs
        
        print(f"\nFold {fold_idx} complete. Best F1: {result['best_macro_f1']:.4f}")
    
    # Compute aggregate CV score
    oof_pred_labels = oof_preds.argmax(axis=1)
    cv_macro_f1 = compute_macro_f1(oof_labels, oof_pred_labels)
    cv_per_class_f1 = compute_per_class_f1(oof_labels, oof_pred_labels)
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS ({cfg.model.backbone})")
    print(f"{'='*60}")
    print(f"CV Macro F1: {cv_macro_f1:.4f}")
    print(f"Per-class F1:")
    for cls_name, f1 in cv_per_class_f1.items():
        print(f"  {cls_name}: {f1:.4f}")
    print(f"Per-fold F1: {[r['best_macro_f1'] for r in fold_results]}")
    print(f"F1 std: {np.std([r['best_macro_f1'] for r in fold_results]):.4f}")
    
    return {
        'fold_results': fold_results,
        'oof_preds': oof_preds,
        'oof_labels': oof_labels,
        'cv_macro_f1': cv_macro_f1,
        'cv_per_class_f1': cv_per_class_f1
    }
```

---

### 8.3.2 `run_multi_backbone_cv(cfg) -> dict`

**Purpose:** Run CV with multiple backbones and combine.

```python
def run_multi_backbone_cv(cfg):
    """
    Run cross-validation with multiple backbone architectures.
    
    Returns:
        dict: {
            'backbone_results': dict of backbone_name -> cv_result,
            'ensemble_oof_preds': np.array — weighted average OOF,
            'ensemble_cv_f1': float,
            'model_paths': list of all checkpoint paths
        }
    """
    backbone_results = {}
    all_oof_preds = []
    backbone_weights = []
    all_model_paths = []
    
    for backbone_name in cfg.crossval.backbones:
        print(f"\n{'#'*60}")
        print(f"Training backbone: {backbone_name}")
        print(f"{'#'*60}\n")
        
        # Update config for this backbone
        cfg_copy = deepcopy(cfg)
        cfg_copy.model.backbone = backbone_name
        
        # Run CV
        cv_result = run_cross_validation(cfg_copy)
        backbone_results[backbone_name] = cv_result
        
        all_oof_preds.append(cv_result['oof_preds'])
        backbone_weights.append(cv_result['cv_macro_f1'])
        
        # Collect model paths
        for r in cv_result['fold_results']:
            all_model_paths.append(r['best_model_path'])
    
    # ---- Ensemble: weighted average by CV score ----
    weights = np.array(backbone_weights)
    weights = weights / weights.sum()  # Normalize
    
    ensemble_oof = np.zeros_like(all_oof_preds[0])
    for preds, w in zip(all_oof_preds, weights):
        ensemble_oof += w * preds
    
    ensemble_labels = backbone_results[cfg.crossval.backbones[0]]['oof_labels']
    ensemble_pred_labels = ensemble_oof.argmax(axis=1)
    ensemble_cv_f1 = compute_macro_f1(ensemble_labels, ensemble_pred_labels)
    
    print(f"\n{'='*60}")
    print(f"MULTI-BACKBONE ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"Individual CV F1s:")
    for name, w in zip(cfg.crossval.backbones, weights):
        print(f"  {name}: {backbone_results[name]['cv_macro_f1']:.4f} (weight: {w:.3f})")
    print(f"Ensemble CV F1: {ensemble_cv_f1:.4f}")
    print(f"Improvement over best single: +{ensemble_cv_f1 - max(backbone_weights):.4f}")
    
    return {
        'backbone_results': backbone_results,
        'ensemble_oof_preds': ensemble_oof,
        'ensemble_cv_f1': ensemble_cv_f1,
        'model_paths': all_model_paths,
        'weights': weights
    }
```

---

### 8.3.3 `predict_probs(model, df, cfg) -> np.array`

**Purpose:** Get softmax probabilities from a trained model.

```python
@torch.no_grad()
def predict_probs(model, df, cfg, transform=None):
    """
    Predict softmax probabilities for all images in df.
    
    Returns:
        np.array: (N, num_classes) softmax probabilities
    """
    if transform is None:
        transform = get_val_transform(cfg)
    
    dataset = TomJerryDataset(df, cfg.paths.image_dir, transform, is_test=True)
    loader = DataLoader(
        dataset, batch_size=cfg.training.batch_size * 2,
        shuffle=False, num_workers=cfg.num_workers
    )
    
    model.eval()
    all_probs = []
    
    for images, _ in loader:
        images = images.to(cfg.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=cfg.training.use_amp):
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_probs)
```

---

### 8.3.4 `load_model(checkpoint_path, cfg) -> nn.Module`

**Purpose:** Load a trained model from checkpoint.

```python
def load_model(checkpoint_path, cfg):
    """Load model from checkpoint."""
    model = TomJerryClassifier(cfg).to(cfg.device)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
```

---

## 8.4 Out-of-Fold (OOF) Prediction

> [!IMPORTANT]
> **OOF prediction is the gold standard for local validation.**
>
> Each training sample is predicted by the fold model that did NOT see it during training. This gives you a clean prediction for every sample, and the aggregate Macro F1 on OOF predictions is the most reliable estimate of test performance.
>
> **Use this CV F1 — not individual fold F1 — to decide your submissions.**

---

## 8.5 Ensemble Weighting Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Equal average** | Each model gets weight 1/N | Default, simple |
| **CV-weighted** | Weight proportional to fold CV F1 | When fold performance varies |
| **Rank-weighted** | Weight proportional to rank | Simple alternative to CV-weighted |
| **Optimized** | Search weights via `scipy.optimize` on OOF | Advanced, risk of overfitting weights |

**Implementation (CV-weighted):**
```python
# Weight by CV F1 score
weights = np.array([fold_f1 for fold_f1 in fold_f1_scores])
weights = weights / weights.sum()
ensemble_probs = sum(w * p for w, p in zip(weights, fold_probs))
```

---

## 8.6 Configuration Knobs

```yaml
crossval:
  n_folds: 5                    # 3 for speed, 5 for robustness
  use_multi_backbone: true
  backbones:
    - "efficientnetv2_s"
    - "convnext_tiny"
    - "swin_tiny"
  ensemble_method: "cv_weighted"  # "equal" | "cv_weighted"
```

---

## 8.7 Expected Results

| Setup | Expected CV Macro F1 | Time |
|-------|---------------------|------|
| EfficientNetV2-S × 5 folds | 0.70–0.78 | 2–3.5 hrs |
| + ConvNeXt-Tiny × 5 folds | 0.72–0.80 | +2–3.5 hrs |
| + Swin-Tiny × 5 folds | 0.73–0.82 | +2–3.5 hrs |
| Ensemble of all 15 models | 0.75–0.84 | — |

> [!NOTE]
> These are rough estimates. Actual performance depends heavily on noise level in the training data and effectiveness of noise cleaning (Phase 3).

---

## 8.8 Outputs

| Output | Description |
|--------|-------------|
| `outputs/checkpoints/fold{i}_best.pt` | Best model per fold |
| `outputs/logs/cv_summary.txt` | CV results across all folds/backbones |
| `outputs/logs/oof_predictions.npy` | OOF probability arrays |
| `outputs/logs/cv_confusion_matrix.png` | Aggregate confusion matrix on OOF |
