# Phase 5 — Class Imbalance Handling & Loss Functions

```
Module: src/losses.py
Priority: CRITICAL — directly impacts Macro F1 on minority classes
GPU Required: No (definition only)
Estimated Time: Implementation only
Dependencies: Phase 0 (config, class weights)
```

---

## 5.1 Problem Statement

| Class | Count | % of Train | Imbalance Ratio vs Class 1 |
|-------|-------|-----------|---------------------------|
| 0 (Neither) | 368 | 13.7% | 3.4× fewer |
| 1 (Tom) | 1,252 | 46.7% | 1.0× (majority) |
| 2 (Jerry) | 841 | 31.4% | 1.5× fewer |
| 3 (Both) | 219 | 8.2% | **5.7× fewer** |

> [!WARNING]
> With standard CrossEntropyLoss, the model will overwhelmingly predict Class 1 (Tom), achieving high accuracy (~47%) but terrible Macro F1 because Classes 0 and 3 will be near-zero.

**Three complementary strategies:**
1. **Loss function** — Focal Loss or Weighted CE
2. **Sampling** — WeightedRandomSampler
3. **Label smoothing** — Reduces overconfidence

---

## 5.2 Functions to Implement

### 5.2.1 `compute_class_weights(train_df) -> torch.Tensor`

**Purpose:** Compute inverse-frequency class weights.

**Implementation:**
```python
def compute_class_weights(train_df, num_classes=4):
    """
    Compute class weights inversely proportional to frequency.
    Normalized so weights sum to num_classes.
    
    For our data:
      Class 0: 2680/368 = 7.28 → normalized
      Class 1: 2680/1252 = 2.14 → normalized
      Class 2: 2680/841 = 3.19 → normalized
      Class 3: 2680/219 = 12.24 → normalized
    """
    counts = train_df['appearance'].value_counts().sort_index().values
    total = counts.sum()
    weights = total / (num_classes * counts)
    # Normalize so weights sum to num_classes
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)
```

**Expected weights (approximate):**
```
Class 0 (Neither): ~1.18
Class 1 (Tom):     ~0.35
Class 2 (Jerry):   ~0.52
Class 3 (Both):    ~1.99
```

---

### 5.2.2 `FocalLoss(nn.Module)`

**Purpose:** Down-weight easy (well-classified) examples, focus on hard ones.

**Mathematical Formulation:**
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
Where:
- `p_t` = probability of the correct class
- `γ` (gamma) = focusing parameter (default: 2.0) — higher = more focus on hard examples
- `α_t` = class weight (from `compute_class_weights`)

**Implementation:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        """
        Args:
            alpha: Class weights tensor [num_classes]. None = uniform.
            gamma: Focusing parameter. 0 = standard CE. 2 = recommended.
            label_smoothing: Smooth targets (e.g., 0.1)
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits (B, C)
            targets: Class indices (B,)
        """
        # Apply label smoothing to targets
        num_classes = inputs.size(-1)
        if self.label_smoothing > 0:
            # Convert to soft targets
            smooth_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Gather probabilities for true class
        if self.label_smoothing > 0:
            # Use soft CE
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
            pt = (smooth_targets * probs).sum(dim=-1)
        else:
            ce_loss = F.nll_loss(log_probs, targets, reduction='none')
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal modulation
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)
            alpha_t = alpha_t.gather(0, targets)
            focal_weight = focal_weight * alpha_t
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

---

### 5.2.3 `WeightedCrossEntropyLoss` (wrapper)

**Purpose:** Standard CE with class weights and label smoothing.

```python
def get_weighted_ce_loss(class_weights, label_smoothing=0.1):
    """
    Wrapper around nn.CrossEntropyLoss with class weights.
    
    Args:
        class_weights: Tensor of shape [num_classes]
        label_smoothing: Float, typically 0.1
    """
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )
```

---

### 5.2.4 `get_loss_function(cfg, class_weights) -> nn.Module`

**Purpose:** Factory function to create the configured loss.

```python
def get_loss_function(cfg, class_weights):
    """
    Create loss function based on config.
    
    Args:
        cfg: Config object
        class_weights: Tensor from compute_class_weights()
    """
    if cfg.loss.type == "focal":
        alpha = class_weights if cfg.loss.use_class_weights else None
        return FocalLoss(
            alpha=alpha,
            gamma=cfg.loss.focal_gamma,
            label_smoothing=cfg.training.label_smoothing
        )
    elif cfg.loss.type == "cross_entropy":
        weight = class_weights if cfg.loss.use_class_weights else None
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=cfg.training.label_smoothing
        )
    else:
        raise ValueError(f"Unknown loss type: {cfg.loss.type}")
```

---

### 5.2.5 `get_weighted_sampler(train_df) -> WeightedRandomSampler`

**Purpose:** Create a sampler that over-samples minority classes.

```python
def get_weighted_sampler(train_df):
    """
    WeightedRandomSampler ensures each batch has roughly equal class representation.
    
    This effectively up-samples minority classes and down-samples majority classes.
    """
    class_counts = train_df['appearance'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    
    # Assign weight to each sample based on its class
    sample_weights = np.array([class_weights[label] for label in train_df['appearance']])
    sample_weights = torch.FloatTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True  # Must be True to over-sample minority
    )
    
    return sampler
```

---

## 5.3 Strategy Combinations

| Strategy | Config Settings | When to Use |
|----------|----------------|-------------|
| **Focal + Sampler** (recommended) | `loss.type="focal"`, `sampler.use_weighted_sampler=true` | Default — handles imbalance at both loss and sampling level |
| **Focal only** | `loss.type="focal"`, `sampler.use_weighted_sampler=false` | If sampler causes instability |
| **Weighted CE + Sampler** | `loss.type="cross_entropy"`, `loss.use_class_weights=true`, `sampler.use_weighted_sampler=true` | If Focal Loss underperforms |
| **Weighted CE only** | `loss.type="cross_entropy"`, `loss.use_class_weights=true`, `sampler.use_weighted_sampler=false` | Simplest approach |

> [!TIP]
> **Start with Focal Loss (γ=2.0) + WeightedRandomSampler.** This is the most aggressive imbalance handling and usually gives the best Macro F1. Dial back if you see training instability.

---

## 5.4 Label Smoothing Details

| Parameter | Value | Effect |
|-----------|-------|--------|
| `label_smoothing=0` | Hard targets: `[0, 0, 1, 0]` | Overconfident predictions |
| `label_smoothing=0.1` | Soft targets: `[0.033, 0.033, 0.9, 0.033]` | Reduces overconfidence, improves generalization |
| `label_smoothing=0.2` | Soft targets: `[0.067, 0.067, 0.8, 0.067]` | Too aggressive for 4-class — don't go this high |

> [!NOTE]
> Label smoothing is especially helpful here because the training labels are noisy. A hard target on a mislabeled sample is maximally wrong; a soft target is less wrong.

---

## 5.5 Configuration Knobs

```yaml
loss:
  type: "focal"               # "focal" | "cross_entropy"
  focal_gamma: 2.0            # 0 = CE, 2 = recommended, 5 = extreme focus on hard
  focal_alpha: null            # null = auto-compute from class distribution
  use_class_weights: true      # Apply inverse-frequency weights

sampler:
  use_weighted_sampler: true   # Over-sample minority classes

training:
  label_smoothing: 0.1        # Soft targets
```

---

## 5.6 Verification

After implementing, verify by:
1. **Forward pass test:** Create a dummy batch, pass through each loss function, check gradients flow
2. **Class weight sanity:** Print computed weights, confirm minority classes (0, 3) have highest weights
3. **Sampler verification:** Run 1 epoch with sampler, count class frequencies in batches — should be roughly equal
