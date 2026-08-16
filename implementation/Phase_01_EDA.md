# Phase 1 — Exploratory Data Analysis (EDA)

```
Module: src/eda.py
Priority: HIGH — must run first to understand data before any modeling
GPU Required: No (CPU-only)
Estimated Time: 5–10 minutes
```

---

## 1.1 Objective

Thoroughly understand the dataset characteristics before building any model. The EDA module must produce visual reports and statistics that inform decisions in all subsequent phases (splitting strategy, augmentation intensity, loss function choice, etc.).

---

## 1.2 Functions to Implement

### 1.2.1 `plot_class_distribution(train_csv, save_path)`

**Purpose:** Visualize the severe class imbalance.

**Implementation Details:**
- Read `train.csv` with pandas
- Compute value counts for `appearance` column
- Generate a horizontal bar chart with:
  - Bars colored by class (use a consistent 4-color palette throughout project)
  - Count labels on each bar
  - Percentage labels alongside counts
  - Title: "Training Set Class Distribution"
  - X-axis: count, Y-axis: class names ("Neither", "Tom", "Jerry", "Both")
- Save to `outputs/eda/class_distribution.png`

**Expected Output:**
```
Class 1 (Tom):     ████████████████████████  1,252 (46.7%)
Class 2 (Jerry):   ████████████████          841   (31.4%)
Class 0 (Neither): ███████                   368   (13.7%)
Class 3 (Both):    ████                      219   ( 8.2%)
```

> [!IMPORTANT]
> This distribution confirms we need aggressive imbalance handling. The ~5.7:1 ratio between Class 1 and Class 3 means a naive model will almost never predict Class 3.

---

### 1.2.2 `plot_sample_grid(train_csv, image_dir, save_path, n_samples=16)`

**Purpose:** Visual sanity check — see what each class actually looks like.

**Implementation Details:**
- For each of the 4 classes, randomly sample `n_samples // 4` images
- Create a 4×4 grid (or 4 rows × `n` cols)
- Row labels: class names
- Use `matplotlib` subplots with `figsize=(16, 12)`
- Show image with filename as subplot title
- Save to `outputs/eda/sample_grid.png`

**Key Observations to Document:**
- Are "Neither" frames clearly distinct (title cards, backgrounds)?
- Do Tom-only and Jerry-only frames have clear visual patterns?
- Are "Both" frames always action scenes?
- Any obviously mislabeled examples visible?

---

### 1.2.3 `analyze_image_dimensions(image_dir, train_csv, test_csv, save_path)`

**Purpose:** Check if images have consistent dimensions or need resizing.

**Implementation Details:**
- Iterate over all images (both train + test), read dimensions with PIL
- Compute and log:
  - `(width, height)` distribution — histogram
  - Aspect ratio distribution — histogram
  - Min, max, mean, median, std for width, height, aspect ratio
  - Any outliers (extreme dimensions)
- Generate:
  - Scatter plot of `width` vs `height` (color-coded train/test)
  - Histogram of aspect ratios
- Save to `outputs/eda/image_dimensions.png`

**Impact on Pipeline:**
- Determines `image_size` in config (224 vs 256 vs 384)
- Identifies if we need padding vs cropping vs simple resize
- Aspect ratio > 1.5 or < 0.67 → consider center-crop strategy

---

### 1.2.4 `detect_near_duplicates(image_dir, csv_path, save_path, threshold=10)`

**Purpose:** Detect near-duplicate frames (critical for preventing data leakage in splits).

**Implementation Details:**
- Use **perceptual hashing** (`imagehash` library) — `phash` (DCT-based, 64-bit)
- Compute hash for every image in `train.csv`
- Build a pairwise Hamming distance matrix (or use efficient nearest-neighbor search)
- Flag pairs with Hamming distance ≤ `threshold` (default: 10 bits out of 64)
- **Group connected components:** If A≈B and B≈C, group {A, B, C} together
  - Use `scipy.sparse` graph + connected components, OR `networkx`

**Output:**
- `outputs/eda/duplicate_groups.json`:
  ```json
  {
    "group_0": ["img_abc.jpg", "img_def.jpg", "img_ghi.jpg"],
    "group_1": ["img_xyz.jpg", "img_uvw.jpg"],
    ...
  }
  ```
- `outputs/eda/duplicate_stats.txt`:
  - Total unique groups formed
  - Average group size
  - Largest group
  - Distribution of group sizes (histogram)
- `outputs/eda/duplicate_examples.png`:
  - Show 3–5 example groups side-by-side to visually confirm they're near-duplicates

**Performance Consideration:**
> [!TIP]
> For 2,680 train images, pairwise comparison is O(n²) ≈ 3.6M comparisons. With 64-bit hashes stored as integers, this takes < 30 seconds. If the full set (5,478) is used, still feasible in < 2 minutes.

**Why This Matters:**
> [!WARNING]
> Frames were extracted at 1 FPS. Adjacent seconds from the same scene will be visually near-identical. If one is in train and its near-duplicate is in validation, you get **inflated validation scores** that don't reflect true generalization. Phase 2 uses these groups to do scene-aware splitting.

---

### 1.2.5 `check_train_test_overlap(train_csv, test_csv, image_dir, threshold=10)`

**Purpose:** Verify there's no leakage between train and test sets.

**Implementation Details:**
- Compute phash for all test images
- Compare each test hash against all train hashes
- Flag any test images that are within `threshold` Hamming distance of a train image
- Report count and list of overlapping pairs

**Expected Result:** Should be 0 (filenames are anonymized, but worth verifying).

---

### 1.2.6 `compute_channel_statistics(image_dir, csv_path)`

**Purpose:** Compute per-channel mean/std for normalization.

**Implementation Details:**
- Sample ~500 images (or all if fast enough)
- Resize to target `image_size` (224)
- Compute running mean and std per channel (R, G, B)
- Compare to ImageNet defaults: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
- Decide whether to use dataset-specific or ImageNet normalization

**Output:**
```
Dataset mean: [R, G, B]
Dataset std:  [R, G, B]
ImageNet mean: [0.485, 0.456, 0.406]
ImageNet std:  [0.229, 0.224, 0.225]
Recommendation: Use ImageNet stats (transfer learning from pretrained models)
```

> [!NOTE]
> For transfer learning with ImageNet-pretrained models, it's almost always better to use ImageNet normalization stats, even if the dataset stats differ. The pretrained features expect ImageNet-normalized inputs.

---

## 1.3 Main EDA Runner

```python
def run_eda(cfg):
    """
    Run all EDA analyses and save outputs.
    Called as: python -m src.eda
    """
    os.makedirs(cfg.paths.output_dir + "/eda", exist_ok=True)
    
    # 1. Class distribution
    plot_class_distribution(...)
    
    # 2. Sample grid
    plot_sample_grid(...)
    
    # 3. Image dimensions
    analyze_image_dimensions(...)
    
    # 4. Near-duplicate detection
    groups = detect_near_duplicates(...)
    
    # 5. Train-test overlap check
    check_train_test_overlap(...)
    
    # 6. Channel statistics
    compute_channel_statistics(...)
    
    print("EDA complete. Review outputs in outputs/eda/")
```

---

## 1.4 Outputs Checklist

| Output File | Description |
|-------------|-------------|
| `outputs/eda/class_distribution.png` | Bar chart of class counts |
| `outputs/eda/sample_grid.png` | 4×4 image grid per class |
| `outputs/eda/image_dimensions.png` | Width/height scatter + aspect ratio histogram |
| `outputs/eda/duplicate_groups.json` | Near-duplicate groups for Phase 2 splitting |
| `outputs/eda/duplicate_stats.txt` | Summary statistics of grouping |
| `outputs/eda/duplicate_examples.png` | Visual confirmation of near-duplicate groups |
| `outputs/eda/channel_stats.txt` | Per-channel mean/std |

---

## 1.5 Manual Review Checkpoint

> [!CAUTION]
> **STOP after Phase 1.** Before proceeding to Phase 2, manually review:
> 1. The sample grid — do labels look correct? Flag any obvious mislabels.
> 2. The duplicate groups — are they actually visually similar? Adjust `threshold` if too aggressive/loose.
> 3. Image dimensions — decide on `image_size` (224 recommended for GPU budget).
