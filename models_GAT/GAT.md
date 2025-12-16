# Tennis Matchup GAT Model - Technical Documentation

## Overview

I implemented a GAT-inspired (Graph Attention Network) architecture for tennis match prediction, adapting attention mechanisms to model the asymmetric nature of tennis matchups. While traditional GATs operate on graph-structured data with nodes and edges, my architecture borrows the core multi-head attention mechanism to model how one player's skills interact with another's: specifically, how serving ability attacks returning ability.

The model is "GAT-inspired" rather than a pure GAT because tennis matchups aren't naturally graph-structured. Instead of "node attends to neighbor nodes," I use cross-attention where "P1's serve attends to P2's return." Same mathematical machinery, different application.

### Key Innovations

1. **Matchup-Centric Design**: Separate attention modules for 1st serve, 2nd serve, and break point scenarios
2. **Siamese Encoders**: Shared weights ensure P1 and P2 are treated symmetrically
3. **Style Factor Attention**: Special handling for NMF-derived playing style factors with both attention and bilinear interactions
4. **Cross-Attention for Asymmetry**: Models the fundamental tennis asymmetry (serving vs. returning)

## Model Architecture

### High-Level Flow

```
1. INPUT (270 features)
        ↓
2. ENCODE (6 parallel modules)
   - 4 Siamese: 1st Serve, 2nd Serve, Break Points, Style Factors
   - 2 Regular: Ratings, Form
        ↓
3. CROSS-ATTENTION (4 parallel operations)
   - 1st Serve: P1 serve vs P2 return
   - 2nd Serve: P1 serve vs P2 return
   - Break Points: P1 save vs P2 convert
   - Style Factors: P1 style vs P2 style + bilinear
   - (Rating and Form skip this step)
        ↓
4. CONCAT (stack all 6 outputs)
        ↓
5. FUSION MLP (2-layer neural network)
        ↓
6. OUTPUT (sigmoid → probability)
```

### Component Details

#### Encoders

Each encoder transforms raw features into a dense representation of size `hidden_dim` (64, 96, or 128).

**Siamese Encoders** (1st Serve, 2nd Serve, BP, Style):
- Same `nn.Linear` + `nn.ReLU` weights process both P1 and P2
- Ensures symmetry: identical stats → identical encodings
- Prevents the model from learning "P1 position bias"

**Regular Encoders** (Rating, Form):
- Process combined P1+P2 features together
- No need for Siamese since input already contains both players

#### Cross-Attention Mechanism

For each matchup module, cross-attention models how one player's skill interacts with another's:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- **Query (Q)**: From attacker encoding (e.g., P1's serve representation)
- **Key (K)**: From defender encoding (e.g., P2's return representation)
- **Value (V)**: From defender encoding

The attention mechanism learns: "Given P1's specific serve style, which parts of P2's return defense are most relevant?"

**Multi-Head Attention**: Each cross-attention operation uses 2-4 heads internally. Each head can learn different aspects:
- Head 1 might focus on: ace rate vs. first-return depth
- Head 2 might focus on: placement accuracy vs. movement speed
- Heads learn automatically through backpropagation

#### Three Matchup Types

| Module | P1 (Attacker) | P2 (Defender) | What It Models |
|--------|---------------|---------------|----------------|
| 1st Serve | Ace%, 1st serve%, 1st won% | 1st return won% | Power serve vs. return quality |
| 2nd Serve | 2nd serve won%, DF% | 2nd return won% | Consistency under pressure |
| Break Points | BP saved%, hold% | BP converted%, break% | Clutch performance |

#### Style Factor Attention

The Style Factor module has additional complexity because NMF-derived factors represent abstract playing styles:

1. **Cross-Attention**: Standard Q/K/V attention between P1 and P2 style factors
2. **Bilinear Interaction**: Direct multiplication of P1's factors with P2's factors
3. **Fusion**: Combines both attention and bilinear outputs

This captures "how do these two playing styles clash" more directly than pure attention.

#### Fusion Layers

**Concatenation (Step 4)**:
Simply stacks all module outputs:
```
[1st_serve | 2nd_serve | BP | style | rating | form]
```
If `hidden_dim=64` and 6 modules: 6 × 64 = 384 dimensions

**Fusion MLP (Step 5)**:
- `nn.Linear(6 * hidden_dim, hidden_dim)` — compress
- `nn.LayerNorm(hidden_dim)` — normalize for stability
- `nn.ReLU()` — nonlinearity
- `nn.Dropout(p)` — regularization
- `nn.Linear(hidden_dim, hidden_dim)` — second layer
- Residual connection — add input back to output

**Output Layer (Step 6)**:
- `nn.Linear(hidden_dim, 1)` — single logit
- `torch.sigmoid()` — probability in [0, 1]

## Feature Engineering

### Feature Counts

| Category | Per Player | Total |
|----------|------------|-------|
| P1 Features | 116 | 116 |
| P2 Features | 116 | 116 |
| Differential (P1 - P2) | 38 | 38 |
| **Total** | — | **270** |

### Feature Groups (Per Player: 116)

| Group | Count | Description |
|-------|-------|-------------|
| Basic | 3 | age, height, handedness |
| 52-week stats | 31 | serve%, return%, BP%, overall performance |
| 4-week form | 27 | recent performance (same categories) |
| 8-week form | 27 | medium-term form |
| Surface stats | 3 | win%, serve%, return% on current surface |
| Form deltas | 2 | 4w - 52w, 8w - 52w (form trajectory) |
| Rankings | 2 | ATP rank, ranking points |
| Relative stats | 5 | performance vs. peer group (top 10/20/50/100/200) |
| Elo ratings | 2 | overall + surface-specific |
| Glicko-2 | 4 | rating + RD (overall + surface) |
| Style factors | 10 | 5 Elo-NMF + 5 Glicko-NMF |

### Differential Features (38)

Explicit P1 − P2 comparisons for key statistics:
- Ranking difference
- Elo difference (overall + surface)
- Glicko difference
- Win percentage differences (52w, 4w, 8w)
- Serve/return point differences
- And more...

The model doesn't have to "learn" that subtraction matters — I hand it the comparison directly.

## Data Preparation

### Prerequisites

Before running the GAT, the following must exist in Neo4j:

1. **Match nodes**: All ATP matches with outcomes
2. **Player nodes**: With basic attributes (height, handedness, birth date)
3. **PlayerStats nodes**: 52-week rolling statistics
4. **Elo nodes**: From `import_elo_nodes.py`
5. **Glicko2 nodes**: From `import_glicko2_nodes.py`
6. **NMF style factors**: From `import_elo_nmf_nodes.py` or `import_glicko2_nmf_nodes.py`
7. **Rankings**: Historical ATP rankings

### Data Preparation Script

**File**: `prepare_gat_data.py`

```bash
# Generate training data
python prepare_gat_data.py
```

This script:
1. Queries Neo4j for all Masters + Grand Slam matches (2008-2024)
2. For each match, retrieves features using only data available BEFORE match date
3. Applies position randomization (50/50 P1/P2 assignment)
4. Splits into train/val/test by year
5. Saves to `gat_cache/gat_torch_data_v2.pt`

**Output Structure**:
```python
{
    'train': {'features': Tensor, 'labels': Tensor, 'metadata': DataFrame},
    'val': {'features': Tensor, 'labels': Tensor, 'metadata': DataFrame},
    'test': {'features': Tensor, 'labels': Tensor, 'metadata': DataFrame}
}
```

### Temporal Splits

| Split | Years | Purpose |
|-------|-------|---------|
| Train | 2008-2015 | Learn patterns |
| Validation | 2016-2018 | Tune hyperparameters, early stopping |
| Test | 2019-2024 | Final evaluation |

**Critical**: All queries use `date < match_date` (strictly before) to prevent data leakage.

### Tournament Filter

Only Masters 1000 and Grand Slams are included:
- Highest data quality and completeness
- Fewer walkovers/retirements
- Highest stakes matches
- ~2,500 matches in test set

## Training Methodology

### Loss Function

Binary Cross-Entropy (BCE):

$$\mathcal{L} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

Where:
- $y$ = actual outcome (1 if P1 wins, 0 otherwise)
- $\hat{y}$ = predicted probability

I use `nn.BCEWithLogitsLoss` which combines sigmoid + BCE for numerical stability.

### Optimizer

**AdamW** (Adam with decoupled weight decay):
- Adapts learning rate per-parameter
- Uses momentum for smoother updates
- Proper L2 regularization via weight decay

### Learning Rate Scheduler

**ReduceLROnPlateau**:
- Monitors validation AUC
- Reduces LR by 0.5x after 5 epochs without improvement
- Minimum LR: 1e-6

### Early Stopping

- **Criterion**: Validation AUC (maximize)
- **Patience**: 15 epochs without improvement
- **Best checkpoint saved** when validation AUC improves

Why patience=15? Tennis predictions are noisy. A model might dip for a few epochs then recover. 15 epochs gives it time to escape local minima.

### Gradient Clipping

`max_norm=1.0` : prevents exploding gradients during backpropagation.

### Training Loop

```python
for epoch in range(max_epochs):
    # Forward pass
    predictions = model(features)
    loss = criterion(predictions, labels)
    
    # Backward pass
    loss.backward()           # Compute gradients
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()          # Update weights
    optimizer.zero_grad()     # Reset gradients
    
    # Validation
    val_auc = evaluate(model, val_loader)
    scheduler.step(val_auc)
    early_stopping(val_auc)
    
    if early_stopping.early_stop:
        break
```

## Hyperparameters

### Swept Parameters

| Parameter | Values Tested | Description |
|-----------|---------------|-------------|
| `hidden_dim` | 64, 96, 128 | Size of internal representations |
| `num_heads` | 2, 4 | Attention heads per cross-attention module |
| `dropout` | 0.15, 0.2, 0.3 | Fraction of neurons zeroed during training |
| `learning_rate` | 0.0005, 0.001 | Step size for weight updates |

### Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` | 64 | Balance between gradient stability and speed |
| `weight_decay` | 1e-4 | Gentle L2 regularization |
| `max_epochs` | 100 | Usually stopped early by patience |
| `patience` | 15 | Allow recovery from temporary dips |
| `gradient_clip` | 1.0 | Standard for attention models |

### Model Size

100-400K parameters depending on `hidden_dim`. Decently small by deep learning standards: helps prevent overfitting on ~2,500 test matches.

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | % correct predictions | Raw correctness |
| **AUC** | Area under ROC curve | Ranking/discrimination quality |
| **Brier Score** | $(p - o)^2$ | Calibration (lower = better) |
| **Log Loss** | $-[o \log p + (1-o) \log(1-p)]$ | Confidence-weighted calibration |

### Evaluation Breakdowns

- **By Surface**: Hard, Clay, Grass
- **By Round**: R128 through Final
- **By Tournament**: Individual tournament metrics

## Usage

### Order of Operations

1. **Ensure prerequisites exist** (Elo nodes, Glicko nodes, NMF factors, etc.)
2. **Prepare data**: `python prepare_gat_data.py`
3. **Train model**: `python train_tennis_gat.py`
4. **Evaluate**: Results saved to `gat_outputs/`

### Single Training Run

```bash
python train_tennis_gat.py
```

With custom parameters:
```bash
python train_tennis_gat.py --hidden_dim 128 --num_heads 4 --dropout 0.3 --lr 0.0005
```

### Hyperparameter Sweep

```bash
python train_tennis_gat.py --sweep
```

Tests multiple configurations and saves results to `gat_outputs/gat_sweep_results_*.csv`.

### Output Files

| File | Description |
|------|-------------|
| `gat_outputs/gat_model_*.pth` | Saved model weights |
| `gat_outputs/gat_results_*.csv` | Per-match predictions with metadata |
| `gat_outputs/gat_sweep_results_*.csv` | Hyperparameter sweep summary |
| `gat_outputs/training_history_*.json` | Loss/metrics per epoch |

## PyTorch Implementation

### Core Classes Used

| Component | PyTorch Class |
|-----------|---------------|
| Encoders | `nn.Linear` + `nn.ReLU` |
| Cross-Attention | `nn.MultiheadAttention` |
| Layer Normalization | `nn.LayerNorm` |
| Dropout | `nn.Dropout` |
| Loss | `nn.BCEWithLogitsLoss` |
| Optimizer | `torch.optim.AdamW` |
| Scheduler | `optim.lr_scheduler.ReduceLROnPlateau` |

### Key Implementation Notes

**nn.MultiheadAttention** handles Q/K/V projections internally:
```python
self.cross_attn = nn.MultiheadAttention(
    embed_dim=hidden_dim,
    num_heads=num_heads,
    dropout=dropout,
    batch_first=True
)

# Usage
output, attn_weights = self.cross_attn(
    query=p1_encoded,    # Attacker
    key=p2_encoded,      # Defender
    value=p2_encoded     # Defender
)
```

**Residual connections** help gradient flow:
```python
x = self.layer(x) + x  # Add input back to output
```

**LayerNorm** stabilizes training:
```python
x = self.norm(x)  # Normalize to mean=0, std=1
```

## File Structure

```
project/
├── prepare_gat_data_v2.py   # Data preparation from Neo4j
├── tennis_gat_model_v4.py   # Model architecture definition
├── train_tennis_gat_v5.py   # Training script with sweep option
├── feature_index_dict_v2.py # Feature name → index mapping
├── gat_cache/
│   └── gat_torch_data_v2.pt # Prepared training data
└── gat_outputs/
    ├── gat_model_*.pth      # Saved models
    ├── gat_results_*.csv    # Predictions
    └── training_history_*.json
```

## Dependencies

```python
torch >= 1.9
torch_geometric >= 2.0  # For attention patterns
numpy >= 1.19
pandas >= 1.2
scikit-learn >= 0.24
neo4j >= 4.0
tqdm >= 4.60
```

## Computational Performance

- **Data preparation**: ~10-15 minutes (Neo4j queries)
- **Training time**: 10-20 minutes per configuration (CPU)
- **Full hyperparameter sweep**: 2-4 hours
- **Inference**: <10ms per match

## Future Improvements

1. **Graph structure**: Add player similarity graph for true GNN message passing
2. **Temporal attention**: Attend over recent match history, not just aggregated stats
3. **Head-to-head module**: Dedicated attention for historical matchup patterns
4. **Ensemble**: Combine GAT predictions with baseline models (Elo, Glicko)
5. **Uncertainty quantification**: Output prediction intervals, not just point estimates

## References

- Veličković, P. et al. (2018). "Graph Attention Networks." ICLR 2018.
- Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS 2017.
- Sackmann, J. Tennis Abstract. https://www.tennisabstract.com
- Glickman, M.E. (2012). "Example of the Glicko-2 system."

## Author Notes

The GAT-inspired architecture's key insight is that tennis matchups are fundamentally asymmetric: serving and returning are different skills that interact in specific ways. Traditional rating systems (Elo, Glicko) treat matches as symmetric exchanges. By using cross-attention to model "P1's serve attacking P2's return," the hope is that the model can learn nuanced matchup dynamics that simple rating differences miss. Of course, the original plan for this model was to use point-by-point data to break down style (particularly serving and returning style) even further. However, given the time constraints, this current model version only uses overall matchstats like serving and returning percentages instead (which is admittedly less granular).

I should also note that the style factor attention module was particularly important. Since NMF-derived factors from baseline rating models (Elo+NMF, Glicko+NMF) showed significant improvements in those models, it was hoped that giving style factors their own dedicated attention mechanism with bilinear interactions would further help the GAT capture similar matchup-specific patterns.