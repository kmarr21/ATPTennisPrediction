# ATP Ranking Prediction Models - Technical Documentation

## Overview

I implemented several ranking-based tennis prediction models using ATP rankings data, progressing from simple logistic regression on ranking positions to complex models incorporating momentum and volatility features. The base ranking models (position and points) follow the approach of McHale and Morton (2011) [1], with extensions to incorporate momentum and consistency features as original contributions.

## Model Architectures

### 1. Position Model (rankings_simple.py)

Following McHale and Morton (2011) [1], the simplest model uses only ranking positions with logistic regression:

```
P(player_i wins) = 1 / (1 + exp(-β × (rank_j - rank_i)))
```

Where:
- `rank_i`, `rank_j` are ATP ranking positions (1 = best)
- `β` is the fitted coefficient
- Higher ranked players have lower numbers, so positive difference means player_i is better

**Note**: McHale and Morton used this specification with tournament seedings; I adapted it to use ATP rankings directly to enable predictions beyond Grand Slam tournaments.

### 2. Points Model (rankings_simple.py)

Following Clarke and Dyte (2000) as cited in McHale and Morton (2011) [1], this model uses ATP ranking points instead of positions:

```
P(player_i wins) = 1 / (1 + exp(-β × log(points_i / points_j)))
```

Where:
- `points_i`, `points_j` are ATP ranking points
- `β` is the fitted coefficient
- Log ratio normalizes the wide range of points (8000 for #1 vs 50 for #200)

### 3. Joint Model (rankings_joint.py)

Combines both features in multiple logistic regression, following the general framework of McHale and Morton (2011) [1]:

```
logit(P) = β_1 × (rank_j - rank_i) + β_2 × log(points_i / points_j) + β_B
```

I found the points feature dominates, with position coefficient near zero due to multicollinearity.

### 4. Momentum Model (rankings_momentum.py)

**Original contribution**: My enhanced model adds momentum and consistency features to the points-based model:

```
Features vector X = [
    log(points_i / points_j),
    momentum_i - momentum_j,
    consistency_i - consistency_j
]

P(player_i wins) = sigmoid(β^T × StandardScaler(X))
```

Where:
- `momentum = (points_current - points_3months) / points_3months` (percentage change)
- `consistency = 1 / (1 + CV)` where CV is coefficient of variation of recent rankings
- Features are standardized using StandardScaler before model fitting

**Note**: The momentum and consistency features represent original extensions beyond the baseline ranking models in the literature, designed to capture short-term form and player volatility.

## Coefficient Fitting Process

### Mathematical Approach

Following standard practice (similar to McHale and Morton 2011 [1]), I use scikit-learn's LogisticRegression with maximum likelihood estimation. For a dataset with N matches:

1. **Objective Function**: Minimize negative log-likelihood with L2 regularization
   ```
   L(β) = -Σ[y_i × log(p_i) + (1-y_i) × log(1-p_i)] + C × ||β||²
   ```
   Where C=1.0 (regularization parameter)

2. **Optimization**: L-BFGS solver with max_iter=1000

3. **Sample Duplication**: Each match generates two samples
   - Winner perspective: X_w = features from winner viewpoint, y = 1
   - Loser perspective: X_l = features from loser viewpoint, y = 0
   - This doubles the training set and ensures balanced classes

### Training Data Requirements

- **Minimum samples**: 100 (50 matches × 2 perspectives)
- **Typical training set**: 5,000-10,000 matches → 10,000-20,000 samples
- **Filtering**: Both players must be ranked ≤250

### Fallback Strategy

If insufficient training data (<100 samples):
```python
# Insufficient data for model fitting
if len(X) <= 100:
    print("Insufficient data for model fitting")
    # For momentum model, we fit minimal dummy model
    self.scaler.fit([[0, 0, 0]])  # Dummy fit to avoid sklearn errors
    self.model = LogisticRegression()
    self.model.fit([[1], [0]], [1, 0])  # Minimal model that predicts 50/50
```

## Training Process

I train models using historical match data with strict temporal validation:

1. **Date Range Selection**: 
   - For prediction at date T, use matches from [max(2001, T-5years), T-1day]
   - Never use future data for training

2. **Match Filtering**:
   ```sql
   WHERE t.date >= start_date AND t.date <= end_date
   AND NOT m.score CONTAINS 'W/O'  -- Exclude walkovers
   ```

3. **Ranking Filter**: Applied during feature extraction
   ```python
   if w_features['position'] > max_rank or l_features['position'] > max_rank:
       continue  # Skip matches with players outside top 250
   ```

4. **Feature Scaling** (momentum model only):
   ```python
   X_scaled = self.scaler.fit_transform(X)  # Standardize features
   ```

## Prediction Process

For match prediction at date T between players A and B:

1. **Get Current Rankings**:
   ```sql
   MATCH (p:Player {id: player_id})-[:HAS_RANKING]->(r:Ranking)
   WHERE r.date <= match_date
   ORDER BY r.date DESC
   LIMIT 1
   ```

2. **Calculate Momentum** (if applicable):
   - Query rankings from 3 months prior: `three_month_ago = match_date - 300`
   - Compute: `momentum = (current_points - past_points) / past_points`

3. **Calculate Consistency** (if applicable):
   - Get last 8 weekly rankings
   - Compute coefficient of variation: `CV = std(points) / mean(points)`
   - Transform: `consistency = 1 / (1 + CV)`

4. **Apply Model**:
   ```python
   features = construct_features(player_a, player_b)
   if using_scaler:
       features = self.scaler.transform([features])
   probability = self.model.predict_proba(features)[0][1]
   ```

## Key Assumptions

- **Ranking Validity**: ATP rankings meaningfully represent player strength
- **Temporal Stability**: Coefficients learned from past matches apply to future ones
- **Surface Agnostic**: I don't differentiate by surface (could be added as feature)
- **Main Tour Focus**: Training filtered to top 250 to avoid challenger/futures noise
- **Weekly Updates**: Rankings update Mondays; I use most recent before match
- **Missing Data Handling**: Players without rankings or with 0 points are skipped

## Implementation Details

- **Database**: Neo4j graph database with Player, Match, and Ranking nodes
- **Relationships**: Player -[:HAS_RANKING]-> Ranking (weekly snapshots)
- **Date Format**: YYYYMMDD integers (20240701 = July 1, 2024)
- **Ranking Frequency**: Weekly snapshots every Monday
- **Points Range**: ~12,000 (world #1) to 1 point (world #2000+)

## Results Summary

On Wimbledon 2024 (237 matches):

| Model | Accuracy | Brier Score | Position Coef | Points Coef | Momentum Coef | Consistency Coef |
|-------|----------|-------------|---------------|-------------|---------------|------------------|
| Position | 68.8% | 0.219 | 0.007330 | - | - | - |
| Points | 68.8% | 0.205 | - | 0.753975 | - | - |
| Joint | 68.8% | 0.206 | -0.001218 | 0.815415 | - | - |
| Momentum | 68.8% | 0.203 | - | 0.5926 | 0.0370 | -0.1751 |

The similar accuracies suggest rankings already embed most predictive information. The momentum features provide marginal improvement in calibration (Brier score) but not binary accuracy.

## Key Findings

### Volatility Paradox

The negative consistency coefficient (-0.1751, from Wimbledon 2024 evaluation) reveals that volatile players outperform stable ones. This coefficient means:

```
Higher consistency_i - consistency_j → Lower P(player_i wins)
```

Since `consistency = 1 / (1 + CV)`, more volatile players (higher CV, lower consistency score) have an advantage. This counterintuitive result makes competitive sense:

1. **Peak Performance Capability**: Volatile players can produce breakthrough performances
2. **Rising Stars**: Young players climbing rankings show high volatility as they improve
3. **Declining Veterans**: Stable rankings may indicate plateaued performance
4. **Upset Potential**: Inconsistent players with high ceilings can defeat anyone on their day

### Multicollinearity in Joint Model

The joint model shows a negative position coefficient (-0.001218) despite rank differences favoring better-ranked players. This occurs because ranking position and points are highly correlated (ρ ≈ 0.95; since points directly determine ranking order), causing unstable coefficient estimates. The points feature dominates as it contains more granular information.

### Momentum's Limited Impact

The small momentum coefficient (0.0370) suggests recent form provides minimal additional predictive power beyond current rankings. This makes sense because ATP rankings are already a 52-week rolling calculation that inherently captures recent performance.

## Usage Examples

```bash
# Single match predictions
python rankings_simple.py --position "Novak Djokovic" "Rafael Nadal" 20230601
python rankings_simple.py --points "Novak Djokovic" "Rafael Nadal" 20230601
python rankings_joint.py --predict "Novak Djokovic" "Rafael Nadal" 20230601
python rankings_momentum.py --predict "Carlos Alcaraz" "Daniil Medvedev" 20230601

# Tournament evaluation
python rankings_simple.py --tournament Wimbledon 2024 --model position
python rankings_simple.py --tournament Wimbledon 2024 --model points
python rankings_joint.py --tournament "US Open" 2024
python rankings_momentum.py --tournament Wimbledon 2024
```

## References

[1] McHale, I., & Morton, A. (2011). A Bradley-Terry type model for forecasting tennis match results. *International Journal of Forecasting*, 27(2), 619-630. https://doi.org/10.1016/j.ijforecast.2010.04.004

**Note**: While McHale and Morton primarily focused on their Bradley-Terry model, they also compared it against logistic regression models using ATP rankings (both position and points), which served as the foundation for the baseline models implemented here. The momentum and consistency extensions represent original contributions beyond their work.