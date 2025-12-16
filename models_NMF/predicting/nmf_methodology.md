# Non-negative Matrix Factorization for Tennis Match Prediction Enhancement

## Overview

This document describes the implementation of Non-negative Matrix Factorization (NMF) to extract latent style factors from historical head-to-head tennis match data, subsequently used to enhance traditional rating-based prediction models (ELO and Glicko-2). The approach is inspired by Jeff Sackmann's Tennis Abstract article on [an inside-out attempt to classify playing styles](https://www.tennisabstract.com/blog/2025/04/29/an-inside-out-attempt-to-classify-playing-styles/) in professional tennis (though he only did this with Elo).

## Methodology

### 1. Data Preparation

#### Tournament Selection
The analysis encompasses 40 tournaments spanning 2008-2024, including:
- **Grand Slams**: Australian Open, Roland Garros, Wimbledon, US Open
- **Masters 1000**: Indian Wells, Miami, Rome, Cincinnati
- **Years analyzed**: 2008, 2012, 2016, 2019, 2024 (skipped over 2020 due to COVID19 pandemic!)

The tournament list is maintained in `tournaments_to_analyze.csv` and can be modified to include different events or time periods.

#### Temporal Boundaries
For each tournament, three critical dates ensure temporal integrity:
- **Tournament date**: The start date of the main draw
- **Rating cutoff date**: Monday before the tournament (ensures ratings reflect pre-tournament state)
- **H2H analysis window**: 52 weeks preceding the rating cutoff date

### 2. Player Selection

For each tournament, the top 100 players are selected based on tour-level match wins during the 52-week analysis window. This threshold was chosen to balance:
- **Statistical significance**: Sufficient head-to-head matches for meaningful patterns
- **Computational efficiency**: Manageable matrix dimensions (100×100)
- **Coverage**: Captures all likely tournament contenders while avoiding sparse data from lower-ranked players

Tour-level events include Grand Slams, Masters 1000, ATP 500, and ATP 250 tournaments, explicitly excluding Challenger and ITF events.

### 3. Deviation Matrix Construction

For each player pair (i, j), the deviation from expected performance is calculated:

```
deviation[i,j] = actual_win_rate[i,j] - expected_win_rate[i,j]
```

Where:
- **Actual win rate**: Historical H2H record during the analysis window
- **Expected win rate**: Calculated using either ELO or Glicko-2 formulas

#### ELO Expected Probability
```
P(A beats B) = 1 / (1 + 10^((R_B - R_A) / 400))
```

#### Glicko-2 Expected Probability
Following Glickman's formula with combined rating deviation:
```
combined_RD = sqrt(RD_A² + RD_B²)
g(RD) = 1 / sqrt(1 + 3 * (combined_RD/π)² / 400)
P(A beats B) = 1 / (1 + 10^(-g * (R_A - R_B) / 400))
```

### 4. Non-negative Matrix Factorization

NMF decomposes the deviation matrix into latent factors representing playing style matchups. The implementation follows the standard NMF formulation:

```
V ≈ W × H
```

Where:
- **V**: The non-negative shifted deviation matrix (n_players × n_players)
- **W**: Player-factor matrix (n_players × k_factors)
- **H**: Factor-opponent matrix (k_factors × n_players)

#### Implementation Details
- **Shift to non-negative**: `V = deviation_matrix - min(deviation_matrix)`
- **Weighting**: Square root of match counts to avoid over-weighting frequent matchups
- **Algorithm**: Multiplicative update rules with Frobenius norm minimization
- **Initialization**: Non-negative Double Singular Value Decomposition (NNDSVD)
- **Parameters**: 
  - Components: 5 (empirically determined balance between expressiveness and overfitting)
  - Maximum iterations: 500
  - Random state: 42 (for reproducibility)

The choice of 5 components was based on:
1. Reconstruction error analysis showing diminishing returns beyond 5 factors
2. Interpretability considerations (avoiding over-parameterization)
3. Alignment with intuitive tennis playing style dimensions

For mathematical foundations of NMF, see [Lee & Seung (1999)](https://www.nature.com/articles/44565) and the [scikit-learn NMF documentation](https://scikit-learn.org/stable/modules/decomposition.html#nmf).

### 5. Style Factor Integration

Style factors modify base rating predictions through weighted adjustments:

```python
style_adjustment = mean(|factors_A - factors_B|) * (max_adjustment / 2.5)
adjusted_probability = clip(base_probability + style_adjustment, 0.01, 0.99)
```

The normalization factor of 2.5 was empirically determined from the typical range of factor differences.

## Model Integration

### ELO Implementation
The ELO model uses Tennis Abstract's surface-specific approach:
- **Overall rating**: Pure overall ELO rating
- **Surface-specific**: 50/50 blend of overall and surface-specific ratings

### Glicko-2 Implementation
The Glicko-2 model employs uncertainty-weighted blending:
- **Overall rating**: Pure overall Glicko-2 rating
- **Surface-specific**: Weighted average based on inverse rating deviations

```python
weight_overall = 1 / RD_overall
weight_surface = 1 / RD_surface
blended_rating = (R_overall * weight_overall + R_surface * weight_surface) / (weight_overall + weight_surface)
```

## Temporal Integrity Verification

Several mechanisms ensure no data leakage from future matches:

1. **Rating retrieval**: All rating queries use `WHERE date <= match_date`
2. **H2H matches**: Strictly filtered to pre-tournament window
3. **Style factors**: Computed once per tournament using only historical data
4. **Player selection**: Based solely on pre-tournament performance

## Configuration Options

The system evaluates 14 configurations per rating system:
- Baseline (0% style adjustment)
- Style adjustments: 5%, 10%, 25%, 50%, 75%, 100%
- Each tested with both overall and surface-specific ratings

## Output Structure

Results are organized hierarchically:
```
elo_nmf_results/
├── {configuration}_all_matches.csv      # Match-level predictions
└── {configuration}_tournament_summary.csv # Tournament aggregates

glicko2_nmf_results/
├── {configuration}_all_matches.csv
└── {configuration}_tournament_summary.csv
```

## Evaluation Metrics

Performance is assessed using:
- **Accuracy**: Percentage of correct predictions
- **Brier Score**: Mean squared difference between predicted probabilities and actual outcomes

```
Brier Score = (1/n) * Σ(predicted_probability - actual_outcome)²
```

Where actual_outcome = 1 for win, 0 for loss.

## Implementation Notes

- Database: Neo4j graph database with ATP match data (2000-2024)
- Programming language: Python 3.x
- Key dependencies: neo4j-driver, scikit-learn, pandas, numpy
- Computational requirements: ~5-10 minutes per tournament on standard hardware

## References

1. Sackmann, J. (2025). "An Inside-Out Attempt to Classify Playing Styles." Tennis Abstract. https://www.tennisabstract.com/blog/2025/04/29/an-inside-out-attempt-to-classify-playing-styles/ 

2. Lee, D. D., & Seung, H. S. (1999). "Learning the parts of objects by non-negative matrix factorization." Nature, 401(6755), 788-791.

3. Glickman, M. E. (2001). "Dynamic paired comparison models with stochastic variances." Journal of Applied Statistics, 28(6), 673-689.