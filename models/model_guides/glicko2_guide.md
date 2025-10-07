# Glicko-2 Tennis Prediction Model - Technical Documentation

## Overview

I implemented a Glicko-2 rating system for tennis match prediction, adapting Professor Mark Glickman's rating system to the specific dynamics of professional tennis. The model extends beyond simple win/loss probabilities by incorporating rating uncertainty (RD) and player volatility, providing confidence intervals for predictions.

## Model Architecture

### Core Glicko-2 Mathematics

The Glicko-2 system represents each player with three parameters:

- **$\mu$ (mu)**: Rating on Glicko-2 scale (0 = 1500 Elo equivalent)
- **$\phi$ (phi)**: Rating deviation on Glicko-2 scale  
- **$\sigma$ (sigma)**: Volatility (expected fluctuation in performance)

#### Scale Conversion

The system operates on two scales:

**Glicko scale** (human-readable):
- Rating: typically 1200-2800
- RD: typically 30-350

**Glicko-2 scale** (computational):

$$\mu = \frac{\text{rating} - 1500}{173.7178}$$

$$\phi = \frac{\text{RD}}{173.7178}$$

Where 173.7178 is Glickman's conversion constant [1].

### Update Equations

For a rating period with $m$ matches, the update process follows these steps:

#### Step 1: Compute Auxiliary Values

**g-function** (reduces impact of uncertain opponents):

$$g(\phi_j) = \frac{1}{\sqrt{1 + 3\phi_j^2/\pi^2}}$$

**E-function** (expected score):

$$E(\mu, \mu_j, \phi_j) = \frac{1}{1 + \exp(-g(\phi_j)(\mu - \mu_j))}$$

#### Step 2: Compute Variance

$$v = \left[\sum g(\phi_j)^2 \times E(\mu, \mu_j, \phi_j) \times (1 - E(\mu, \mu_j, \phi_j))\right]^{-1}$$

#### Step 3: Compute Performance Improvement

$$\Delta = v \times \sum g(\phi_j) \times (s_j - E(\mu, \mu_j, \phi_j))$$

Where $s_j \in \{0, 1\}$ is the actual match result.

#### Step 4: Update Volatility

I use the Illinois algorithm (a root-finding method) to solve for $\sigma'$. Glickman describes this iterative approach in his 2012 paper [1]:

$$f(x) = \frac{e^x(\Delta^2 - \phi^2 - v - e^x)}{2(\phi^2 + v + e^x)^2} - \frac{x - \ln(\sigma^2)}{\tau^2}$$

Where $\tau$ is the system constant controlling volatility change rate. The Illinois algorithm is a modification of the regula falsi method that prevents stagnation [1].

#### Step 5: Update RD and Rating

$$\phi^* = \sqrt{\phi^2 + \sigma'^2}$$

$$\phi' = \frac{1}{\sqrt{1/\phi^{*2} + 1/v}}$$

$$\mu' = \mu + \phi'^2 \times \sum g(\phi_j) \times (s_j - E(\mu, \mu_j, \phi_j))$$

### Match Prediction

For predicting a match between players A and B:

$$P(\text{A wins}) = \frac{1}{1 + 10^{-g(\sqrt{RD_A^2 + RD_B^2}) \times (R_A - R_B) / 400}}$$

This accounts for combined uncertainty, giving less extreme predictions when either player has high RD.

## Tennis-Specific Adaptations

### Rating Periods

I use **14-day rating periods** rather than Glickman's monthly periods. This bi-weekly approach better captures tennis tournament dynamics where players compete intensively for 1-2 weeks then rest. All matches within a period are processed together before updating ratings.

### Surface-Specific Ratings

Each player maintains four separate rating sets:
- Overall (all surfaces)
- Hard court
- Clay court  
- Grass court

**Surface Initialization**: When a player first competes on a new surface, I initialize their surface rating at their current overall rating but with elevated uncertainty (RD = 200 instead of maintaining current RD). This reflects that surface proficiency is uncertain until demonstrated.

**Surface Blending for Predictions**: For surface-specific predictions, I use an uncertainty-weighted average:

$$\text{weight}_{\text{overall}} = \frac{1}{RD_{\text{overall}}}$$

$$\text{weight}_{\text{surface}} = \frac{1}{RD_{\text{surface}}}$$

$$\text{rating}_{\text{blended}} = \frac{\text{rating}_{\text{overall}} \times \text{weight}_{\text{overall}} + \text{rating}_{\text{surface}} \times \text{weight}_{\text{surface}}}{\text{weight}_{\text{overall}} + \text{weight}_{\text{surface}}}$$

### Tournament Level Weighting

After discovering that Jacob Fearnley reached an inflated rating of 2128 (18th in my system) despite never defeating a top-100 player, I realized the original weights were insufficient. Fearnley had accumulated his rating through consistent Challenger-level victories, exposing a flaw where quantity could overwhelm quality.

I adjusted the tournament weights as follows:

```python
if tournament_level == 'G':     # Grand Slam
    weight = 1.3
elif tournament_level == 'M':   # Masters 1000
    weight = 1.15
elif tournament_level == 'A':   # ATP 250/500
    weight = 1.0                # baseline
elif tournament_level == 'C':   # Challenger
    weight = 0.7                # reduced from 0.9
```

The original 0.9 weight for Challengers was too generous. The new 0.7 weight ensures that defeating lower-ranked players at Challengers contributes less to rating growth, preventing artificial inflation from "farming" weak fields.

### Temporal Validation

**Critical**: When making predictions for date T, I only use rating nodes created from matches before date T. The weekly snapshots ensure temporal integrity - I query for the most recent snapshot where `date <= match_date`.

### Off-Season Handling

December is treated as off-season for tennis. When calculating periods of inactivity for RD decay, I exclude December weeks, preventing artificial uncertainty growth during the standard break period.

### Retirement Detection

Players inactive for more than 2 years are excluded from active player listings in verification reports, though their historical nodes remain for potential comeback scenarios.

## Implementation Details

### Node Structure

```cypher
CREATE (g:Glicko2 {
    player_id: String,
    date: Integer,          // YYYYMMDD format
    rating_overall: Float,
    rating_hard: Float,
    rating_clay: Float,
    rating_grass: Float,
    rd_overall: Float,
    rd_hard: Float,
    rd_clay: Float,
    rd_grass: Float,
    volatility_overall: Float,
    volatility_hard: Float,
    volatility_clay: Float,
    volatility_grass: Float,
    total_matches: Integer
})

CREATE (p:Player)-[:HAS_GLICKO2]->(g:Glicko2)
```

### Configurable Parameters

I made key parameters configurable to enable variant testing:

```python
tau = 0.4           # System constant (Glickman recommends 0.3-1.2, default 0.5) [1]
period_days = 14    # Rating period length (Glickman uses monthly for chess) [1]
rd_decay = 30       # RD growth per inactive period (my choice based on testing)
surface_rd = 200    # Initial RD for new surface (my adaptation for tennis)
```

**Parameter Justifications:**

- **tau = 0.4**: Slightly below Glickman's default of 0.5 [1]. I chose this after observing that tennis ratings are more stable than online gaming environments due to fewer matches and more consistent competition levels.

- **period_days = 14**: Glickman uses monthly periods for chess [1]. I chose bi-weekly to better capture tennis tournament cycles where players compete intensively for 1-2 weeks then rest.

- **rd_decay = 30**: This is my empirical choice. Glickman doesn't specify a fixed value, instead noting it should reflect the sport's dynamics [2]. I found 30 points per period maintains reasonable uncertainty growth during breaks.

- **surface_rd = 200**: My tennis-specific addition. Starting at 350 (Glickman's new player default [1]) seemed excessive for established players trying a new surface, while maintaining current RD ignored surface-specific uncertainty.

### Model Variants

To facilitate comparison, I support multiple model variants stored with different node labels:

```python
node_label = f"Glicko2{suffix}"  # e.g., Glicko2_stable, Glicko2_volatile
```

This allows parallel storage and evaluation of different parameter sets.

## Usage Examples

### Import Rating Nodes

```bash
# Default configuration
python import_glicko2_nodes.py

# Stable variant (tau=0.3)
python import_glicko2_nodes.py --tau 0.3 --suffix "_stable"

# Volatile variant (tau=0.7)
python import_glicko2_nodes.py --tau 0.7 --suffix "_volatile"

# Different period length
python import_glicko2_nodes.py --period 7 --suffix "_weekly"
```

### Single Match Predictions

```bash
# Basic prediction
python glicko2_v1.py "Djokovic" "Alcaraz" 20240601

# Surface-specific
python glicko2_v1.py "Djokovic" "Alcaraz" 20240601 --surface clay

# Using variant
python glicko2_v1.py "Djokovic" "Alcaraz" 20240601 --variant _stable
```

### Tournament Evaluation

```bash
# Evaluate Wimbledon 2024
python glicko2_v1.py --evaluate-tournament Wimbledon 2024 --surface grass

# Batch evaluation with CSV output
python glicko2_v1.py --batch-evaluate --tournament "Roland Garros" --year 2024 --surface clay

# Compare variants
python glicko2_v1.py --batch-evaluate --year 2024 --variant _stable
python glicko2_v1.py --batch-evaluate --year 2024 --variant _volatile
```

## Results and Analysis

### Key Findings

**Accuracy vs Calibration Trade-off**: Surface-specific predictions improve accuracy but worsen Brier scores. This occurs because surface-specific ratings have higher uncertainty (RD), leading to more confident predictions that, when wrong, severely penalize the Brier score.

**Uncertainty Paradox**: For grass court predictions, high-uncertainty matches showed better prediction accuracy than low-uncertainty ones. This counterintuitive result reflects that high RD often indicates players who rarely compete on grass - the model correctly identifies these surface specialists or weaknesses.

**Rating Inflation Control**: The adjusted Challenger weight (0.7) successfully prevents rating inflation from lower-tier success while still rewarding consistent performance at that level.

## System Requirements and Dependencies

```python
# Core dependencies
neo4j >= 4.0
numpy >= 1.19
pandas >= 1.2
scipy >= 1.5
tqdm >= 4.60
datetime (standard library)
math (standard library)
collections (standard library)
argparse (standard library)
```

## Database Connection

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "RolandGarros2195!"
```

## Computational Performance

- **Import time**: ~30-45 minutes for 2000-2024 data (locally)
- **Node count**: ~650,000 nodes per variant (500 players × 1,300 weeks)
- **Storage**: ~130 MB per variant
- **Prediction time**: <100ms per match
- **Tournament evaluation**: ~5-10 seconds for 127 matches

## Future Improvements

1. **Dynamic tau**: Adjust volatility parameter based on player age/experience
2. **Head-to-head adjustments**: Incorporate historical matchup data
3. **Injury modeling**: Detect and adjust for post-injury performance changes
4. **Qualifier handling**: Better initialization for players entering from qualifying
5. **Doubles integration**: Extend system to doubles with pair synergy factors

## References

[1] Glickman, M.E. (2012). "Example of the Glicko-2 system." Boston University. http://www.glicko.net/glicko/glicko2.pdf

[2] Glickman, M.E. (2001). "Dynamic paired comparison models with stochastic variances." Journal of Applied Statistics, 28(6), 673-689.

[3] Glickman, M.E. (1999). "Parameter estimation in large dynamic paired comparison experiments." Applied Statistics, 48, 377-394.

[4] Tennis Abstract. "Elo Ratings for ATP Tennis." http://www.tennisabstract.com/blog/category/elo-ratings/

## Author Notes

The Glicko-2 system's incorporation of uncertainty (RD) and volatility provides richer predictions than simple ELO ratings. While the additional complexity doesn't dramatically improve accuracy, it offers valuable confidence intervals and better handles players with limited data. The tennis-specific adaptations, particularly the revised tournament weights and surface handling, were crucial for producing realistic ratings that align with professional tennis dynamics.