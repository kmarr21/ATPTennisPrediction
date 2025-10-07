# Tennis Abstract ELO Implementation Guide

## Part I: ELO Rating System Implementation

### Overview

My ELO rating system follows the methodology described by Jeff Sackmann's Tennis Abstract[^1][^2], with implementation details inferred from available documentation (primarily via the Tennis Abstract blog). The system calculates dynamic ratings for tennis players based on match results, with separate ratings for overall performance and surface-specific play.

### Core ELO Calculation

#### Expected Score Formula

The probability of Player A defeating Player B is calculated using the standard ELO formula:

$$P(A) = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

Where:
- $R_A$ = Player A's rating
- $R_B$ = Player B's rating

#### Rating Update

After a match, ratings are updated as:

$$R_{new} = R_{old} + K \times (S - E)$$

Where:
- $K$ = K-factor (dynamic, see below)
- $S$ = Actual score (1 for win, 0 for loss)
- $E$ = Expected score from probability formula

### Dynamic K-Factor

Following Tennis Abstract's approach, I use a match-count dependent K-factor:

$$K = \frac{250}{(m + 5)^{0.4}}$$

Where $m$ = total career matches played

This formula produces:
- New players (0 matches): $K \approx 63$
- After 100 matches: $K \approx 39$
- After 1000 matches: $K \approx 20$

#### Grand Slam Bonus

Grand Slam matches receive a 1.1x multiplier to the K-factor, reflecting their increased importance:

$$K_{GS} = 1.1 \times K_{base}$$

### Surface-Specific Ratings

Players maintain four separate ratings:
1. **Overall rating**: Updated based on head-to-head overall ratings
2. **Surface ratings** (hard, clay, grass): Updated using a blended approach

#### Surface Rating Calculation

For surface-specific matches, the expected score uses a 50/50 blend:

$$R_{blend} = 0.5 \times R_{overall} + 0.5 \times R_{surface}$$

The expected score is calculated using these blended ratings, but surface ratings are updated independently with their own K-factors based on surface-specific match counts.

#### Surface Initialization

When a player first competes on a surface, their surface rating is initialized to their current overall rating (not 1500). This prevents unrealistic rating gaps for established players trying new surfaces.

### Absence and Injury Handling

Based on Tennis Abstract's methodology[^2], players returning from extended absences face rating penalties and receive temporary K-factor boosts.

#### Absence Penalty

Penalties are applied when a player returns after 8+ weeks away:

- **8-10 weeks**: 100 points
- **10-52 weeks**: Linear interpolation from 100 to 150 points
- **52+ weeks**: 150 points

The penalty is calculated as:

$$P = \begin{cases}
0 & \text{if } w < 8 \\
100 & \text{if } 8 \leq w \leq 10 \\
100 + \frac{50(w - 10)}{42} & \text{if } 10 < w \leq 52 \\
150 & \text{if } w > 52
\end{cases}$$

Where $w$ = weeks absent (excluding December)

#### December Exclusion

December is excluded from absence calculations as it represents the traditional ATP off-season. This prevents artificial penalties for the standard tour break. Certain tournaments are played in December (e.g., NextGen Finals), but these are few and far between. These matches do still count towards updating Elo rankings, they are just not included when considering absence/injury handling.

#### Post-Return K-Factor Boost

Returning players receive an increased K-factor that declines over 20 matches:

$$K_{multiplier} = 1.5 - \frac{0.5 \times m_{return}}{20}$$

Where $m_{return}$ = matches played since return (capped at 20).

This gives returning players 1.5x K-factor for their first match back, declining linearly to 1.0x after 20 matches.

#### Multiple Layoffs

If a player has multiple layoffs within 2 years, the absence lengths are combined for penalty calculation (attempt at handling recurrent injury).

### Implementation Details

#### Starting Ratings

All players begin at 1500 ELO points, consistent with standard ELO systems.

#### Rating Floor

Ratings cannot fall below 1000 points, preventing unrealistic negative spirals.

#### Weekly Snapshots

ELO ratings are stored as weekly snapshots (every Monday) in the database, enabling point-in-time queries for historical analysis and predictions.

#### Data Range and Stability

- **Data coverage**: January 2000 - December 2024
- **Stabilization period**: ~2 years recommended
- **Minimum reliability**: January 2001 (1 year of data)
- **Full reliability**: January 2002+ (2 years of data)

Predictions before 2002 may be unreliable as the system requires time to differentiate player skill levels from the uniform 1500 starting point.

### Conservative Nature of ELO

ELO systems are inherently conservative: established players with extensive match histories (like Djokovic with 1400+ matches) have very low K-factors (~18), making their ratings resistant to change. This means:

- Recent form may be underweighted for established players
- Young players' improvements may be captured slowly
- Historical dominance can overshadow current performance

This is a fundamental characteristic of ELO systems. I tried some workarounds to make my Elo system a bit more flexible, but there is only so much you can do without changing the fundamentals of an Elo model.

---

## Part II: ELO Prediction Model

### Model Overview

The prediction model (`elo_v2.py`) uses pre-calculated ELO ratings to predict match outcomes. It retrieves the most recent ratings before a match date and calculates win probabilities.

### Prediction Methodology

#### Overall Predictions

For overall predictions, I use players' overall ELO ratings directly:

$$P(A) = \frac{1}{1 + 10^{(R_{B,overall} - R_{A,overall})/400}}$$

#### Surface-Specific Predictions

For surface predictions, I use the Tennis Abstract 50/50 blend:

$$R_{A,blend} = 0.5 \times R_{A,overall} + 0.5 \times R_{A,surface}$$

$$R_{B,blend} = 0.5 \times R_{B,overall} + 0.5 \times R_{B,surface}$$

$$P(A) = \frac{1}{1 + 10^{(R_{B,blend} - R_{A,blend})/400}}$$

### Usage Examples

#### Single Match Prediction

```bash
# Overall ELO prediction
python elo_v2.py "Novak Djokovic" "Rafael Nadal" 20230601

# Surface-specific prediction
python elo_v2.py "Novak Djokovic" "Rafael Nadal" 20230601 --surface clay
```

#### Evaluation Modes

```bash
# Evaluate specific year
python elo_v2.py --evaluate-year 2021
python elo_v2.py --evaluate-year 2019 --surface clay

# Evaluate specific tournament
python elo_v2.py --evaluate-tournament "Wimbledon" 2018
python elo_v2.py --evaluate-tournament "Wimbledon" 2018 --surface grass

# Batch evaluation with CSV output
python elo_v2.py --batch-evaluate --year 2021
python elo_v2.py --batch-evaluate --year 2019 --surface clay
python elo_v2.py --batch-evaluate --tournament "Wimbledon" --year 2018
```

#### Interactive Mode

```bash
# Launch interactive prediction interface
python elo_v2.py
```

### Output Files

Batch evaluations generate timestamped CSV files in the `elo_output_files/` directory with columns including:
- Match identifiers and date
- Player names and actual winner
- Predicted probabilities
- ELO ratings used
- Brier scores for calibration assessment

### Model Evaluation Metrics

#### Accuracy
Percentage of correct predictions (favorite wins)

#### Brier Score
Measures probability calibration:

$$BS = (p - o)^2$$

Where $p$ = predicted probability, $o$ = actual outcome (0 or 1)

Lower scores indicate better calibration (perfect = 0, worst = 1)

#### Coverage
Percentage of matches where predictions could be made (both players have ELO ratings)

### Important Considerations

1. **Early predictions**: Avoid predictions before 2002 when possible due to rating stabilization
2. **Surface context**: Use surface-specific predictions for surface-specific tournaments (e.g., grass for Wimbledon)
3. **Absence penalties**: The model uses ratings that already incorporate absence penalties from the import process
4. **Match uniqueness**: Each match is counted once (no double-counting for both players)

---

## References

[^1]: Sackmann, J. (2019). "An Introduction to Tennis Elo." Tennis Abstract. https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/

[^2]: Sackmann, J. (2018). "Handling Injuries and Absences with Tennis Elo." Tennis Abstract. https://www.tennisabstract.com/blog/2018/05/15/handling-injuries-and-absences-with-tennis-elo/