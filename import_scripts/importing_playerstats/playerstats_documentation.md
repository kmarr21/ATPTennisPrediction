# PlayerStats Node Documentation

## Overview
PlayerStats nodes contain weekly aggregated tennis statistics for each player. Each node represents a snapshot of a player's performance metrics calculated for a specific Monday, using different time windows to capture recent form and longer-term trends. These are only statistics that can be derived from the match nodes: point-by-point data and derived stats are separate.

## Node Identification Properties
- **player_id**: Unique identifier linking to the Player node
- **date**: The Monday date (YYYYMMDD format) for which statistics are calculated

## Time Windows
Statistics are calculated across five time windows:
- **2w**: Previous 2 weeks
- **4w**: Previous 4 weeks  
- **8w**: Previous 8 weeks
- **12w**: Previous 12 weeks
- **52w**: Previous 52 weeks (full year)

## Statistics by Category

### Basic Performance Metrics
For each time window (2w, 4w, 8w, 12w, 52w):
- **matches_played**: Total matches played in the window
- **wins**: Number of matches won
- **losses**: Number of matches lost
- **win_pct**: Win percentage (wins / matches_played)

### Serve Statistics

#### Percentages
- **ace_pct**: Aces divided by total serve points
- **df_pct**: Double faults divided by total serve points
- **first_serve_pct**: First serves in divided by total serve points
- **first_serve_won_pct**: Points won on first serve divided by first serves in
- **second_serve_won_pct**: Points won on second serve divided by second serves attempted
- **serve_points_won_pct**: Total serve points won divided by total serve points
- **service_games_held_pct**: Service games held divided by total service games

#### Break Point Statistics
- **bp_faced_total**: Total break points faced when serving
- **bp_saved_total**: Total break points saved when serving
- **bp_saved_pct**: Break points saved divided by break points faced
- **bp_faced_per_game**: Average break points faced per service game

#### Efficiency Metrics
- **service_game_efficiency**: Average points needed per service game (total serve points / total service games)

### Return Statistics

#### Percentages
- **return_points_won_pct**: Return points won divided by total return points
- **first_return_won_pct**: Points won against first serves divided by first serves faced
- **second_return_won_pct**: Points won against second serves divided by second serves faced
- **return_games_broken_pct**: Return games broken divided by total return games

#### Break Point Statistics
- **bp_created_total**: Total break points created when returning
- **bp_converted_total**: Total break points converted when returning
- **bp_converted_pct**: Break points converted divided by break points created
- **bp_created_per_return_game**: Average break points created per return game

#### Efficiency Metrics
- **return_game_impact**: Opponent's average points per game when player is returning (opponent serve points / opponent service games)

### Combined Metrics
- **total_points_won_pct**: Total points won (serve + return) divided by total points played
- **efficiency_ratio**: Opponent's service game efficiency divided by player's service game efficiency

### Dominance Metrics
- **games_ratio**: Games won divided by games lost
- **sets_ratio**: Sets won divided by sets lost
- **straight_sets_pct**: Percentage of wins achieved in straight sets
- **tiebreak_pct**: Tiebreaks won divided by total tiebreaks played

### Activity Metrics (2w, 4w, 8w only)
- **minutes_total**: Total minutes played in the window
- **minutes_avg**: Average minutes per match
- **matches_per_week**: Average matches per week in the window

### Pressure Situation Metrics (12w, 52w only)
- **deciding_set_pct**: Win percentage in deciding sets (3rd set in best-of-3, 5th set in best-of-5)
- **close_match_pct**: Win percentage in matches decided by 3 or fewer games
- **upset_rate**: Percentage of wins when ranked lower than opponent
- **upset_avg_magnitude**: Average ranking points differential in upset wins
- **defend_rate**: Percentage of wins when ranked higher than opponent

### Surface-Specific Statistics (52w only)
For each surface (Hard, Clay, Grass):
- **{surface}_matches**: Number of matches played on surface
- **{surface}_win_pct**: Win percentage on surface
- **{surface}_serve_pct**: Serve points won percentage on surface
- **{surface}_return_pct**: Return points won percentage on surface

### Streak Statistics
- **streak_matches_won**: Current consecutive matches won (0 if on losing streak)
- **streak_matches_lost**: Current consecutive matches lost (0 if on winning streak)
- **streak_tiebreaks**: Current tiebreak streak (positive for wins, negative for losses)

## Value Conventions
- Percentages are stored as decimals (0.0 to 1.0)
- A value of -1 indicates the statistic could not be calculated (division by zero or insufficient data)
- All counting statistics (totals, matches, games) are stored as integers or floats
- Date values are stored as integers in YYYYMMDD format

## Node Structure Summary
Each PlayerStats node contains:
- 2 identification properties
- 34 properties for 2-week window
- 34 properties for 4-week window  
- 34 properties for 8-week window
- 36 properties for 12-week window (includes pressure situations)
- 48 properties for 52-week window (includes pressure situations and surfaces)
- 3 streak properties

**Total: 191 properties per node**