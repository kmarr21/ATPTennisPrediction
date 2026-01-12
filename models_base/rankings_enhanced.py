#!/usr/bin/env python3

# ATP rankings enhanced model (momentum + consistency)
"""
Enhanced ranking model with momentum and consistency features
Original contribution beyond McHale & Morton (2011) baseline models

Features:
- Points ratio (log-transformed)
- 3-month momentum (% change in points)
- Consistency (inverse of coefficient of variation)

Usage:
    # single match prediction:
    python rankings_enhanced.py --predict "Novak Djokovic" "Rafael Nadal" 20230601
    
    # batch evaluation:
    python rankings_enhanced.py --batch-evaluate --year 2024
    python rankings_enhanced.py --batch-evaluate --year 2024 --surface clay
    python rankings_enhanced.py --batch-evaluate --year 2024 --tournament Wimbledon
    python rankings_enhanced.py --batch-evaluate --year 2024 --round F
"""

import sys
import os
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "put_your_password_here"

class RankingEnhancedModel:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.model_name = "rankings_enhanced"
        
        print("=" * 60)
        print("ATP RANKINGS ENHANCED MODEL")
        print("Features: Points, Momentum, Consistency")
        print("=" * 60)
        
        # create output directory if needed
        self.output_dir = "model_outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created {self.output_dir}/ directory")
        
        self.model = None
        self.scaler = StandardScaler()
        
    def close(self):
        self.driver.close()
    
    def find_player(self, player_name):
        # find player by name
        with self.driver.session() as session:
            query = """
            MATCH (p:Player)
            WHERE toLower(p.first_name + ' ' + p.last_name) = toLower($name)
            RETURN p.id as id, p.first_name + ' ' + p.last_name as full_name
            """
            result = session.run(query, name=player_name)
            match = result.single()
            
            if match:
                return match['id'], match['full_name']
            
            query = """
            MATCH (p:Player)
            WHERE toLower(p.last_name) CONTAINS toLower($name)
            RETURN p.id as id, p.first_name + ' ' + p.last_name as full_name
            LIMIT 10
            """
            result = session.run(query, name=player_name)
            matches = list(result)
            
            if len(matches) == 1:
                return matches[0]['id'], matches[0]['full_name']
            elif len(matches) > 1:
                print("\nMultiple players found:")
                for i, m in enumerate(matches, 1): print(f"{i}. {m['full_name']}")
                try:
                    choice = int(input("Select player (0 to cancel): "))
                    if 1 <= choice <= len(matches):
                        return matches[choice-1]['id'], matches[choice-1]['full_name']
                except: pass
            
            return None, None
    
    def get_player_features(self, player_id, match_date):
        # get features: current points, 3-month momentum, consistency
        with self.driver.session() as session:
            # get current ranking
            current = session.run("""
                MATCH (p:Player {id: $player_id})-[:HAS_RANKING]->(r:Ranking)
                WHERE r.date < $match_date
                WITH r
                ORDER BY r.date DESC
                LIMIT 1
                RETURN r.rank as position, r.points as points, r.date as date
            """, player_id=player_id, match_date=match_date).single()
            
            if not current or current['points'] <= 0:
                return None
            
            # get 3-month ago ranking
            three_month_ago = match_date - 300  # ~3 months
            past = session.run("""
                MATCH (p:Player {id: $player_id})-[:HAS_RANKING]->(r:Ranking)
                WHERE r.date < $three_month_date AND r.date > $three_month_date - 100
                WITH r
                ORDER BY r.date DESC
                LIMIT 1
                RETURN r.points as points
            """, player_id=player_id, three_month_date=three_month_ago).single()
            
            # get recent history for consistency
            recent = session.run("""
                MATCH (p:Player {id: $player_id})-[:HAS_RANKING]->(r:Ranking)
                WHERE r.date < $match_date AND r.date > $match_date - 200
                RETURN r.points as points
                ORDER BY r.date DESC
                LIMIT 8
            """, player_id=player_id, match_date=match_date)
            recent_points = [r['points'] for r in recent if r['points'] > 0]
            
            # calc features
            features = {
                'points': current['points'],
                'position': current['position']}
            
            # momentum: points change rate over 3 months
            if past and past['points'] > 0:
                features['momentum'] = (current['points'] - past['points']) / past['points']
            else: features['momentum'] = 0
            
            # consistency: inverse of coefficient of variation
            if len(recent_points) >= 3:
                cv = np.std(recent_points) / np.mean(recent_points)
                features['consistency'] = 1 / (1 + cv)
            else: features['consistency'] = 0.5
            
            return features
    
    def fit_model(self, before_date=None, max_rank=250):
        # fit model with momentum and consistency features
        if before_date:
            end_date = before_date - 1
            start_date = max(20010101, before_date - 50000)
        else:
            start_date = 20100101
            end_date = 20191231
        
        print(f"Fitting enhanced model on data from {start_date} to {end_date}...")
        print(f"Filtering to players ranked {max_rank} or better...")
        
        with self.driver.session() as session:
            query = """
            MATCH (p1:Player)-[:WON]->(m:Match)<-[:LOST]-(p2:Player)
            MATCH (m)-[:PLAYED_IN]->(t:Tournament)
            WHERE t.date >= $start_date AND t.date <= $end_date
            AND NOT m.score CONTAINS 'W/O'
            RETURN p1.id as winner_id, p2.id as loser_id, t.date as match_date
            ORDER BY RAND()
            LIMIT 10000
            """
            result = session.run(query, start_date=start_date, end_date=end_date)
            matches = list(result)
        
        print(f"Processing {len(matches)} matches...")
        
        X = []
        y = []
        valid_count = 0
        
        for match in tqdm(matches, desc="Extracting features"):
            w_features = self.get_player_features(match['winner_id'], match['match_date'])
            l_features = self.get_player_features(match['loser_id'], match['match_date'])
            
            if not w_features or not l_features:
                continue
            
            if w_features['position'] > max_rank or l_features['position'] > max_rank:
                continue
            
            valid_count += 1
            
            # create feature vector: [log_points_ratio, momentum_diff, consistency_diff]
            features_w = [
                np.log(w_features['points'] / l_features['points']),
                w_features['momentum'] - l_features['momentum'],
                w_features['consistency'] - l_features['consistency']]
            
            X.append(features_w)
            y.append(1)
            
            # loser perspective
            features_l = [
                np.log(l_features['points'] / w_features['points']),
                l_features['momentum'] - w_features['momentum'],
                l_features['consistency'] - w_features['consistency']
            ]
            
            X.append(features_l)
            y.append(0)
        
        print(f"Created {len(X)} training samples from {valid_count} matches")
        
        if len(X) > 100:
            # scale features
            X_scaled = self.scaler.fit_transform(X)
            # fit model w/ regularization
            self.model = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
            self.model.fit(X_scaled, y)
            
            feature_names = ['log_points_ratio', 'momentum_diff', 'consistency_diff']
            
            print(f"\nModel coefficients:")
            for name, coef in zip(feature_names, self.model.coef_[0]): print(f"  {name}: {coef:.4f}")
            print(f"  Intercept: {self.model.intercept_[0]:.4f}")
        else:
            print("Insufficient data for model fitting - using fallback")
            self.scaler.fit([[0, 0, 0]]) # dummy fit if needed
            self.model = LogisticRegression()
            self.model.fit([[1], [0]], [1, 0]) # minimal model
    
    def predict_match(self, player1_name, player2_name, match_date, verbose=True):
        # predict using enhanced model
        if self.model is None:
            self.fit_model(before_date=match_date)
        
        p1_id, p1_full = self.find_player(player1_name)
        p2_id, p2_full = self.find_player(player2_name)
        
        if not p1_id or not p2_id:
            if verbose: print("Could not find one or both players")
            return None
        
        p1_features = self.get_player_features(p1_id, match_date)
        p2_features = self.get_player_features(p2_id, match_date)
        
        if not p1_features or not p2_features:
            if verbose: print("Insufficient data for prediction")
            return None
        
        # create feature vector
        features = [
            np.log(p1_features['points'] / p2_features['points']),
            p1_features['momentum'] - p2_features['momentum'],
            p1_features['consistency'] - p2_features['consistency']]
        
        # scale & predict
        features_scaled = self.scaler.transform([features])
        prob1 = self.model.predict_proba(features_scaled)[0][1]
        
        if verbose:
            print(f"\nENHANCED MODEL PREDICTION")
            print(f"{p1_full} vs {p2_full}")
            print(f"\nPoints: {p1_features['points']:.0f} vs {p2_features['points']:.0f}")
            print(f"3-month momentum: {p1_features['momentum']:+.1%} vs {p2_features['momentum']:+.1%}")
            print(f"Consistency: {p1_features['consistency']:.2f} vs {p2_features['consistency']:.2f}")
            print(f"\nProbability {p1_full} wins: {prob1:.1%}")
        
        return {
            'player1_name': p1_full,
            'player2_name': p2_full,
            'player1_prob': prob1,
            'player2_prob': 1 - prob1}
    
    def batch_evaluate(self, year, surface='overall', tournament=None, round=None, output_file=None):
        # batch eval enhanced model w/ specified filters
        """
        Args:
            year: year to evaluate (required)
            surface: surface filter ('overall', 'hard', 'clay', 'grass')
            tournament: tournament name to filter (partial match, optional)
            round: round to filter (e.g., 'F', 'SF', 'QF', optional)
            output_file: explicit output path, or None for auto-generate
        
        Returns:
            DataFrame with comprehensive results
        """
        # fit model on data before this period
        tournament_start = year * 10000 + 101
        self.fit_model(before_date=tournament_start)
        
        start_date = year * 10000 + 101
        end_date = year * 10000 + 1231
        
        print(f"\nBatch Evaluating Rankings ENHANCED Model:")
        print(f"  Year: {year}")
        print(f"  Surface: {surface}")
        if tournament: print(f"  Tournament: {tournament}")
        if round: print(f"  Round: {round}")
        
        # build query w/ filters
        filters = [
            f"t.date >= {start_date}",
            f"t.date <= {end_date}",
            "NOT m.score CONTAINS 'W/O'"
        ]
        
        if surface != 'overall': filters.append(f"toLower(t.surface) = '{surface.lower()}'")
        if tournament: filters.append(f"toLower(t.name) CONTAINS toLower('{tournament}')")
        if round: filters.append(f"m.round = '{round}'")
        where_clause = " AND ".join(filters)
        
        with self.driver.session() as session:
            query = f"""
            MATCH (p1:Player)-[:WON]->(m:Match)<-[:LOST]-(p2:Player)
            MATCH (m)-[:PLAYED_IN]->(t:Tournament)
            WHERE {where_clause}
            RETURN 
                p1.id as winner_id,
                p2.id as loser_id,
                p1.first_name + ' ' + p1.last_name as winner_name,
                p2.first_name + ' ' + p2.last_name as loser_name,
                t.date as match_date,
                m.id as match_id,
                t.surface as surface,
                t.name as tournament,
                t.level as tournament_level,
                m.round as round
            ORDER BY t.date, m.match_num
            """
            result = session.run(query)
            test_matches = list(result)
        
        print(f"Found {len(test_matches)} matches to evaluate...")
        
        results = []
        skipped = 0
        
        for match in tqdm(test_matches, desc="Evaluating"):
            w_features = self.get_player_features(match['winner_id'], match['match_date'])
            l_features = self.get_player_features(match['loser_id'], match['match_date'])
            
            if not w_features or not l_features:
                skipped += 1
                continue
            
            features = [
                np.log(w_features['points'] / l_features['points']),
                w_features['momentum'] - l_features['momentum'],
                w_features['consistency'] - l_features['consistency']
            ]
            
            features_scaled = self.scaler.transform([features])
            w_prob = self.model.predict_proba(features_scaled)[0][1]
            
            results.append({
                # std cols
                'match_id': match['match_id'],
                'match_date': match['match_date'],
                'player1_name': match['winner_name'],
                'player2_name': match['loser_name'],
                'actual_winner': 1,
                'predicted_prob_p1': w_prob,
                'predicted_prob_p2': 1 - w_prob,
                'predicted_winner': 1 if w_prob > 0.5 else 2,
                'correct_prediction': 1 if w_prob > 0.5 else 0,
                'brier_score': (w_prob - 1) ** 2,
                'surface': match['surface'],
                'tournament': match['tournament'],
                'tournament_level': match['tournament_level'],
                'round': match['round'],
                
                # enhanced model specific cols
                'player1_points': w_features['points'],
                'player2_points': l_features['points'],
                'player1_momentum': w_features['momentum'],
                'player2_momentum': l_features['momentum'],
                'player1_consistency': w_features['consistency'],
                'player2_consistency': l_features['consistency']
            })
        
        if results:
            df = pd.DataFrame(results)
            accuracy = df['correct_prediction'].mean()
            avg_brier = df['brier_score'].mean()
            coverage = len(results) / len(test_matches) if test_matches else 0
            
            print(f"\nEVALUATION RESULTS")
            print("=" * 50)
            print(f"Total matches analyzed: {len(test_matches)}")
            print(f"Predictions made: {len(results)}")
            print(f"Coverage: {coverage:.1%}")
            print(f"Accuracy: {accuracy:.1%}")
            print(f"Brier Score: {avg_brier:.3f}")
            
            # breakdown by round if sufficient data
            if len(df) > 50:
                print(f"\nAccuracy by Round:")
                for round_name in ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']:
                    round_df = df[df['round'] == round_name]
                    if len(round_df) > 0:
                        acc = round_df['correct_prediction'].mean()
                        print(f"  {round_name}: {acc:.1%} ({len(round_df)} matches)")
            
            # generate output filename if not specified
            if not output_file:
                output_file = self.generate_output_filename(
                    year=year, surface=surface, tournament=tournament, round=round
                )
            
            # ensure output file is in model_outputs directory
            if not output_file.startswith(self.output_dir):
                output_file = os.path.join(self.output_dir, os.path.basename(output_file))
            
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            return df
        else:
            print("No valid predictions could be made!")
            return None
    
    def generate_output_filename(self, year, surface='overall', tournament=None, round=None):
        #generate standardized filename in model_outputs/
        parts = [self.model_name]
        
        if tournament: parts.append(tournament.replace(' ', '_'))
        if round: parts.append(round)
        parts.append(str(year))
        if surface != 'overall': parts.append(surface)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parts.append(timestamp)
        
        filename = '_'.join(parts) + '.csv'
        return os.path.join(self.output_dir, filename)

def main():
    model = RankingEnhancedModel()
    
    try:
        if '--batch-evaluate' in sys.argv:
            # batch evaluation mode
            if '--year' not in sys.argv:
                print("Error: --year is required for batch evaluation")
                print("Usage: python rankings_enhanced.py --batch-evaluate --year YYYY [options]")
                return
            
            # parse year
            year_idx = sys.argv.index('--year')
            year = int(sys.argv[year_idx + 1])
            
            # parse optional parameters
            surface = 'overall'
            if '--surface' in sys.argv:
                surf_idx = sys.argv.index('--surface')
                surface = sys.argv[surf_idx + 1]
            
            tournament = None
            if '--tournament' in sys.argv:
                tourn_idx = sys.argv.index('--tournament')
                tournament = sys.argv[tourn_idx + 1]
            
            round_filter = None
            if '--round' in sys.argv:
                round_idx = sys.argv.index('--round')
                round_filter = sys.argv[round_idx + 1]
            
            output_file = None
            if '--output' in sys.argv:
                out_idx = sys.argv.index('--output')
                output_file = sys.argv[out_idx + 1]
            
            # run batch eval
            model.batch_evaluate(
                year=year,
                surface=surface,
                tournament=tournament,
                round=round_filter,
                output_file=output_file
            )
        
        elif '--predict' in sys.argv and len(sys.argv) >= 5:
            # single prediction
            model.predict_match(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        
        else:
            print("Usage examples:")
            print("\n  Single prediction:")
            print("    python rankings_enhanced.py --predict 'Carlos Alcaraz' 'Daniil Medvedev' 20230601")
            print("\n  Batch evaluation:")
            print("    python rankings_enhanced.py --batch-evaluate --year 2024")
            print("    python rankings_enhanced.py --batch-evaluate --year 2024 --surface clay")
            print("    python rankings_enhanced.py --batch-evaluate --year 2024 --tournament Wimbledon")
            print("    python rankings_enhanced.py --batch-evaluate --year 2024 --round F")
    
    finally:
        model.close()

if __name__ == "__main__":
    main()
