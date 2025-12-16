#!/usr/bin/env python3

# ATP Rankings Joint Model (position + points)
"""
Uses both ranking position and points in a combined logistic regression model
Based on McHale & Morton (2011) framework

Usage:
    # single match prediction:
    python rankings_joint.py --predict "Novak Djokovic" "Rafael Nadal" 20230601
    
    # batch evaluation:
    python rankings_joint.py --batch-evaluate --year 2024
    python rankings_joint.py --batch-evaluate --year 2024 --surface clay
    python rankings_joint.py --batch-evaluate --year 2024 --tournament Wimbledon
    python rankings_joint.py --batch-evaluate --year 2024 --round F
"""

import sys
import os
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "put_your_password_here"

class RankingJointModel:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.model_name = "rankings_joint"
        
        print("=" * 60)
        print("ATP RANKINGS JOINT MODEL (Position + Points)")
        print("=" * 60)
        
        # create output directory if needed
        self.output_dir = "model_outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created {self.output_dir}/ directory")
        
        self.model = None
        self.position_coef = None
        self.points_coef = None
        
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
                except:
                    pass
            
            return None, None
    
    def get_ranking_at_date(self, player_id, match_date):
        # get player's ranking at or before specified date
        with self.driver.session() as session:
            query = """
            MATCH (p:Player {id: $player_id})-[:HAS_RANKING]->(r:Ranking)
            WHERE r.date < $match_date
            WITH r
            ORDER BY r.date DESC
            LIMIT 1
            RETURN r.rank as position, r.points as points, r.date as ranking_date
            """
            result = session.run(query, player_id=player_id, match_date=match_date)
            return result.single()
    
    def fit_model(self, before_date=None, max_rank=250):
        # fit joint model using both position AND points
        if before_date:
            end_date = before_date - 1
            start_date = max(20010101, before_date - 50000)
        else:
            start_date = 20100101
            end_date = 20191231
        
        print(f"Fitting joint model on data from {start_date} to {end_date}...")
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
        
        X = [] # features: [rank_diff, log_points_ratio]
        y = [] # labels
        
        filtered_count = 0
        
        for match in tqdm(matches, desc="Processing matches"):
            w_rank = self.get_ranking_at_date(match['winner_id'], match['match_date'])
            l_rank = self.get_ranking_at_date(match['loser_id'], match['match_date'])
            
            if w_rank and l_rank and w_rank['points'] > 0 and l_rank['points'] > 0:
                if w_rank['position'] <= max_rank and l_rank['position'] <= max_rank:
                    filtered_count += 1
                    
                    # winner perspective
                    rank_diff_w = l_rank['position'] - w_rank['position']
                    points_ratio_w = np.log(w_rank['points'] / l_rank['points'])
                    X.append([rank_diff_w, points_ratio_w])
                    y.append(1)
                    
                    # loser perspective
                    rank_diff_l = w_rank['position'] - l_rank['position']
                    points_ratio_l = np.log(l_rank['points'] / w_rank['points'])
                    X.append([rank_diff_l, points_ratio_l])
                    y.append(0)
        
        print(f"Found {filtered_count} matches with both players in top {max_rank}")
        
        if len(X) > 100:
            self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
            self.model.fit(X, y)
            self.position_coef = self.model.coef_[0][0]
            self.points_coef = self.model.coef_[0][1]
            
            print(f"Joint model coefficients:")
            print(f"  Position coefficient: {self.position_coef:.6f}")
            print(f"  Points coefficient: {self.points_coef:.6f}")
            print(f"  Intercept: {self.model.intercept_[0]:.6f}")
        else:
            print("Insufficient data for model fitting")
    
    def predict_match(self, player1_name, player2_name, match_date, verbose=True):
        #predict using joint model
        if self.model is None:
            self.fit_model(before_date=match_date)
        
        p1_id, p1_full = self.find_player(player1_name)
        p2_id, p2_full = self.find_player(player2_name)
        
        if not p1_id or not p2_id:
            if verbose: print("Could not find one or both players")
            return None
        
        p1_rank = self.get_ranking_at_date(p1_id, match_date)
        p2_rank = self.get_ranking_at_date(p2_id, match_date)
        
        if not p1_rank or not p2_rank or p1_rank['points'] <= 0 or p2_rank['points'] <= 0:
            if verbose: print(f"Insufficient ranking data")
            return None
        
        # calc features
        rank_diff = p2_rank['position'] - p1_rank['position']
        points_ratio = np.log(p1_rank['points'] / p2_rank['points'])
        
        # predict probability
        features = [[rank_diff, points_ratio]]
        prob1 = self.model.predict_proba(features)[0][1]
        
        if verbose:
            print(f"\nJOINT MODEL PREDICTION")
            print(f"{p1_full} vs {p2_full}")
            print(f"  Ranking: #{p1_rank['position']} vs #{p2_rank['position']} (diff: {rank_diff:+d})")
            print(f"  Points: {p1_rank['points']:.0f} vs {p2_rank['points']:.0f} (log ratio: {points_ratio:.3f})")
            print(f"\nProbability {p1_full} wins: {prob1:.1%}")
        
        return {
            'player1_name': p1_full,
            'player2_name': p2_full,
            'player1_prob': prob1,
            'player2_prob': 1 - prob1,
            'player1_rank': p1_rank['position'],
            'player2_rank': p2_rank['position'],
            'player1_points': p1_rank['points'],
            'player2_points': p2_rank['points']
        }
    
    def batch_evaluate(self, year, surface='overall', tournament=None, round=None, output_file=None):
        # batch eval joint model w/ specified filters
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
        
        print(f"\nBatch Evaluating Rankings JOINT Model:")
        print(f"  Year: {year}")
        print(f"  Surface: {surface}")
        if tournament: print(f"  Tournament: {tournament}")
        if round: print(f"  Round: {round}")
        
        # build query with filters
        filters = [
            f"t.date >= {start_date}",
            f"t.date <= {end_date}",
            "NOT m.score CONTAINS 'W/O'"]
        
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
            w_rank = self.get_ranking_at_date(match['winner_id'], match['match_date'])
            l_rank = self.get_ranking_at_date(match['loser_id'], match['match_date'])
            
            if not w_rank or not l_rank or w_rank['points'] <= 0 or l_rank['points'] <= 0:
                skipped += 1
                continue
            
            rank_diff = l_rank['position'] - w_rank['position']
            points_ratio = np.log(w_rank['points'] / l_rank['points'])
            
            features = [[rank_diff, points_ratio]]
            w_prob = self.model.predict_proba(features)[0][1]
            
            results.append({
                # standard cols
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
                
                # joint model specific cols
                'player1_rank': w_rank['position'],
                'player2_rank': l_rank['position'],
                'rank_diff': rank_diff,
                'player1_points': w_rank['points'],
                'player2_points': l_rank['points'],
                'log_points_ratio': points_ratio
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
            
            #ensure output file is in model_outputs directory
            if not output_file.startswith(self.output_dir):
                output_file = os.path.join(self.output_dir, os.path.basename(output_file))
            
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            return df
        else:
            print("No valid predictions could be made!")
            return None
    
    def generate_output_filename(self, year, surface='overall', tournament=None, round=None):
        # generate standardized filename in model_outputs/
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
    model = RankingJointModel()
    
    try:
        if '--batch-evaluate' in sys.argv:
            # batch evaluation mode
            if '--year' not in sys.argv:
                print("Error: --year is required for batch evaluation")
                print("Usage: python rankings_joint.py --batch-evaluate --year YYYY [options]")
                return
            
            # parse year
            year_idx = sys.argv.index('--year')
            year = int(sys.argv[year_idx + 1])
            
            # parse optional params
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
            print("    python rankings_joint.py --predict 'Novak Djokovic' 'Rafael Nadal' 20230601")
            print("\n  Batch evaluation:")
            print("    python rankings_joint.py --batch-evaluate --year 2024")
            print("    python rankings_joint.py --batch-evaluate --year 2024 --surface clay")
            print("    python rankings_joint.py --batch-evaluate --year 2024 --tournament Wimbledon")
            print("    python rankings_joint.py --batch-evaluate --year 2024 --round F")
    
    finally:
        model.close()

if __name__ == "__main__":
    main()