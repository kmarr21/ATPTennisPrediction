#!/usr/bin/env python3
"""
Bradley-Terry Tennis Prediction Model
======================================

Uses pre-computed Bradley-Terry strength nodes for fast predictions.
Based on McHale & Morton (2011) Bradley-Terry type model.

Usage:
    # Single match prediction
    python bt.py "Novak Djokovic" "Rafael Nadal" 20230601 --surface clay
    
    # Batch evaluation
    python bt.py --batch-evaluate --year 2024
    python bt.py --batch-evaluate --year 2024 --surface clay
    python bt.py --batch-evaluate --year 2024 --tournament Wimbledon
    python bt.py --batch-evaluate --year 2024 --round F
    
    # Interactive mode
    python bt.py
"""

import sys
import os
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Neo4j connection parameters
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "RolandGarros2195!"

class BradleyTerryModel:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.model_name = "bt"
        
        print("=" * 60)
        print("BRADLEY-TERRY PREDICTION MODEL")
        print("Using pre-computed strength nodes")
        print("=" * 60)
        print("Connected to Neo4j database.")
        
        # Create output directory
        self.output_dir = "model_outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created {self.output_dir}/ directory")
    
    def close(self):
        self.driver.close()
    
    def find_player(self, player_name):
        """Find player by name"""
        with self.driver.session() as session:
            # Exact match
            query = """
            MATCH (p:Player)
            WHERE toLower(p.first_name + ' ' + p.last_name) = toLower($name)
            RETURN p.id as id, p.first_name + ' ' + p.last_name as full_name
            """
            result = session.run(query, name=player_name)
            match = result.single()
            
            if match:
                return match['id'], match['full_name']
            
            # Partial match
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
                for i, m in enumerate(matches, 1):
                    print(f"{i}. {m['full_name']}")
                try:
                    choice = int(input("Select player (0 to cancel): "))
                    if 1 <= choice <= len(matches):
                        return matches[choice-1]['id'], matches[choice-1]['full_name']
                except:
                    pass
            
            return None, None
    
    def get_bt_strengths_at_date(self, player_id, match_date):
        """Get Bradley-Terry strengths for a player at or before specified date"""
        with self.driver.session() as session:
            query = """
            MATCH (p:Player {id: $player_id})-[:HAS_BT]->(bt:BradleyTerry)
            WHERE bt.date <= $match_date
            WITH bt
            ORDER BY bt.date DESC
            LIMIT 1
            RETURN 
                bt.strength_overall as overall,
                bt.strength_hard as hard,
                bt.strength_clay as clay,
                bt.strength_grass as grass,
                bt.date as strength_date
            """
            
            result = session.run(query, player_id=player_id, match_date=match_date)
            return result.single()
    
    def get_surface_strength(self, bt_data, surface):
        """Get appropriate strength based on surface"""
        if not bt_data:
            return 1.0  # Default strength if no data
        
        surface_map = {
            'overall': 'overall',
            'hard': 'hard',
            'carpet': 'hard',  # Carpet treated as hard
            'clay': 'clay',
            'grass': 'grass'
        }
        
        surface_key = surface_map.get(surface.lower() if surface else 'overall', 'overall')
        
        if surface_key == 'overall':
            return bt_data['overall']
        else:
            return bt_data[surface_key]
    
    def predict_match(self, player1_name, player2_name, match_date, 
                     surface='overall', verbose=True, return_strengths=False):
        """Predict match outcome using pre-computed Bradley-Terry strengths"""
        if verbose:
            print(f"\nBradley-Terry Prediction: {player1_name} vs {player2_name}")
            print(f"Date: {match_date}, Surface: {surface}")
            print("-" * 50)
        
        # Find players
        p1_id, p1_full = self.find_player(player1_name)
        p2_id, p2_full = self.find_player(player2_name)
        
        if not p1_id or not p2_id:
            if verbose:
                print("Could not find one or both players")
            return None
        
        # Get BT strengths from nodes
        p1_bt = self.get_bt_strengths_at_date(p1_id, match_date)
        p2_bt = self.get_bt_strengths_at_date(p2_id, match_date)
        
        if not p1_bt or not p2_bt:
            if verbose:
                print("No Bradley-Terry data available for one or both players")
            return None
        
        # Get appropriate surface strengths
        alpha1 = self.get_surface_strength(p1_bt, surface)
        alpha2 = self.get_surface_strength(p2_bt, surface)
        
        # Calculate win probability: P(1 wins) = alpha1 / (alpha1 + alpha2)
        prob1 = alpha1 / (alpha1 + alpha2)
        prob2 = 1 - prob1
        
        result = {
            'player1_name': p1_full,
            'player2_name': p2_full,
            'player1_prob': prob1,
            'player2_prob': prob2,
            'match_date': match_date,
            'surface': surface
        }
        
        if return_strengths:
            result['strength1'] = alpha1
            result['strength2'] = alpha2
            result['strength_date'] = max(p1_bt['strength_date'], p2_bt['strength_date'])
        
        if verbose:
            print(f"\nPREDICTION RESULTS")
            print("=" * 40)
            print(f"{p1_full}: {prob1:.1%}")
            print(f"{p2_full}: {prob2:.1%}")
            print(f"\nStrength parameters ({surface}):")
            print(f"  {p1_full}: {alpha1:.3f}")
            print(f"  {p2_full}: {alpha2:.3f}")
            if p1_bt and p2_bt:
                print(f"\nStrengths from: {max(p1_bt['strength_date'], p2_bt['strength_date'])}")
        
        return result
    
    def batch_evaluate(self, year, surface='overall', tournament=None,
                      round=None, output_file=None):
        """
        Batch evaluate Bradley-Terry model with specified filters
        
        Args:
            year: Year to evaluate (required)
            surface: Surface filter ('overall', 'hard', 'clay', 'grass')
            tournament: Tournament name to filter (partial match, optional)
            round: Round to filter (e.g., 'F', 'SF', 'QF', optional)
            output_file: Explicit output path, or None for auto-generate
        
        Returns:
            DataFrame with comprehensive results
        """
        start_date = year * 10000 + 101
        end_date = year * 10000 + 1231
        
        print(f"\nBatch Evaluating Bradley-Terry Model:")
        print(f"  Year: {year}")
        print(f"  Surface: {surface}")
        if tournament:
            print(f"  Tournament: {tournament}")
        if round:
            print(f"  Round: {round}")
        
        # Build query with filters
        filters = [
            f"t.date >= {start_date}",
            f"t.date <= {end_date}",
            "m.score IS NOT NULL",
            "NOT m.score CONTAINS 'W/O'"
        ]
        
        if surface != 'overall':
            filters.append(f"toLower(t.surface) = '{surface.lower()}'")
        
        if tournament:
            filters.append(f"toLower(t.name) CONTAINS toLower('{tournament}')")
        
        if round:
            filters.append(f"m.round = '{round}'")
        
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
                t.surface as actual_surface,
                t.name as tournament,
                t.level as tournament_level,
                m.round as round,
                m.score as score
            ORDER BY t.date, m.match_num
            """
            
            result = session.run(query)
            test_matches = list(result)
        
        print(f"Found {len(test_matches)} matches to evaluate...")
        
        if not test_matches:
            print("No matches found!")
            return None
        
        # Evaluate
        results = []
        predictions_made = 0
        
        for match in tqdm(test_matches, desc="Evaluating"):
            # Get strengths from BT nodes
            w_bt = self.get_bt_strengths_at_date(match['winner_id'], match['match_date'])
            l_bt = self.get_bt_strengths_at_date(match['loser_id'], match['match_date'])
            
            if not w_bt or not l_bt:
                continue  # Skip if no BT data
            
            # Get surface strengths
            eval_surface = match['actual_surface'] if surface != 'overall' else surface
            w_strength = self.get_surface_strength(w_bt, eval_surface)
            l_strength = self.get_surface_strength(l_bt, eval_surface)
            
            # Calculate probability
            w_prob = w_strength / (w_strength + l_strength)
            
            predictions_made += 1
            
            results.append({
                # Required columns (all models)
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
                'surface': match['actual_surface'],
                'tournament': match['tournament'],
                'tournament_level': match['tournament_level'],
                'round': match['round'],
                
                # Bradley-Terry specific columns
                'strength1': w_strength,
                'strength2': l_strength,
                'strength_ratio': w_strength / l_strength if l_strength > 0 else None
            })
        
        # Calculate metrics
        if results:
            df = pd.DataFrame(results)
            
            accuracy = df['correct_prediction'].mean()
            avg_brier = df['brier_score'].mean()
            coverage = predictions_made / len(test_matches)
            
            print(f"\nEVALUATION RESULTS")
            print("=" * 50)
            print(f"Total matches: {len(test_matches)}")
            print(f"Predictions made: {predictions_made}")
            print(f"Coverage: {coverage:.1%}")
            print(f"Accuracy: {accuracy:.1%}")
            print(f"Average Brier Score: {avg_brier:.3f}")
            
            # Additional breakdowns
            if len(df) > 100:
                print(f"\nAccuracy by Strength Ratio:")
                df['log_strength_ratio'] = np.log(df['strength_ratio'])
                
                for threshold in [0.1, 0.2, 0.3, 0.5]:
                    subset = df[df['log_strength_ratio'].abs() <= threshold]
                    if len(subset) > 20:
                        acc = subset['correct_prediction'].mean()
                        print(f"  log ratio ±{threshold}: {acc:.1%} ({len(subset)} matches)")
                
                if tournament is None:
                    print(f"\nAccuracy by Tournament Level:")
                    for level in ['G', 'M', 'A']:
                        subset = df[df['tournament_level'] == level]
                        if len(subset) > 20:
                            acc = subset['correct_prediction'].mean()
                            print(f"  Level {level}: {acc:.1%} ({len(subset)} matches)")
            
            # Generate output filename if not specified
            if not output_file:
                output_file = self.generate_output_filename(
                    year=year, surface=surface, tournament=tournament, round=round
                )
            
            # Ensure output file is in model_outputs directory
            if not output_file.startswith(self.output_dir):
                output_file = os.path.join(self.output_dir, os.path.basename(output_file))
            
            # Save results
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            return df
        else:
            print("No predictions could be made!")
            return None
    
    def generate_output_filename(self, year, surface='overall',
                                tournament=None, round=None):
        """
        Generate standardized filename in model_outputs/
        
        Pattern: {model}_{tournament}_{round}_{year}_{surface}_{timestamp}.csv
        """
        parts = [self.model_name]
        
        if tournament:
            parts.append(tournament.replace(' ', '_'))
        
        if round:
            parts.append(round)
        
        parts.append(str(year))
        
        if surface != 'overall':
            parts.append(surface)
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parts.append(timestamp)
        
        # Combine and add extension
        filename = '_'.join(parts) + '.csv'
        
        return os.path.join(self.output_dir, filename)
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print("\nInteractive Bradley-Terry Prediction Mode")
        print("Enter 'quit' to exit")
        print("-" * 40)
        
        while True:
            try:
                player1 = input("\nEnter Player 1 name: ").strip()
                if player1.lower() == 'quit':
                    break
                
                player2 = input("Enter Player 2 name: ").strip()
                if player2.lower() == 'quit':
                    break
                
                date_input = input("Enter match date (YYYYMMDD): ").strip()
                if not date_input:
                    print("Date is required!")
                    continue
                
                try:
                    match_date = int(date_input)
                except ValueError:
                    print("Invalid date format!")
                    continue
                
                surface = input("Enter surface (overall/hard/clay/grass) [overall]: ").strip()
                if not surface:
                    surface = 'overall'
                
                self.predict_match(player1, player2, match_date, surface=surface)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    model = BradleyTerryModel()
    
    try:
        if len(sys.argv) >= 4 and not sys.argv[1].startswith('--'):
            # Single match prediction
            surface = 'overall'
            if '--surface' in sys.argv:
                idx = sys.argv.index('--surface')
                if idx + 1 < len(sys.argv):
                    surface = sys.argv[idx + 1]
            
            model.predict_match(sys.argv[1], sys.argv[2], int(sys.argv[3]), surface=surface)
        
        elif '--batch-evaluate' in sys.argv:
            # Batch evaluation mode
            if '--year' not in sys.argv:
                print("Error: --year is required for batch evaluation")
                print("Usage: python bt.py --batch-evaluate --year YYYY [options]")
                return
            
            # Parse year
            year_idx = sys.argv.index('--year')
            year = int(sys.argv[year_idx + 1])
            
            # Parse optional parameters
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
            
            # Run batch evaluation
            model.batch_evaluate(
                year=year,
                surface=surface,
                tournament=tournament,
                round=round_filter,
                output_file=output_file
            )
        
        else:
            # Interactive mode or show usage
            if len(sys.argv) == 1:
                model.interactive_mode()
            else:
                print("Usage examples:")
                print("\n  Single prediction:")
                print("    python bt.py 'Novak Djokovic' 'Rafael Nadal' 20230601")
                print("    python bt.py 'Novak Djokovic' 'Rafael Nadal' 20230601 --surface clay")
                print("\n  Batch evaluation:")
                print("    python bt.py --batch-evaluate --year 2024")
                print("    python bt.py --batch-evaluate --year 2024 --surface clay")
                print("    python bt.py --batch-evaluate --year 2024 --tournament Wimbledon")
                print("    python bt.py --batch-evaluate --year 2024 --round F")
    
    finally:
        model.close()

if __name__ == "__main__":
    main()