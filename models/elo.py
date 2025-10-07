#!/usr/bin/env python3
"""
Tennis Abstract ELO Prediction Model
=====================================

Predicts match outcomes using Tennis Abstract ELO ratings.
Uses weekly ELO snapshots for temporal validation.

Usage:
    # Single match prediction
    python elo.py "Novak Djokovic" "Rafael Nadal" 20230601 --surface clay
    
    # Batch evaluation
    python elo.py --batch-evaluate --year 2024
    python elo.py --batch-evaluate --year 2024 --surface clay
    python elo.py --batch-evaluate --year 2024 --tournament Wimbledon
    python elo.py --batch-evaluate --year 2024 --round F
    
    # Interactive mode
    python elo.py
"""

import sys
import os
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Neo4j connection parameters
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "RolandGarros2195!"

# ELO stabilization parameters
ELO_START_DATE = 20000101  # When ELO ratings begin
ELO_WARNING_DATE = 20020101  # Warn before this date (2 years)
ELO_MINIMUM_DATE = 20010101  # Hard minimum (1 year)

class EloModel:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.model_name = "elo"
        
        print("=" * 60)
        print("TENNIS ABSTRACT ELO PREDICTION MODEL")
        print("=" * 60)
        print("Connected to Neo4j database.")
        
        # Create output directory if needed
        self.output_dir = "model_outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created {self.output_dir}/ directory")
    
    def close(self):
        self.driver.close()
    
    def check_date_validity(self, match_date, force=False):
        """
        Check if date is valid for ELO predictions
        Returns: True if valid, False if should abort
        """
        if match_date < ELO_MINIMUM_DATE:
            print(f"\n⚠️  ERROR: Cannot make predictions before {ELO_MINIMUM_DATE}")
            print(f"ELO ratings need at least 1 year to stabilize (started {ELO_START_DATE})")
            return False
        
        if match_date < ELO_WARNING_DATE and not force:
            print(f"\n⚠️  WARNING: Match date {match_date} is very early in ELO history")
            print(f"Ratings may not be stable (system started {ELO_START_DATE})")
            response = input("Continue anyway? (yes/no): ").strip().lower()
            return response == 'yes'
        
        return True
    
    def find_player(self, player_name):
        """Find player by name (fuzzy matching)"""
        with self.driver.session() as session:
            # First try exact match
            exact_query = """
            MATCH (p:Player)
            WHERE toLower(p.first_name + ' ' + p.last_name) = toLower($name)
            RETURN p.id as id, p.first_name + ' ' + p.last_name as full_name
            """
            result = session.run(exact_query, name=player_name)
            exact_match = result.single()
            
            if exact_match:
                return exact_match['id'], exact_match['full_name']
            
            # Try partial match on last name
            partial_query = """
            MATCH (p:Player)
            WHERE toLower(p.last_name) CONTAINS toLower($name) 
               OR toLower(p.first_name + ' ' + p.last_name) CONTAINS toLower($name)
            RETURN p.id as id, p.first_name + ' ' + p.last_name as full_name
            ORDER BY p.last_name
            LIMIT 10
            """
            result = session.run(partial_query, name=player_name)
            matches = list(result)
            
            if matches:
                print(f"\nDid you mean one of these players?")
                for i, match in enumerate(matches, 1):
                    print(f"{i}. {match['full_name']}")
                
                try:
                    choice = int(input("\nEnter number (or 0 to cancel): "))
                    if 1 <= choice <= len(matches):
                        selected = matches[choice - 1]
                        return selected['id'], selected['full_name']
                except:
                    pass
            
            return None, None
    
    def get_elo_at_date(self, player_id, match_date):
        """Get player's ELO ratings at or before specified date"""
        with self.driver.session() as session:
            # Get most recent ELO before match date
            elo_query = """
            MATCH (p:Player {id: $player_id})-[:HAS_ELO]->(e:ELO)
            WHERE e.date <= $match_date
            WITH e
            ORDER BY e.date DESC
            LIMIT 1
            RETURN 
                e.overall as overall,
                e.hard as hard,
                e.clay as clay,
                e.grass as grass,
                e.date as rating_date,
                e.matches_played as matches_played,
                e.last_match_date as last_match_date
            """
            
            result = session.run(elo_query, player_id=player_id, match_date=match_date)
            return result.single()
    
    def calculate_elo_probability(self, rating_a, rating_b):
        """
        Calculate win probability using Tennis Abstract ELO formula
        P(A wins) = 1 / (1 + 10^((rating_b - rating_a) / 400))
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def get_surface_rating(self, elo_data, surface_type):
        """
        Get appropriate rating based on surface type
        Tennis Abstract uses 50/50 blend of overall and surface-specific
        """
        if not elo_data:
            return None
            
        if surface_type == 'overall':
            return elo_data['overall']
        
        # Map surface types
        surface_map = {
            'hard': 'hard',
            'clay': 'clay', 
            'grass': 'grass',
            'carpet': 'hard'  # Treat carpet as hard
        }
        
        surface_key = surface_map.get(surface_type.lower(), 'hard')
        
        # Tennis Abstract: 50/50 blend for surface predictions
        overall_rating = elo_data['overall']
        surface_rating = elo_data[surface_key]
        
        # Return blended rating
        return 0.5 * overall_rating + 0.5 * surface_rating
    
    def predict_match(self, player1_name, player2_name, match_date, 
                     surface='overall', verbose=True, return_ratings=False, force=False):
        """
        Predict match outcome using ELO ratings at specified date
        """
        # Check date validity
        if not self.check_date_validity(match_date, force):
            return None
        
        if verbose:
            print(f"\nELO Prediction: {player1_name} vs {player2_name}")
            print(f"Match Date: {match_date}")
            print(f"Surface: {surface}")
            print("-" * 50)
        
        # Find players
        p1_id, p1_full_name = self.find_player(player1_name)
        if not p1_id:
            if verbose:
                print(f"Player '{player1_name}' not found.")
            return None
        
        p2_id, p2_full_name = self.find_player(player2_name)
        if not p2_id:
            if verbose:
                print(f"Player '{player2_name}' not found.")
            return None
        
        if verbose:
            print(f"Found: {p1_full_name} vs {p2_full_name}")
        
        # Get ELO ratings at match date
        p1_elo = self.get_elo_at_date(p1_id, match_date)
        p2_elo = self.get_elo_at_date(p2_id, match_date)
        
        if not p1_elo:
            if verbose:
                print(f"No ELO data for {p1_full_name} before {match_date}")
            return None
        
        if not p2_elo:
            if verbose:
                print(f"No ELO data for {p2_full_name} before {match_date}")
            return None
        
        # Get appropriate ratings based on surface
        p1_rating = self.get_surface_rating(p1_elo, surface)
        p2_rating = self.get_surface_rating(p2_elo, surface)
        
        # Calculate probabilities
        p1_probability = self.calculate_elo_probability(p1_rating, p2_rating)
        p2_probability = 1 - p1_probability
        
        # Prepare result
        result = {
            'player1_name': p1_full_name,
            'player2_name': p2_full_name,
            'player1_prob': p1_probability,
            'player2_prob': p2_probability,
            'match_date': match_date,
            'surface': surface,
            'rating_difference': p1_rating - p2_rating,
            'p1_matches_played': p1_elo['matches_played'],
            'p2_matches_played': p2_elo['matches_played'],
            'rating_date': max(p1_elo['rating_date'], p2_elo['rating_date'])
        }
        
        if return_ratings:
            result['p1_rating'] = p1_rating
            result['p2_rating'] = p2_rating
            result['p1_elo_data'] = dict(p1_elo)
            result['p2_elo_data'] = dict(p2_elo)
        
        if verbose:
            print(f"\nELO PREDICTION RESULTS")
            print("=" * 40)
            print(f"{p1_full_name}: {p1_probability:.1%}")
            print(f"{p2_full_name}: {p2_probability:.1%}")
            print("")
            print(f"ELO Ratings ({surface}):")
            print(f"  {p1_full_name}: {p1_rating:.1f}")
            print(f"  {p2_full_name}: {p2_rating:.1f}")
            print(f"  Difference: {p1_rating - p2_rating:+.1f}")
            print("")
            print(f"Ratings from: {result['rating_date']}")
            print(f"Match experience: {p1_elo['matches_played']} vs {p2_elo['matches_played']} matches")
        
        return result
    
    def batch_evaluate(self, year, surface='overall', tournament=None,
                      round=None, output_file=None):
        """
        Batch evaluate ELO model with specified filters
        
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
        
        print(f"\nBatch Evaluating ELO Model:")
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
            eval_query = f"""
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
                m.round as round
            ORDER BY t.date, m.match_num
            """
            
            result = session.run(eval_query)
            test_matches = list(result)
        
        print(f"Found {len(test_matches)} matches to evaluate...")
        
        # Evaluate matches
        results = []
        predictions_made = 0
        
        for i, match in enumerate(tqdm(test_matches, desc="Evaluating")):
            # Use actual surface for prediction
            eval_surface = match['actual_surface'] if surface != 'overall' else 'overall'
            
            # Get prediction
            prediction = self.predict_match(
                match['winner_name'],
                match['loser_name'], 
                match['match_date'],
                surface=eval_surface,
                verbose=False,
                return_ratings=True,
                force=True  # Skip date warnings for batch
            )
            
            if prediction:
                predictions_made += 1
                
                # Store results in standardized format
                result_row = {
                    # Required columns (all models)
                    'match_id': match['match_id'],
                    'match_date': match['match_date'],
                    'player1_name': match['winner_name'],
                    'player2_name': match['loser_name'],
                    'actual_winner': 1,
                    'predicted_prob_p1': prediction['player1_prob'],
                    'predicted_prob_p2': prediction['player2_prob'],
                    'predicted_winner': 1 if prediction['player1_prob'] > 0.5 else 2,
                    'correct_prediction': 1 if prediction['player1_prob'] > 0.5 else 0,
                    'brier_score': (prediction['player1_prob'] - 1) ** 2,
                    'surface': match['actual_surface'],
                    'tournament': match['tournament'],
                    'tournament_level': match['tournament_level'],
                    'round': match['round'],
                    
                    # ELO specific columns
                    'p1_rating': prediction.get('p1_rating', None),
                    'p2_rating': prediction.get('p2_rating', None),
                    'rating_diff': prediction['rating_difference']
                }
                
                results.append(result_row)
        
        # Calculate statistics
        if results:
            df = pd.DataFrame(results)
            
            # Calculate metrics
            accuracy = df['correct_prediction'].mean()
            avg_brier = df['brier_score'].mean()
            coverage = len(results) / len(test_matches) if test_matches else 0
            
            print(f"\nEVALUATION RESULTS")
            print("=" * 50)
            print(f"Total matches: {len(test_matches)}")
            print(f"Predictions made: {len(results)}")
            print(f"Coverage: {coverage:.1%}")
            print(f"Accuracy: {accuracy:.1%}")
            print(f"Average Brier Score: {avg_brier:.3f}")
            
            # Additional breakdowns
            if len(df) > 100:
                print(f"\nAccuracy by Rating Difference:")
                for threshold in [50, 100, 150, 200]:
                    subset = df[df['rating_diff'].abs() <= threshold]
                    if len(subset) > 20:
                        acc = subset['correct_prediction'].mean()
                        print(f"  ±{threshold} points: {acc:.1%} ({len(subset)} matches)")
                
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
        """Interactive mode for testing predictions"""
        print("\nInteractive ELO Prediction Mode")
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
                
                # Match date
                match_date_input = input("Enter match date (YYYYMMDD): ").strip()
                if not match_date_input:
                    print("Match date is required!")
                    continue
                
                try:
                    match_date = int(match_date_input)
                except ValueError:
                    print("Invalid date format. Use YYYYMMDD")
                    continue
                
                # Surface (optional)
                surface = input("Enter surface (overall/hard/clay/grass) [overall]: ").strip()
                if not surface:
                    surface = 'overall'
                
                self.predict_match(player1, player2, match_date, surface=surface, verbose=True)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    model = EloModel()
    
    try:
        # Parse command line arguments
        if len(sys.argv) >= 4 and not sys.argv[1].startswith('--'):
            # Single match prediction: player1 player2 date [--surface type]
            surface = 'overall'
            if '--surface' in sys.argv:
                idx = sys.argv.index('--surface')
                if idx + 1 < len(sys.argv):
                    surface = sys.argv[idx + 1]
            
            result = model.predict_match(sys.argv[1], sys.argv[2], int(sys.argv[3]), surface=surface)
        
        elif '--batch-evaluate' in sys.argv:
            # Batch evaluation mode
            if '--year' not in sys.argv:
                print("Error: --year is required for batch evaluation")
                print("Usage: python elo.py --batch-evaluate --year YYYY [options]")
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
                print("    python elo.py 'Novak Djokovic' 'Rafael Nadal' 20230601")
                print("    python elo.py 'Novak Djokovic' 'Rafael Nadal' 20230601 --surface clay")
                print("\n  Batch evaluation:")
                print("    python elo.py --batch-evaluate --year 2024")
                print("    python elo.py --batch-evaluate --year 2024 --surface clay")
                print("    python elo.py --batch-evaluate --year 2024 --tournament Wimbledon")
                print("    python elo.py --batch-evaluate --year 2024 --round F")
    
    finally:
        model.close()

if __name__ == "__main__":
    main()