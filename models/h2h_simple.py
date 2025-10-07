#!/usr/bin/env python3
"""
Head-to-Head Tennis Prediction Model
========================================

Predicts match outcomes based on historical head-to-head records.
Uses Beta distribution for confidence intervals.

Usage:
    # Single match prediction
    python h2h_simple.py "Novak Djokovic" "Rafael Nadal" 20230601
    
    # Batch evaluation modes
    python h2h_simple.py --batch-evaluate --year 2024                      # All 2024 matches
    python h2h_simple.py --batch-evaluate --year 2024 --surface clay       # 2024 clay only
    python h2h_simple.py --batch-evaluate --year 2024 --tournament Wimbledon  # Wimbledon 2024
    python h2h_simple.py --batch-evaluate --year 2024 --round F            # 2024 finals only
    python h2h_simple.py --batch-evaluate --year 2024 --output my_results.csv  # Custom output
    
    # Interactive mode
    python h2h_simple.py
"""

import sys
import os
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Neo4j connection parameters
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "RolandGarros2195!"

class H2HModel:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.model_name = "h2h"
        
        print("=" * 60)
        print("HEAD-TO-HEAD PREDICTION MODEL")
        print("=" * 60)
        print("Connected to Neo4j database.")
        
        # Create output directory if needed
        self.output_dir = "model_outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created {self.output_dir}/ directory")
    
    def close(self):
        self.driver.close()
    
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
                    else:
                        return None, None
                except ValueError:
                    return None, None
            
            return None, None
    
    def get_h2h_record(self, player1_id, player2_id, before_date):
        """Get head-to-head record between two players before specified date"""
        with self.driver.session() as session:
            h2h_query = """
            MATCH (p1:Player {id: $player1_id})-[r1:WON|LOST]->(m:Match)<-[r2:WON|LOST]-(p2:Player {id: $player2_id})
            MATCH (m)-[:PLAYED_IN]->(t:Tournament)
            WHERE t.date < $before_date
            RETURN 
                CASE WHEN type(r1) = 'WON' THEN 1 ELSE 0 END as p1_won,
                t.date as match_date,
                t.name as tournament,
                m.round as round,
                m.score as score,
                t.surface as surface
            ORDER BY t.date
            """
            
            result = session.run(h2h_query, 
                               player1_id=player1_id, 
                               player2_id=player2_id,
                               before_date=before_date)
            
            matches = list(result)
            return matches
    
    def calculate_h2h_probability(self, h2h_matches):
        """Calculate win probability and confidence interval based on H2H record"""
        if not h2h_matches:
            return None
        
        total_matches = len(h2h_matches)
        player1_wins = sum(match['p1_won'] for match in h2h_matches)
        
        # Use Beta distribution for confidence intervals
        # Beta(wins + 1, losses + 1) is conjugate prior for binomial
        alpha = player1_wins + 1
        beta = (total_matches - player1_wins) + 1
        
        # Point estimate (mean of Beta distribution)
        p1_probability = alpha / (alpha + beta)
        p2_probability = 1 - p1_probability
        
        # 95% confidence interval
        ci_lower = stats.beta.ppf(0.025, alpha, beta)
        ci_upper = stats.beta.ppf(0.975, alpha, beta)
        
        # Mathematical confidence score (inverse of CI width)
        ci_width = ci_upper - ci_lower
        confidence_score = 1 - ci_width  # Score from 0 to 1
        
        # Standard error for probability
        std_error = math.sqrt(p1_probability * (1 - p1_probability) / (total_matches + 2))
        
        return {
            'player1_prob': p1_probability,
            'player2_prob': p2_probability,
            'player1_wins': player1_wins,
            'player2_wins': total_matches - player1_wins,
            'total_matches': total_matches,
            'confidence_score': confidence_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std_error': std_error,
            'recent_matches': h2h_matches[-3:] if len(h2h_matches) >= 3 else h2h_matches
        }
    
    def predict_match(self, player1_name, player2_name, match_date, verbose=True):
        """
        Predict match outcome based on H2H before the specified date
        
        Args:
            player1_name: Name of player 1
            player2_name: Name of player 2  
            match_date: Date of match (YYYYMMDD format)
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with prediction results or None if no H2H history
        """
        if verbose:
            print(f"\nAnalyzing H2H: {player1_name} vs {player2_name}")
            print(f"Match Date: {match_date}")
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
        
        # Get H2H record before match date
        h2h_matches = self.get_h2h_record(p1_id, p2_id, match_date)
        
        # Calculate probabilities
        result = self.calculate_h2h_probability(h2h_matches)
        
        if result is None:
            if verbose:
                print(f"No head-to-head history between {p1_full_name} and {p2_full_name} before {match_date}")
            return None
        
        # Add player names to result
        result['player1_name'] = p1_full_name
        result['player2_name'] = p2_full_name
        result['match_date'] = match_date
        
        if verbose:
            # Format output
            print(f"\nHEAD-TO-HEAD PREDICTION")
            print("=" * 40)
            print(f"{p1_full_name}: {result['player1_prob']:.1%}")
            print(f"{p2_full_name}: {result['player2_prob']:.1%}")
            print("")
            print(f"Historical Record: {result['player1_wins']}-{result['player2_wins']} ({result['total_matches']} matches)")
            print(f"Confidence Score: {result['confidence_score']:.3f}")
            print(f"95% CI: [{result['ci_lower']:.1%}, {result['ci_upper']:.1%}]")
            print(f"Standard Error: ±{result['std_error']:.3f}")
            
            if result['recent_matches']:
                print("\nRecent H2H Matches:")
                for match in reversed(result['recent_matches']):
                    winner = p1_full_name if match['p1_won'] else p2_full_name
                    print(f"  {match['match_date']}: {winner} won {match['score']} at {match['tournament']} ({match['surface']})")
        
        return result
    
    def batch_evaluate(self, year, surface='overall', tournament=None, 
                      round=None, output_file=None):
        """
        Batch evaluate H2H model with specified filters
        
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
        
        print(f"\nBatch Evaluating H2H Model:")
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
                t.surface as surface,
                t.name as tournament,
                t.level as tournament_level,
                m.round as round
            ORDER BY t.date, m.match_num
            """
            
            result = session.run(eval_query)
            test_matches = list(result)
        
        print(f"Found {len(test_matches)} matches to evaluate...")
        
        # Prepare results storage
        results = []
        predictions_made = 0
        
        for i, match in enumerate(tqdm(test_matches, desc="Evaluating")):
            # Get H2H record before this match
            h2h_matches = self.get_h2h_record(
                match['winner_id'],
                match['loser_id'],
                match['match_date']
            )
            
            # Calculate prediction
            h2h_result = self.calculate_h2h_probability(h2h_matches)
            
            if h2h_result:  # Only record if there's H2H history
                predictions_made += 1
                
                # Store detailed results in standardized format
                result_row = {
                    # Required columns (all models)
                    'match_id': match['match_id'],
                    'match_date': match['match_date'],
                    'player1_name': match['winner_name'],
                    'player2_name': match['loser_name'],
                    'actual_winner': 1,  # Winner is always player1 in our query
                    'predicted_prob_p1': h2h_result['player1_prob'],
                    'predicted_prob_p2': h2h_result['player2_prob'],
                    'predicted_winner': 1 if h2h_result['player1_prob'] > 0.5 else 2,
                    'correct_prediction': 1 if h2h_result['player1_prob'] > 0.5 else 0,
                    'brier_score': (h2h_result['player1_prob'] - 1) ** 2,
                    'surface': match['surface'],
                    'tournament': match['tournament'],
                    'tournament_level': match['tournament_level'],
                    'round': match['round'],
                    
                    # H2H-specific columns
                    'h2h_matches': h2h_result['total_matches'],
                    'h2h_record': f"{h2h_result['player1_wins']}-{h2h_result['player2_wins']}",
                    'confidence_score': h2h_result['confidence_score'],
                    'ci_lower': h2h_result['ci_lower'],
                    'ci_upper': h2h_result['ci_upper']
                }
                
                results.append(result_row)
        
        # Create DataFrame
        if results:
            df = pd.DataFrame(results)
            
            # Calculate summary statistics
            accuracy = df['correct_prediction'].mean()
            avg_brier = df['brier_score'].mean()
            coverage = len(results) / len(test_matches) if test_matches else 0
            avg_h2h_matches = df['h2h_matches'].mean()
            
            print(f"\nEVALUATION RESULTS")
            print("=" * 50)
            print(f"Total matches analyzed: {len(test_matches)}")
            print(f"Predictions made: {len(results)}")
            print(f"Coverage: {coverage:.1%}")
            print(f"Accuracy: {accuracy:.1%}")
            print(f"Average Brier Score: {avg_brier:.3f}")
            print(f"Average H2H matches: {avg_h2h_matches:.1f}")
            
            # Accuracy by H2H sample size
            print(f"\nAccuracy by H2H Sample Size:")
            for min_matches in [1, 2, 5, 10]:
                subset = df[df['h2h_matches'] >= min_matches]
                if len(subset) > 0:
                    acc = subset['correct_prediction'].mean()
                    print(f"  {min_matches}+ H2H matches: {acc:.1%} ({len(subset)} predictions)")
            
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
            print("No predictions could be made (no H2H history)!")
            return None
    
    def generate_output_filename(self, year, surface='overall', 
                                tournament=None, round=None):
        """
        Generate standardized filename in model_outputs/
        
        Pattern: {model}_{tournament}_{round}_{year}_{surface}_{timestamp}.csv
        Example: h2h_Wimbledon_F_2024_grass_20250107_143052.csv
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
        print("\nInteractive H2H Prediction Mode")
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
                
                # Match date is required for proper temporal validation
                match_date_input = input("Enter match date (YYYYMMDD): ").strip()
                if not match_date_input:
                    print("Match date is required for temporal validation!")
                    continue
                
                try:
                    match_date = int(match_date_input)
                except ValueError:
                    print("Invalid date format. Use YYYYMMDD")
                    continue
                
                self.predict_match(player1, player2, match_date, verbose=True)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    model = H2HModel()
    
    try:
        if len(sys.argv) >= 4 and not sys.argv[1].startswith('--'):
            # Single match prediction: player1 player2 date
            result = model.predict_match(sys.argv[1], sys.argv[2], int(sys.argv[3]))
        
        elif '--batch-evaluate' in sys.argv:
            # Batch evaluation mode with filters
            if '--year' not in sys.argv:
                print("Error: --year is required for batch evaluation")
                print("Usage: python h2h_simple.py --batch-evaluate --year 2024 [options]")
                return
            
            # Parse parameters
            year_idx = sys.argv.index('--year')
            year = int(sys.argv[year_idx + 1])
            
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
                print("  Single prediction:")
                print("    python h2h_simple.py 'Novak Djokovic' 'Rafael Nadal' 20230601")
                print("")
                print("  Batch evaluation:")
                print("    python h2h_simple.py --batch-evaluate --year 2024")
                print("    python h2h_simple.py --batch-evaluate --year 2024 --surface clay")
                print("    python h2h_simple.py --batch-evaluate --year 2024 --tournament Wimbledon")
                print("    python h2h_simple.py --batch-evaluate --year 2024 --round F")
                print("    python h2h_simple.py --batch-evaluate --year 2024 --output results.csv")
    
    finally:
        model.close()

if __name__ == "__main__":
    main()