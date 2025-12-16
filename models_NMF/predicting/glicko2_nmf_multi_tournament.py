#!/usr/bin/env python3

# Glicko2+NMF model (multi-tournament version)

from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import math
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Glicko2WithNMF:
    def __init__(self, uri="neo4j://localhost:7687", user="neo4j", password="put_your_password_here"):
        # init connection to Neo4j database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def load_style_factors(self, tournament_name, year):
        #load the NMF style factors for a specific tournament
        # generate filename based on tournament
        filename_prefix = f"{tournament_name.replace(' ', '_')}_{year}"
        style_file = Path('style_factors') / f"style_factors_glicko_{filename_prefix}.csv"
        
        if style_file.exists():
            self.style_factors_glicko = pd.read_csv(style_file)
            print(f"  Loaded Glicko2 style factors for {tournament_name} {year}: {len(self.style_factors_glicko)} players")
        else:
            print(f"  WARNING: No style factors found for {tournament_name} {year}")
            self.style_factors_glicko = pd.DataFrame()
        
        # create lookup dict
        self.style_dict = {}
        if not self.style_factors_glicko.empty:
            factor_cols = [f'glicko_factor_{i+1}' for i in range(5)]
            for _, row in self.style_factors_glicko.iterrows():
                self.style_dict[row['player_id']] = row[factor_cols].values
    
    def close(self):
        # close db connection
        self.driver.close()
    
    def find_player(self, player_name):
        # find player ID by name
        with self.driver.session() as session:
            # try exact match first
            exact_query = """
            MATCH (p:Player)
            WHERE toLower(p.first_name + ' ' + p.last_name) = toLower($name)
            RETURN p.id as id, p.first_name + ' ' + p.last_name as full_name
            LIMIT 1
            """
            result = session.run(exact_query, name=player_name)
            record = result.single()
            
            if record:
                return record['id'], record['full_name']
            
            # try partial match
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
            
            if len(matches) == 1:
                return matches[0]['id'], matches[0]['full_name']
            elif len(matches) > 1:
                # for batch mode, just take the first match
                return matches[0]['id'], matches[0]['full_name']
            
            return None, None
    
    def get_glicko2_at_date(self, player_id, match_date):
        # get player's Glicko2 ratings at/before specified date
        with self.driver.session() as session:
            # get most recent Glicko2 before match date
            glicko_query = """
            MATCH (p:Player {id: $player_id})-[:HAS_GLICKO2]->(g:Glicko2)
            WHERE g.date < $match_date
            WITH g
            ORDER BY g.date DESC
            LIMIT 1
            RETURN 
                g.rating_overall as rating_overall,
                g.rating_hard as rating_hard,
                g.rating_clay as rating_clay,
                g.rating_grass as rating_grass,
                g.rd_overall as rd_overall,
                g.rd_hard as rd_hard,
                g.rd_clay as rd_clay,
                g.rd_grass as rd_grass,
                g.volatility_overall as volatility_overall,
                g.volatility_hard as volatility_hard,
                g.volatility_clay as volatility_clay,
                g.volatility_grass as volatility_grass,
                g.date as rating_date,
                g.total_matches as total_matches
            """
            
            result = session.run(glicko_query, player_id=player_id, match_date=match_date)
            record = result.single()
            
            if record:
                return {
                    'rating_overall': record['rating_overall'] if record['rating_overall'] else 1500,
                    'rating_hard': record['rating_hard'] if record['rating_hard'] else 1500,
                    'rating_clay': record['rating_clay'] if record['rating_clay'] else 1500,
                    'rating_grass': record['rating_grass'] if record['rating_grass'] else 1500,
                    'rd_overall': record['rd_overall'] if record['rd_overall'] else 350,
                    'rd_hard': record['rd_hard'] if record['rd_hard'] else 350,
                    'rd_clay': record['rd_clay'] if record['rd_clay'] else 350,
                    'rd_grass': record['rd_grass'] if record['rd_grass'] else 350,
                    'volatility_overall': record['volatility_overall'] if record['volatility_overall'] else 0.06,
                    'volatility_hard': record['volatility_hard'] if record['volatility_hard'] else 0.06,
                    'volatility_clay': record['volatility_clay'] if record['volatility_clay'] else 0.06,
                    'volatility_grass': record['volatility_grass'] if record['volatility_grass'] else 0.06,
                    'rating_date': record['rating_date'],
                    'total_matches': record['total_matches'] if record['total_matches'] else 0
                }
            
            return None
    
    def calculate_glicko2_probability(self, rating_a, rd_a, rating_b, rd_b):
        # calculate win probability using Glicko2 formula
        # Based on Glickman's formula:
        #   P(A wins) = 1 / (1 + 10^(-g(sqrt(RD_a^2 + RD_b^2)) * (R_a - R_b) / 400))
        #   where g(RD) = 1 / sqrt(1 + 3 * (RD/pi)^2 / 400)

        # combined RD for the match 
        combined_rd = math.sqrt(rd_a * rd_a + rd_b * rd_b)
        
        # g function
        g = 1 / math.sqrt(1 + 3 * (combined_rd / math.pi) ** 2 / 400)
        
        # win probability
        prob = 1 / (1 + 10 ** (-g * (rating_a - rating_b) / 400))
        
        return prob
    
    def get_surface_ratings(self, glicko2_data, surface_type):
        # get ratings based on surface type
        # returns: (rating, rd, volatility)
        if not glicko2_data:
            return None, None, None
            
        if surface_type == 'overall':
            return (glicko2_data['rating_overall'], 
                   glicko2_data['rd_overall'],
                   glicko2_data['volatility_overall'])
        
        # map surface types
        surface_map = {
            'hard': 'hard',
            'clay': 'clay', 
            'grass': 'grass',
            'carpet': 'hard'  # treat carpet as hard!!
        }
        
        surface_key = surface_map.get(surface_type.lower(), 'hard')
        
        # surface predictions -> blend overall and surface-specific
        # weight by inverse of RD (more certain = more weight)
        overall_rating = glicko2_data['rating_overall']
        overall_rd = glicko2_data['rd_overall']
        surface_rating = glicko2_data[f'rating_{surface_key}']
        surface_rd = glicko2_data[f'rd_{surface_key}']
        surface_volatility = glicko2_data[f'volatility_{surface_key}']
        
        # weight by inverse of RD
        overall_weight = 1 / overall_rd if overall_rd > 0 else 1
        surface_weight = 1 / surface_rd if surface_rd > 0 else 1
        total_weight = overall_weight + surface_weight
        
        # weighted avg
        blended_rating = (overall_rating * overall_weight + surface_rating * surface_weight) / total_weight
        blended_rd = math.sqrt((overall_rd**2 * overall_weight + surface_rd**2 * surface_weight) / total_weight)
        
        # for volatility, simple avg
        blended_volatility = (glicko2_data['volatility_overall'] + surface_volatility) / 2
        
        return blended_rating, blended_rd, blended_volatility
    
    def calculate_style_adjustment(self, p1_id, p2_id, max_adjustment=0.05):
        #calc style-based adjustment to probability
        if p1_id not in self.style_dict or p2_id not in self.style_dict:
            return 0.0
        
        p1_factors = self.style_dict[p1_id]
        p2_factors = self.style_dict[p2_id]
        
        # calc style difference
        factor_diff = np.abs(p1_factors - p2_factors)
        
        # avg diff across factors
        avg_diff = np.mean(factor_diff)
        
        # convert to adjustment (scale by max_adjustment)
        # assuming avg_diff ranges from 0 to ~2.5
        adjustment = avg_diff * (max_adjustment / 2.5)
        
        return np.clip(adjustment, -max_adjustment, max_adjustment)
    
    def predict_match(self, player1_name, player2_name, match_date, surface='overall', verbose=True, return_ratings=False, force=False, max_adjustment=0.05):
        #predict match outcome using Glicko-2 ratings w/ style adjustments
        # find players
        p1_id, p1_full_name = self.find_player(player1_name)
        if not p1_id:
            if verbose: print(f"Player '{player1_name}' not found.")
            return None
        
        p2_id, p2_full_name = self.find_player(player2_name)
        if not p2_id:
            if verbose: print(f"Player '{player2_name}' not found.")
            return None
        
        if verbose:
            print(f"Found: {p1_full_name} vs {p2_full_name}")
        
        # get Glicko-2 ratings at match date
        p1_glicko2 = self.get_glicko2_at_date(p1_id, match_date)
        p2_glicko2 = self.get_glicko2_at_date(p2_id, match_date)
        
        if not p1_glicko2:
            if verbose: print(f"No Glicko-2 data for {p1_full_name} before {match_date}")
            return None
        
        if not p2_glicko2:
            if verbose: print(f"No Glicko-2 data for {p2_full_name} before {match_date}")
            return None
        
        # get appropriate ratings based on surface
        p1_rating, p1_rd, p1_vol = self.get_surface_ratings(p1_glicko2, surface)
        p2_rating, p2_rd, p2_vol = self.get_surface_ratings(p2_glicko2, surface)
        
        # calc base probabilities
        p1_base_probability = self.calculate_glicko2_probability(p1_rating, p1_rd, p2_rating, p2_rd)
        
        # add style adjustment
        style_adjustment = self.calculate_style_adjustment(p1_id, p2_id, max_adjustment)
        p1_adjusted_probability = np.clip(p1_base_probability + style_adjustment, 0.01, 0.99)
        p2_adjusted_probability = 1 - p1_adjusted_probability
        
        # calc confidence intervals (95% = rating ± 2*RD)
        p1_conf_lower = p1_rating - 2 * p1_rd
        p1_conf_upper = p1_rating + 2 * p1_rd
        p2_conf_lower = p2_rating - 2 * p2_rd
        p2_conf_upper = p2_rating + 2 * p2_rd
        
        # prep result
        result = {
            'player1_name': p1_full_name,
            'player2_name': p2_full_name,
            'player1_prob': p1_adjusted_probability,
            'player2_prob': p2_adjusted_probability,
            'match_date': match_date,
            'surface': surface,
            'rating_difference': p1_rating - p2_rating,
            'p1_total_matches': p1_glicko2['total_matches'],
            'p2_total_matches': p2_glicko2['total_matches'],
            'rating_date': max(p1_glicko2['rating_date'], p2_glicko2['rating_date']),
            'p1_rd': p1_rd,
            'p2_rd': p2_rd,
            'p1_volatility': p1_vol,
            'p2_volatility': p2_vol,
            'combined_uncertainty': math.sqrt(p1_rd**2 + p2_rd**2),
            'style_adjustment': style_adjustment,
            'base_prob': p1_base_probability
        }
        
        if return_ratings:
            result['p1_rating'] = p1_rating
            result['p2_rating'] = p2_rating
            result['p1_conf_lower'] = p1_conf_lower
            result['p1_conf_upper'] = p1_conf_upper
            result['p2_conf_lower'] = p2_conf_lower
            result['p2_conf_upper'] = p2_conf_upper
        
        if verbose:
            print(f"\nRatings:")
            print(f"  {p1_full_name}: {p1_rating:.1f} ± {p1_rd:.1f}")
            print(f"  {p2_full_name}: {p2_rating:.1f} ± {p2_rd:.1f}")
            print(f"\nBase probability: {p1_full_name} {p1_base_probability:.1%}")
            print(f"Style adjustment: {style_adjustment:+.3f}")
            print(f"Adjusted probability: {p1_full_name} {p1_adjusted_probability:.1%}")
        
        return result
    
    def batch_evaluate_tournament(self, tournament_name, year, surface_type='overall', max_adjustment=0.05, tournament_surface=None):
        # eval specific tournament
        print(f"\n  Evaluating {tournament_name} {year}")
        
        # if surface_type is 'surface', use tournament's actual surface
        if surface_type == 'surface' and tournament_surface:
            # use the actual tournament surface for surface-specific eval
            eval_surface = tournament_surface.lower()
            print(f"    Surface: {eval_surface} (tournament surface), Max adjustment: {max_adjustment*100:.0f}%")
        elif surface_type == 'overall':
            # use overall ratings only
            eval_surface = 'overall'
            print(f"    Surface: overall, Max adjustment: {max_adjustment*100:.0f}%")
        else:
            # fallback for any other value
            eval_surface = surface_type
            print(f"    Surface: {eval_surface}, Max adjustment: {max_adjustment*100:.0f}%")
        
        # build query
        filters = [
            f"t.date >= {year}0101 AND t.date <= {year}1231",
            "NOT m.score CONTAINS 'W/O'"]
        
        filters.append(f"toLower(t.name) CONTAINS toLower('{tournament_name}')")
        
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
        
        print(f"    Found {len(test_matches)} matches to evaluate...")
        
        # eval matches
        results = []
        predictions_made = 0
        
        for i, match in enumerate(test_matches):
            # get prediction w/ style adjustments
            prediction = self.predict_match(
                match['winner_name'],
                match['loser_name'], 
                match['match_date'],
                surface=eval_surface, # use determined surface
                verbose=False,
                return_ratings=True,
                force=True,
                max_adjustment=max_adjustment
            )
            
            if prediction:
                predictions_made += 1
                
                # store results in std format
                result_row = {
                    # required cols (all models)
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
                    
                    # Glicko-2 specific cols
                    'p1_rating': prediction.get('p1_rating', None),
                    'p2_rating': prediction.get('p2_rating', None),
                    'p1_rd': prediction['p1_rd'],
                    'p2_rd': prediction['p2_rd'],
                    'rating_diff': prediction['rating_difference'],
                    'combined_uncertainty': prediction['combined_uncertainty'],
                    
                    # style adjustment cols
                    'style_adjustment': prediction.get('style_adjustment', 0),
                    'base_prob': prediction.get('base_prob', prediction['player1_prob'])
                }
                
                results.append(result_row)
        
        print(f"    Made {predictions_made} predictions")
        
        # calc accuracy & Brier
        if results:
            df_results = pd.DataFrame(results)
            accuracy = df_results['correct_prediction'].mean() * 100
            avg_brier = df_results['brier_score'].mean()
            print(f"    Accuracy: {accuracy:.1f}%, Brier: {avg_brier:.4f}")
            
            return df_results
        else:
            print("    No predictions made!")
            return None

def main():
    # run evals w/ different configurations
    
    # load tournament list
    tournaments_df = pd.read_csv('tournaments_to_analyze.csv')
    
    # filter to tournaments w/ valid data
    valid_tournaments = tournaments_df[tournaments_df['tournament_date'].notna()]
    print(f"Processing {len(valid_tournaments)} tournaments with valid data")
    
    # config sets to test
    # 'overall' uses overall rating, 'surface' uses tournament-specific surface
    configs = [
        {'surface': 'overall', 'adjustment': 0.00, 'name': 'glicko2_overall_baseline'},
        {'surface': 'surface', 'adjustment': 0.00, 'name': 'glicko2_surface_baseline'},
        {'surface': 'overall', 'adjustment': 0.05, 'name': 'glicko2_overall_nmf_5pct'},
        {'surface': 'overall', 'adjustment': 0.10, 'name': 'glicko2_overall_nmf_10pct'},
        {'surface': 'overall', 'adjustment': 0.25, 'name': 'glicko2_overall_nmf_25pct'},
        {'surface': 'overall', 'adjustment': 0.50, 'name': 'glicko2_overall_nmf_50pct'},
        {'surface': 'overall', 'adjustment': 0.75, 'name': 'glicko2_overall_nmf_75pct'},
        {'surface': 'overall', 'adjustment': 1.00, 'name': 'glicko2_overall_nmf_100pct'},
        {'surface': 'surface', 'adjustment': 0.05, 'name': 'glicko2_surface_nmf_5pct'},
        {'surface': 'surface', 'adjustment': 0.10, 'name': 'glicko2_surface_nmf_10pct'},
        {'surface': 'surface', 'adjustment': 0.25, 'name': 'glicko2_surface_nmf_25pct'},
        {'surface': 'surface', 'adjustment': 0.50, 'name': 'glicko2_surface_nmf_50pct'},
        {'surface': 'surface', 'adjustment': 0.75, 'name': 'glicko2_surface_nmf_75pct'},
        {'surface': 'surface', 'adjustment': 1.00, 'name': 'glicko2_surface_nmf_100pct'},
    ]
    
    # create output directory
    output_dir = Path('glicko2_nmf_results')
    output_dir.mkdir(exist_ok=True)
    
    model = Glicko2WithNMF()
    
    try:
        # process each config
        for config in configs:
            print(f"\n{'='*60}")
            print(f"Running: {config['name']}")
            print(f"{'='*60}")
            
            all_results = []
            tournament_summaries = []
            
            for _, tournament in valid_tournaments.iterrows():
                # load style factors for this tournament
                model.load_style_factors(
                    tournament['tournament_name'],
                    int(tournament['year'])
                )
                
                # run eval -> PASS THE TOURNAMENT SURFACE FROM CSV!!
                results = model.batch_evaluate_tournament(
                    tournament_name=tournament['tournament_name'],
                    year=int(tournament['year']),
                    surface_type=config['surface'],
                    max_adjustment=config['adjustment'],
                    tournament_surface=tournament['surface'] # pass actual surface from CSV
                )
                
                if results is not None:
                    # add config info
                    results['tournament_name'] = tournament['tournament_name']
                    results['year'] = tournament['year']
                    results['model'] = config['name']
                    results['surface_type'] = config['surface']
                    results['max_adjustment'] = config['adjustment']
                    
                    all_results.append(results)
                    
                    # calc tournament-specific metrics
                    accuracy = results['correct_prediction'].mean()
                    avg_brier = results['brier_score'].mean()
                    
                    tournament_summaries.append({
                        'tournament': tournament['tournament_name'],
                        'year': tournament['year'],
                        'model': config['name'],
                        'accuracy': accuracy * 100,
                        'brier_score': avg_brier,
                        'n_matches': len(results)
                    })
            
            # save combined results (..if any)
            if all_results:
                combined_df = pd.concat(all_results, ignore_index=True)
                
                # save detailed results
                detail_file = output_dir / f"{config['name']}_all_matches.csv"
                combined_df.to_csv(detail_file, index=False)
                
                # save tournament summary
                summary_df = pd.DataFrame(tournament_summaries)
                summary_file = output_dir / f"{config['name']}_tournament_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                
                # overall stats
                overall_accuracy = combined_df['correct_prediction'].mean() * 100
                overall_brier = combined_df['brier_score'].mean()
                
                print(f"\n  OVERALL for {config['name']}:")
                print(f"    {len(tournament_summaries)} tournaments")
                print(f"    Accuracy: {overall_accuracy:.1f}%")
                print(f"    Brier Score: {overall_brier:.4f}")
    
    finally:
        model.close()
    
    print("\n" + "="*60)
    print("GLICKO2 NMF MULTI-TOURNAMENT EVALUATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()