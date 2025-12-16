#!/usr/bin/env python3

# Elo model w/ NMF (multi-tournament version)

from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TennisAbstractELOWithNMF:
    def __init__(self, uri="neo4j://localhost:7687", user="neo4j", password="put_your_password_here"):
        # init connection to neo4j db
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def load_style_factors(self, tournament_name, year):
        # load NMF style factors for specific tournament
        # generate filename based on tournament
        filename_prefix = f"{tournament_name.replace(' ', '_')}_{year}"
        style_file = Path('style_factors') / f"style_factors_elo_{filename_prefix}.csv"
        
        if style_file.exists():
            self.style_factors_elo = pd.read_csv(style_file)
            print(f"  Loaded ELO style factors for {tournament_name} {year}: {len(self.style_factors_elo)} players")
        else:
            print(f"  WARNING: No style factors found for {tournament_name} {year}")
            self.style_factors_elo = pd.DataFrame()
        
        # create lookup dict
        self.style_dict = {}
        if not self.style_factors_elo.empty:
            factor_cols = [f'elo_factor_{i+1}' for i in range(5)]
            for _, row in self.style_factors_elo.iterrows():
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
    
    def get_elo_at_date(self, player_id, match_date):
        # get player's Elo ratings at or before specified date
        with self.driver.session() as session:
            # get most recent ELO before match date
            elo_query = """
            MATCH (p:Player {id: $player_id})-[:HAS_ELO]->(e:ELO)
            WHERE e.date < $match_date
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
            record = result.single()
            
            if record:
                return {
                    'overall': record['overall'] if record['overall'] else 1500,
                    'hard': record['hard'] if record['hard'] else 1500,
                    'clay': record['clay'] if record['clay'] else 1500,
                    'grass': record['grass'] if record['grass'] else 1500,
                    'rating_date': record['rating_date'],
                    'matches_played': record['matches_played'] if record['matches_played'] else 0,
                    'last_match_date': record['last_match_date']
                }
            
            return None
    
    def get_surface_rating(self, elo_data, surface_type):
        # get rating based on surface type
        # TennisAbstract uses a 50/50 blend of overall and surface-specific
        if not elo_data: return None
        if surface_type == 'overall': return elo_data['overall']
        
        # map surface types
        surface_map = {
            'hard': 'hard',
            'clay': 'clay', 
            'grass': 'grass',
            'carpet': 'hard'  # treat carpet as hard!!
        }
        
        surface_key = surface_map.get(surface_type.lower(), 'hard')
        
        # TennisAbstract: 50/50 blend for surface predictions
        overall_rating = elo_data['overall']
        surface_rating = elo_data[surface_key]
        
        # 50/50 blend
        return (overall_rating + surface_rating) / 2
    
    def calculate_elo_probability(self, rating1, rating2):
        # calc win probability for player 1
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    
    def calculate_style_adjustment(self, p1_id, p2_id, max_adjustment=0.05):
        # calc style-based adjustment to probability
        if p1_id not in self.style_dict or p2_id not in self.style_dict: return 0.0
        
        p1_factors = self.style_dict[p1_id]
        p2_factors = self.style_dict[p2_id]
        
        # calculate style diff
        factor_diff = np.abs(p1_factors - p2_factors)
        
        # avg diff across factors
        avg_diff = np.mean(factor_diff)
        
        # convert to adjustment (scale by max_adjustment)
        # assuming avg_diff ranges from 0 to ~2.5
        adjustment = avg_diff * (max_adjustment / 2.5)
        
        return np.clip(adjustment, -max_adjustment, max_adjustment)
    
    def predict_match(self, player1_name, player2_name, match_date, surface='overall', verbose=True, return_ratings=False, force=False, max_adjustment=0.05):
        # predict match outcome with style adjustments
        # SAME AS ORIGINAL but adds style factor adjustment!!
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
        
        # get ELO ratings at match date
        p1_elo = self.get_elo_at_date(p1_id, match_date)
        p2_elo = self.get_elo_at_date(p2_id, match_date)
        
        if not p1_elo:
            if verbose: print(f"No ELO data for {p1_full_name} before {match_date}")
            return None
        
        if not p2_elo:
            if verbose: print(f"No ELO data for {p2_full_name} before {match_date}")
            return None
        
        # get appropriate ratings based on surface
        p1_rating = self.get_surface_rating(p1_elo, surface)
        p2_rating = self.get_surface_rating(p2_elo, surface)
        
        # calc base probabilities
        p1_base_probability = self.calculate_elo_probability(p1_rating, p2_rating)
        
        # add style adjustment!!
        style_adjustment = self.calculate_style_adjustment(p1_id, p2_id, max_adjustment)
        p1_adjusted_probability = np.clip(p1_base_probability + style_adjustment, 0.01, 0.99)
        p2_adjusted_probability = 1 - p1_adjusted_probability
        
        # prep result
        result = {
            'player1_name': p1_full_name,
            'player2_name': p2_full_name,
            'player1_prob': p1_adjusted_probability, # adjusted
            'player2_prob': p2_adjusted_probability, # adjusted
            'match_date': match_date,
            'surface': surface,
            'rating_difference': p1_rating - p2_rating,
            'p1_matches_played': p1_elo['matches_played'],
            'p2_matches_played': p2_elo['matches_played'],
            'rating_date': max(p1_elo['rating_date'], p2_elo['rating_date']),
            'style_adjustment': style_adjustment,  # tracking degree of style adjustment
            'base_prob': p1_base_probability  # track base for comparison
        }
        
        if return_ratings:
            result['p1_rating'] = p1_rating
            result['p2_rating'] = p2_rating
        
        if verbose:
            print(f"\nPrediction for {match_date}:")
            print(f"  {p1_full_name}: {p1_rating:.1f} ELO")
            print(f"  {p2_full_name}: {p2_rating:.1f} ELO")
            print(f"  Base probability: {p1_full_name} {p1_base_probability:.1%}")
            print(f"  Style adjustment: {style_adjustment:+.3f}")
            print(f"  Adjusted probability: {p1_full_name} {p1_adjusted_probability:.1%}")
        
        return result
    
    def batch_evaluate_tournament(self, tournament_name, year, surface_type='overall', max_adjustment=0.05, tournament_surface=None):
        # eval a specific tournament
        #   uses actual surface now listed in CSV
        print(f"\n  Evaluating {tournament_name} {year}")
        
        if surface_type == 'surface' and tournament_surface:
            # use the actual tournament surface for surface-specific evaluation
            eval_surface = tournament_surface.lower()
            print(f"    Surface: {eval_surface} (tournament surface), Max adjustment: {max_adjustment*100:.0f}%")
        elif surface_type == 'overall':
            # use overall ratings only
            eval_surface = 'overall'
            print(f"    Surface: overall, Max adjustment: {max_adjustment*100:.0f}%")
        else:
            # fallback for any other value in case
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
            # get pred w/ style adjustments
            prediction = self.predict_match(
                match['winner_name'],
                match['loser_name'], 
                match['match_date'],
                surface=eval_surface, # determined surface
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
                    'actual_winner': 1,  # winner is always player1 in our query
                    'predicted_prob_p1': prediction['player1_prob'],
                    'predicted_prob_p2': prediction['player2_prob'],
                    'predicted_winner': 1 if prediction['player1_prob'] > 0.5 else 2,
                    'correct_prediction': 1 if prediction['player1_prob'] > 0.5 else 0,
                    'brier_score': (prediction['player1_prob'] - 1) ** 2,
                    'surface': match['actual_surface'],
                    'tournament': match['tournament'],
                    'tournament_level': match['tournament_level'],
                    'round': match['round'],
                    
                    # ELO specific cols
                    'p1_rating': prediction.get('p1_rating', None),
                    'p2_rating': prediction.get('p2_rating', None),
                    'rating_diff': prediction['rating_difference'],
                    
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
    #   'overall' uses overall rating, 'surface' uses tournament-specific surface
    configs = [
        {'surface': 'overall', 'adjustment': 0.00, 'name': 'elo_overall_baseline'},
        {'surface': 'surface', 'adjustment': 0.00, 'name': 'elo_surface_baseline'},
        {'surface': 'overall', 'adjustment': 0.05, 'name': 'elo_overall_nmf_5pct'},
        {'surface': 'overall', 'adjustment': 0.10, 'name': 'elo_overall_nmf_10pct'},
        {'surface': 'overall', 'adjustment': 0.25, 'name': 'elo_overall_nmf_25pct'},
        {'surface': 'overall', 'adjustment': 0.50, 'name': 'elo_overall_nmf_50pct'},
        {'surface': 'overall', 'adjustment': 0.75, 'name': 'elo_overall_nmf_75pct'},
        {'surface': 'overall', 'adjustment': 1.00, 'name': 'elo_overall_nmf_100pct'},
        {'surface': 'surface', 'adjustment': 0.05, 'name': 'elo_surface_nmf_5pct'},
        {'surface': 'surface', 'adjustment': 0.10, 'name': 'elo_surface_nmf_10pct'},
        {'surface': 'surface', 'adjustment': 0.25, 'name': 'elo_surface_nmf_25pct'},
        {'surface': 'surface', 'adjustment': 0.50, 'name': 'elo_surface_nmf_50pct'},
        {'surface': 'surface', 'adjustment': 0.75, 'name': 'elo_surface_nmf_75pct'},
        {'surface': 'surface', 'adjustment': 1.00, 'name': 'elo_surface_nmf_100pct'},
    ]
    
    # create output directory
    output_dir = Path('elo_nmf_results')
    output_dir.mkdir(exist_ok=True)
    
    model = TennisAbstractELOWithNMF()
    
    try:
        # process all configs
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
                    int(tournament['year']))
                
                # run eval (passing tournament from CSV + surface)
                results = model.batch_evaluate_tournament(
                    tournament_name=tournament['tournament_name'],
                    year=int(tournament['year']),
                    surface_type=config['surface'],
                    max_adjustment=config['adjustment'],
                    tournament_surface=tournament['surface']
                )
                
                if results is not None:
                    # + config info
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
    print("ELO NMF MULTI-TOURNAMENT EVALUATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()