#!/usr/bin/env python3

# GAT data preparation for tennis prediction (V2)
"""
Memory-efficient data preparation with proper Neo4j node access patterns
Uses INTEGER IDs for all node types

V2 CHANGES:
- replaced 2w/4w recent form with 4w/8w (27 properties each)
- added 5 new 52w stats: losses, matches_played, wins, service_game_efficiency, straight_sets_pct
- form deltas now use 4w and 8w instead of 2w and 4w
- updated differential features for key 4w/8w stats
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import pickle
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import gc

class GATDataPreparation:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="put_your_passowrd_here"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.cache_dir = Path("gat_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # set random seed for reproducibility
        np.random.seed(42)
        
        # feature groups (matching db structure)
        # V2: Added 5 new 52w stats to performance_52w
        self.feature_groups = {
            'performance_52w': [
                'stats_52w_win_pct', 'stats_52w_serve_points_won_pct', 
                'stats_52w_return_points_won_pct', 'stats_52w_ace_pct', 'stats_52w_df_pct',
                'stats_52w_first_serve_pct', 'stats_52w_first_serve_won_pct',
                'stats_52w_second_serve_won_pct', 'stats_52w_bp_converted_pct', 
                'stats_52w_bp_saved_pct', 'stats_52w_bp_created_per_return_game',
                'stats_52w_bp_faced_per_game', 'stats_52w_games_ratio', 
                'stats_52w_total_points_won_pct', 'stats_52w_close_match_pct', 
                'stats_52w_deciding_set_pct', 'stats_52w_tiebreak_pct',
                'stats_52w_upset_rate', 'stats_52w_upset_avg_magnitude',
                'stats_52w_efficiency_ratio', 'stats_52w_defend_rate',
                'stats_52w_return_game_impact', 'stats_52w_return_games_broken_pct',
                'stats_52w_service_games_held_pct', 'stats_52w_first_return_won_pct',
                'stats_52w_second_return_won_pct',
                # V2: NEW 52w stats
                'stats_52w_losses', 'stats_52w_matches_played', 'stats_52w_wins',
                'stats_52w_service_game_efficiency', 'stats_52w_straight_sets_pct'
            ],
            'surface_specific': [  # these will be formatted w/ actual surface
                'stats_52w_Hard_win_pct', 'stats_52w_Clay_win_pct', 'stats_52w_Grass_win_pct',
                'stats_52w_Hard_serve_pct', 'stats_52w_Clay_serve_pct', 'stats_52w_Grass_serve_pct',
                'stats_52w_Hard_return_pct', 'stats_52w_Clay_return_pct', 'stats_52w_Grass_return_pct'
            ],
            # V2: replaced 2w/4w with 4w/8w stats (27 properties each)
            'recent_form_4w': [
                'stats_4w_ace_pct', 'stats_4w_bp_converted_pct', 'stats_4w_bp_created_per_return_game',
                'stats_4w_bp_faced_per_game', 'stats_4w_bp_saved_pct', 'stats_4w_df_pct',
                'stats_4w_efficiency_ratio', 'stats_4w_first_return_won_pct', 'stats_4w_first_serve_pct',
                'stats_4w_first_serve_won_pct', 'stats_4w_games_ratio', 'stats_4w_losses',
                'stats_4w_matches_per_week', 'stats_4w_matches_played', 'stats_4w_return_game_impact',
                'stats_4w_return_games_broken_pct', 'stats_4w_return_points_won_pct',
                'stats_4w_second_return_won_pct', 'stats_4w_second_serve_won_pct',
                'stats_4w_serve_points_won_pct', 'stats_4w_service_game_efficiency',
                'stats_4w_service_games_held_pct', 'stats_4w_straight_sets_pct',
                'stats_4w_tiebreak_pct', 'stats_4w_total_points_won_pct', 'stats_4w_win_pct',
                'stats_4w_wins'
            ],
            'recent_form_8w': [
                'stats_8w_ace_pct', 'stats_8w_bp_converted_pct', 'stats_8w_bp_created_per_return_game',
                'stats_8w_bp_faced_per_game', 'stats_8w_bp_saved_pct', 'stats_8w_df_pct',
                'stats_8w_efficiency_ratio', 'stats_8w_first_return_won_pct', 'stats_8w_first_serve_pct',
                'stats_8w_first_serve_won_pct', 'stats_8w_games_ratio', 'stats_8w_losses',
                'stats_8w_matches_per_week', 'stats_8w_matches_played', 'stats_8w_return_game_impact',
                'stats_8w_return_games_broken_pct', 'stats_8w_return_points_won_pct',
                'stats_8w_second_return_won_pct', 'stats_8w_second_serve_won_pct',
                'stats_8w_serve_points_won_pct', 'stats_8w_service_game_efficiency',
                'stats_8w_service_games_held_pct', 'stats_8w_straight_sets_pct',
                'stats_8w_tiebreak_pct', 'stats_8w_total_points_won_pct', 'stats_8w_win_pct',
                'stats_8w_wins'
            ]
        }
        
        # V2: differential features to calculate for 4w/8w
        self.recent_form_diff_stats = [
            'bp_converted_pct', 'bp_created_per_return_game', 'bp_faced_per_game', 'bp_saved_pct',
            'first_serve_pct', 'first_serve_won_pct', 'return_game_impact', 'return_games_broken_pct',
            'return_points_won_pct', 'second_serve_won_pct', 'serve_points_won_pct',
            'service_game_efficiency', 'straight_sets_pct', 'total_points_won_pct', 'win_pct'
        ]
    
    def close(self):
        self.driver.close()
    
    def get_tournaments_for_split(self, start_year, end_year):
        # get all Masters and Grand Slam tournaments in date range
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Tournament)
                WHERE t.level IN ['G', 'M', '1000', 'Masters']
                AND t.date >= $start_date AND t.date <= $end_date
                RETURN t.id as tournament_id, t.name as name, 
                       t.date as date, t.surface as surface, t.level as level
                ORDER BY t.date
            """, start_date=start_year*10000, end_date=end_year*10000+1231)
            
            tournaments = pd.DataFrame([dict(r) for r in result])
            print(f"Found {len(tournaments)} Masters/Grand Slam tournaments from {start_year}-{end_year}")
            return tournaments
    
    def get_tournament_matches(self, tournament_id, tournament_date):
        # get all matches from a tournament w/ player IDs
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Tournament {id: $tournament_id})
                MATCH (m:Match)-[:PLAYED_IN]->(t)
                MATCH (winner:Player)-[:WON]->(m)<-[:LOST]-(loser:Player)
                WHERE NOT m.score CONTAINS 'W/O'
                RETURN m.id as match_id,
                       winner.id as winner_id,
                       loser.id as loser_id,
                       m.round as round,
                       m.date as match_date,
                       t.surface as surface
            """, tournament_id=tournament_id)
            
            matches = pd.DataFrame([dict(r) for r in result])
            if len(matches) > 0:
                # use tournament date if match dates are missing
                matches['match_date'] = matches['match_date'].fillna(tournament_date)
            return matches
    
    def get_player_features(self, player_id, date, surface):
        # get all features for a player at a specific date
        features = {}
        
        with self.driver.session() as session:
            # 1. get Player basic info (incl. dob)
            player_result = session.run("""
                MATCH (p:Player {id: $player_id})
                RETURN p.hand as hand, p.dob as dob, p.height as height, p.country as country
            """, player_id=player_id) # player_id as INTEGER
            
            player_record = player_result.single()
            if player_record:
                features['is_righthanded'] = 1 if player_record['hand'] == 'R' else 0
                if player_record['dob']:
                    match_year = date // 10000
                    birth_year = player_record['dob'] // 10000
                    features['age'] = match_year - birth_year
                else:
                    features['age'] = 25
                features['height'] = player_record['height'] if player_record['height'] else 183
            else:
                features['is_righthanded'] = 1
                features['age'] = 25
                features['height'] = 183
            
            # 2. get PlayerStats (find most recent before date)
            stats_result = session.run("""
                MATCH (p:Player {id: $player_id})-[:HAS_STATS]->(ps:PlayerStats)
                WHERE ps.date < $date
                RETURN ps
                ORDER BY ps.date DESC
                LIMIT 1
            """, player_id=player_id, date=date)
            
            stats_record = stats_result.single()
            if stats_record and stats_record['ps']:
                stats = dict(stats_record['ps'])
                
                # extract 52w features
                for feature in self.feature_groups['performance_52w']:
                    features[feature] = stats.get(feature, 0.0)
                
                # Ssrface-specific stats
                surface_cap = surface.capitalize()
                features[f'surface_win_pct'] = stats.get(f'stats_52w_{surface_cap}_win_pct', 0.0)
                features[f'surface_serve_pct'] = stats.get(f'stats_52w_{surface_cap}_serve_pct', 0.0)
                features[f'surface_return_pct'] = stats.get(f'stats_52w_{surface_cap}_return_pct', 0.0)
                
                # V2: recent form :4w stats (27 properties)
                for feature in self.feature_groups['recent_form_4w']:
                    features[feature] = stats.get(feature, 0.0)
                
                # V2: recent form : 8w stats (27 properties)
                for feature in self.feature_groups['recent_form_8w']:
                    features[feature] = stats.get(feature, 0.0)
                
                # V2: form deltas now use 4w and 8w
                features['form_delta_4w'] = features.get('stats_4w_win_pct', 0) - features.get('stats_52w_win_pct', 0)
                features['form_delta_8w'] = features.get('stats_8w_win_pct', 0) - features.get('stats_52w_win_pct', 0)
            else:
                # default values if no stats
                for feature in self.feature_groups['performance_52w']: features[feature] = 0.0
                for feature in self.feature_groups['recent_form_4w']: features[feature] = 0.0
                for feature in self.feature_groups['recent_form_8w']: features[feature] = 0.0
                features['form_delta_4w'] = 0.0
                features['form_delta_8w'] = 0.0
                features['surface_win_pct'] = 0.0
                features['surface_serve_pct'] = 0.0
                features['surface_return_pct'] = 0.0
            
            # 3. get rankings
            rank_result = session.run("""
                MATCH (p:Player {id: $player_id})-[:HAS_RANKING]->(r:Ranking)
                WHERE r.date < $date
                RETURN r.rank as rank, r.points as points
                ORDER BY r.date DESC
                LIMIT 1
            """, player_id=player_id, date=date)
            
            rank_record = rank_result.single()
            if rank_record:
                features['rank'] = rank_record['rank']
                features['ranking_points'] = rank_record['points']
            else:
                features['rank'] = 500
                features['ranking_points'] = 0
            
            # 4. get avg comparison
            avg_group = self._get_comparison_group(features.get('rank', 500))
            avg_result = session.run("""
                MATCH (ps:PlayerStats {player_id: $avg_id})
                WHERE ps.date < $date
                RETURN ps
                ORDER BY ps.date DESC
                LIMIT 1
            """, avg_id=f"avg_top_{avg_group}", date=date)
            
            avg_record = avg_result.single()
            if avg_record and avg_record['ps']:
                avg_stats = dict(avg_record['ps'])
                for feature in ['stats_52w_win_pct', 'stats_52w_serve_points_won_pct', 
                               'stats_52w_return_points_won_pct', 'stats_52w_games_ratio',
                               'stats_52w_total_points_won_pct']:
                    features[f'rel_{feature}'] = features.get(feature, 0) - avg_stats.get(feature, 0)
            else:
                for feature in ['stats_52w_win_pct', 'stats_52w_serve_points_won_pct', 
                               'stats_52w_return_points_won_pct', 'stats_52w_games_ratio',
                               'stats_52w_total_points_won_pct']:
                    features[f'rel_{feature}'] = 0.0
            
            # 5. get ELO
            elo_result = session.run("""
                MATCH (p:Player {id: $player_id})-[:HAS_ELO]->(e:ELO)
                WHERE e.date < $date
                RETURN e.overall as elo_overall, 
                       e.hard as elo_hard,
                       e.clay as elo_clay,
                       e.grass as elo_grass
                ORDER BY e.date DESC
                LIMIT 1
            """, player_id=player_id, date=date)
            
            elo_record = elo_result.single()
            if elo_record:
                features['elo_overall'] = elo_record['elo_overall'] or 1500
                # get surface-specific ELO
                if surface.lower() == 'hard':
                    features['elo_surface'] = elo_record['elo_hard'] or 1500
                elif surface.lower() == 'clay':
                    features['elo_surface'] = elo_record['elo_clay'] or 1500
                elif surface.lower() == 'grass':
                    features['elo_surface'] = elo_record['elo_grass'] or 1500
                else:
                    features['elo_surface'] = features['elo_overall']
            else:
                features['elo_overall'] = 1500
                features['elo_surface'] = 1500
            
            # 6. get Glicko2
            glicko_result = session.run("""
                MATCH (p:Player {id: $player_id})-[:HAS_GLICKO2]->(g:Glicko2)
                WHERE g.date < $date
                RETURN g.rating_overall as rating_overall,
                       g.rd_overall as rd_overall,
                       g.rating_hard as rating_hard,
                       g.rd_hard as rd_hard,
                       g.rating_clay as rating_clay,
                       g.rd_clay as rd_clay,
                       g.rating_grass as rating_grass,
                       g.rd_grass as rd_grass
                ORDER BY g.date DESC
                LIMIT 1
            """, player_id=player_id, date=date)
            
            glicko_record = glicko_result.single()
            if glicko_record:
                features['glicko_rating'] = glicko_record['rating_overall'] or 1500
                features['glicko_rd'] = glicko_record['rd_overall'] or 350
                
                # surface-specific Glicko2
                if surface.lower() == 'hard':
                    features['glicko_surface'] = glicko_record['rating_hard'] or 1500
                    features['glicko_surface_rd'] = glicko_record['rd_hard'] or 350
                elif surface.lower() == 'clay':
                    features['glicko_surface'] = glicko_record['rating_clay'] or 1500
                    features['glicko_surface_rd'] = glicko_record['rd_clay'] or 350
                elif surface.lower() == 'grass':
                    features['glicko_surface'] = glicko_record['rating_grass'] or 1500
                    features['glicko_surface_rd'] = glicko_record['rd_grass'] or 350
                else:
                    features['glicko_surface'] = features['glicko_rating']
                    features['glicko_surface_rd'] = features['glicko_rd']
            else:
                features['glicko_rating'] = 1500
                features['glicko_rd'] = 350
                features['glicko_surface'] = 1500
                features['glicko_surface_rd'] = 350
            
            # 7. get StyleFactors
            style_result = session.run("""
                MATCH (p:Player {id: $player_id})-[:HAS_STYLE_FACTORS]->(sf:StyleFactor)
                WHERE sf.date < $date
                RETURN sf.elo_factors as elo_factors, sf.glicko_factors as glicko_factors
                ORDER BY sf.date DESC
                LIMIT 1
            """, player_id=player_id, date=date)
            
            style_record = style_result.single()
            if style_record:
                elo_factors = style_record['elo_factors'] or []
                glicko_factors = style_record['glicko_factors'] or []
                for i in range(5):
                    features[f'elo_factor_{i+1}'] = elo_factors[i] if i < len(elo_factors) else 0.5
                    features[f'glicko_factor_{i+1}'] = glicko_factors[i] if i < len(glicko_factors) else 0.5
            else:
                for i in range(1, 6):
                    features[f'elo_factor_{i}'] = 0.5
                    features[f'glicko_factor_{i}'] = 0.5
        
        return features
    
    def _get_comparison_group(self, rank):
        # determine which avg. group to compare against
        if rank <= 10:
            return 10
        elif rank <= 20:
            return 20
        elif rank <= 50:
            return 50
        elif rank <= 100:
            return 100
        else:
            return 200
    
    def process_tournament(self, tournament_id, tournament_date, surface):
        # process all matches in tournament w/ RANDOMIZED player positions
        matches = self.get_tournament_matches(tournament_id, tournament_date)
        if len(matches) == 0: return None
        tournament_data = []
        
        for _, match in tqdm(matches.iterrows(), total=len(matches), desc=f"Processing {tournament_id}", leave=False):
            
            # RANDOMIZE who is player1 and player2 for training
            if np.random.random() < 0.5:
                # winner is player1
                player1_id = match['winner_id']
                player2_id = match['loser_id']
                label = 1  # Player1 wins
            else:
                # loser is player1
                player1_id = match['loser_id']
                player2_id = match['winner_id']
                label = 0  # Player1 loses
            
            # get features for both players
            p1_features = self.get_player_features(player1_id, match['match_date'], surface)
            p2_features = self.get_player_features(player2_id, match['match_date'], surface)
            
            # create match record with randomized positions
            match_record = {
                'match_id': match['match_id'],
                'tournament_id': tournament_id,
                'date': match['match_date'],
                'surface': surface,
                'round': match['round'],
                'player1_id': player1_id, # randomized!
                'player2_id': player2_id, # randomized!
                'actual_winner_id': match['winner_id'], # store actual for reference
                'actual_loser_id': match['loser_id'],
                'label': label  # 1 if player1 won, 0 if player2 won
            }
            
            # add features with p1_ and p2_ prefixes
            for key, value in p1_features.items(): match_record[f'p1_{key}'] = value
            for key, value in p2_features.items(): match_record[f'p2_{key}'] = value
            
            # add differential features (p1 - p2)
            # orig 52w differentials
            for key in ['rank', 'elo_overall', 'elo_surface', 'glicko_rating', 
                       'stats_52w_win_pct', 'stats_52w_serve_points_won_pct',
                       'stats_52w_return_points_won_pct', 'stats_52w_total_points_won_pct']:
                if f'p1_{key}' in match_record and f'p2_{key}' in match_record:
                    match_record[f'diff_{key}'] = match_record[f'p1_{key}'] - match_record[f'p2_{key}']
            
            # V2: add differential features for key 4w/8w stats
            for window in ['4w', '8w']:
                for stat in self.recent_form_diff_stats:
                    key = f'stats_{window}_{stat}'
                    if f'p1_{key}' in match_record and f'p2_{key}' in match_record:
                        match_record[f'diff_{key}'] = match_record[f'p1_{key}'] - match_record[f'p2_{key}']
            
            tournament_data.append(match_record)
        
        return pd.DataFrame(tournament_data)
    
    def create_dataset_splits(self):
        # create train/val/test splits
        print("="*70)
        print("CREATING GAT DATASET")
        print("="*70)
        print("NOTE: Player positions are RANDOMIZED to avoid position bias")
        print("All node access uses INTEGER IDs")
        print("V2: Using 4w/8w stats instead of 2w/4w, added 5 new 52w stats")
        print("="*70)
        
        # define splits
        splits = {
            'train': (2008, 2015), # 8 years of training data
            'val': (2016, 2018), # 3 years validation
            'test': (2019, 2024) # 6 years test
            }
        
        datasets = {}
        
        for split_name, (start_year, end_year) in splits.items():
            print(f"\n{split_name.upper()} SET: {start_year}-{end_year}")
            print("-"*40)
            
            # get tournaments
            tournaments = self.get_tournaments_for_split(start_year, end_year)
            
            # check cache (make sure for V2)
            cache_file = self.cache_dir / f"{split_name}_data_v2.pkl"
            if cache_file.exists():
                print(f"Loading from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    datasets[split_name] = pickle.load(f)
                continue
            
            # process tournaments in batches
            all_data = []
            batch_size = 5 # process 5 tournaments at a time
            
            for i in range(0, len(tournaments), batch_size):
                batch = tournaments.iloc[i:i+batch_size]
                print(f"\nProcessing batch {i//batch_size + 1}/{(len(tournaments)-1)//batch_size + 1}")
                
                for _, tourn in batch.iterrows():
                    tourn_data = self.process_tournament(
                        tourn['tournament_id'],
                        tourn['date'],
                        tourn['surface'])
                    if tourn_data is not None:
                        all_data.append(tourn_data)
                
                # garbage collection
                gc.collect()
            
            # combine & save
            if all_data:
                final_data = pd.concat(all_data, ignore_index=True)
                datasets[split_name] = final_data
                
                # save to cache
                with open(cache_file, 'wb') as f: pickle.dump(final_data, f)
                
                print(f"\n{split_name} complete: {len(final_data)} matches")
                print(f"Label distribution: {final_data['label'].value_counts().to_dict()}")
                print(f"Cached to: {cache_file}")
            
        return datasets
    
    def prepare_for_pytorch(self, datasets):
        # convert to pytorch ready format
        import torch
        
        processed = {}
        
        for split_name, df in datasets.items():
            # get feature cols
            p1_feature_cols = [col for col in df.columns if col.startswith('p1_')]
            p2_feature_cols = [col for col in df.columns if col.startswith('p2_')]
            diff_feature_cols = [col for col in df.columns if col.startswith('diff_')]
            
            # create feature matrix
            features = []
            labels = []
            
            for _, row in df.iterrows():
                # combine features: p1 + p2 + differential
                p1_feats = [row[col] for col in p1_feature_cols]
                p2_feats = [row[col] for col in p2_feature_cols]
                diff_feats = [row[col] for col in diff_feature_cols]
                
                combined_features = p1_feats + p2_feats + diff_feats
                features.append(combined_features)
                labels.append(row['label'])
            
            # convert to tensors
            features_tensor = torch.tensor(features, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            
            # handle NaN values
            features_tensor[torch.isnan(features_tensor)] = 0.0
            
            processed[split_name] = {
                'features': features_tensor,
                'labels': labels_tensor,
                'metadata': df[['match_id', 'tournament_id', 'date', 'surface', 
                               'player1_id', 'player2_id', 'actual_winner_id', 
                               'actual_loser_id', 'round', 'label']]}
            
            print(f"\n{split_name}: {len(features)} matches, {len(features[0])} features per match")
            print(f"  Label distribution: {labels_tensor.sum().item():.0f} wins (1s), "
                  f"{(1-labels_tensor).sum().item():.0f} losses (0s)")
            print(f"  Win rate for player1: {labels_tensor.mean().item():.2%}")
            
            # print feature breakdown
            print(f"  - Player 1 features: {len(p1_feature_cols)}")
            print(f"  - Player 2 features: {len(p2_feature_cols)}")
            print(f"  - Differential features: {len(diff_feature_cols)}")
        
        return processed
    
    def create_prediction_example(self, player1_id, player2_id, date, surface):
        # create single pred example for a specific match
        # returns features in same format as training data
        # model output = P(player1 wins)

        # get features for both players
        p1_features = self.get_player_features(player1_id, date, surface)
        p2_features = self.get_player_features(player2_id, date, surface)
        
        # combine features in same order as training
        features = []
        
        # add p1 features (sorted for consistency)
        for key in sorted(p1_features.keys()): features.append(p1_features[key])
        # add p2 features (sorted for consistency)
        for key in sorted(p2_features.keys()): features.append(p2_features[key])
        
        # add differential features (original 52w)
        for key in ['rank', 'elo_overall', 'elo_surface', 'glicko_rating',
                   'stats_52w_win_pct', 'stats_52w_serve_points_won_pct',
                   'stats_52w_return_points_won_pct', 'stats_52w_total_points_won_pct']:
            if key in p1_features and key in p2_features:
                features.append(p1_features[key] - p2_features[key])
        
        # V2: add differential features for key 4w/8w stats
        for window in ['4w', '8w']:
            for stat in self.recent_form_diff_stats:
                key = f'stats_{window}_{stat}'
                if key in p1_features and key in p2_features:
                    features.append(p1_features[key] - p2_features[key])
        
        return features  # model will return P(player1 wins)

def main():
    prep = GATDataPreparation()
    
    try:
        # create datasets w/ caching
        datasets = prep.create_dataset_splits()
        
        # convert to PT format
        torch_data = prep.prepare_for_pytorch(datasets)
        
        # save final PT data
        torch_file = prep.cache_dir / "gat_torch_data_v2.pt"
        import torch
        torch.save(torch_data, torch_file)
        
        print(f"\n{'='*70}")
        print("DATA PREPARATION COMPLETE - V2")
        print(f"{'='*70}")
        print(f"Final data saved to: {torch_file}")
        print("\nDataset sizes:")
        for split, data in torch_data.items():
            print(f"  {split:5s}: {data['features'].shape}")
        
        print("\n" + "="*50)
        print("V2 CHANGES FROM ORIGINAL:")
        print("="*50)
        print("- Replaced 2w stats with 8w stats (27 properties)")
        print("- Replaced 4w stats (3 props) with expanded 4w (27 properties)")
        print("- Added 5 new 52w stats: losses, matches_played, wins,")
        print("  service_game_efficiency, straight_sets_pct")
        print("- Form deltas now use 4w and 8w (instead of 2w and 4w)")
        print("- Added 30 differential features for key 4w/8w stats")
        print("\nKEPT FROM ORIGINAL:")
        print("- All player basic features (age, height, handedness)")
        print("- All original 52w stats (26 properties)")
        print("- Surface-specific stats (current surface only)")
        print("- All ranking features")
        print("- All ELO ratings (overall and surface)")
        print("- All Glicko2 ratings (overall and surface with RD)")
        print("- All style factors (5 ELO + 5 Glicko)")
        print("- All relative-to-average stats")
        print("- All original 8 differential features")
        
        print("\n" + "="*50)
        print("IMPORTANT: Training & Prediction")
        print("="*50)
        print("Training: Player positions are RANDOMIZED")
        print("  - ~50% of examples have label=1 (player1 wins)")
        print("  - ~50% have label=0 (player2 wins)")
        print("\nPrediction: Order matters!")
        print("  features = create_prediction_example(player1_id, player2_id, date, surface)")
        print("  probability = model(features)  # Returns P(player1 wins)")
        print("\nAll IDs are INTEGERS, all property names match actual database")
        
    finally:
        prep.close()

if __name__ == "__main__":
    main()