#!/usr/bin/env python3

# generate StyleFactors for multiple tournaments
#   compatible w/ older scikit-learn versions

from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from pathlib import Path
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings('ignore')

class StyleFactorGenerator:
    def __init__(self, uri="neo4j://localhost:7687", user="neo4j", password="put_your_password_here"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.N_PLAYERS = 100  # top 100 players by tour-level wins
        self.N_COMPONENTS = 5  # num of style factors
        
    def close(self):
        self.driver.close()
    
    def get_top_players(self, start_date, end_date):
        # get top N players by TOUR LEVEL wins in date range
        # NO challengers!! (disorted outcomes..)
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Player)-[:WON]->(m:Match)-[:PLAYED_IN]->(t:Tournament)
                WHERE t.date >= $start_date AND t.date < $end_date
                AND t.level IN ['G', 'M', 'A', 'F', 'D', '1000', '500', '250']  // Tour-level events
                WITH p, count(m) as wins
                ORDER BY wins DESC
                LIMIT $n_players
                RETURN p.id as player_id, 
                       p.first_name + ' ' + p.last_name as name,
                       wins
                ORDER BY wins DESC
            """, start_date=start_date, end_date=end_date, n_players=self.N_PLAYERS)
            
            players = []
            for record in result:
                players.append({
                    'player_id': record['player_id'],
                    'name': record['name'],
                    'wins': record['wins']})
            
            return pd.DataFrame(players)
    
    def get_ratings_at_date(self, player_ids, cutoff_date):
        # get Elo and Glicko2 ratings for players at cutoff date
        # gets rating AT OR BEFORE cutoff date (Monday before tournament)
        with self.driver.session() as session:
            # get ELO ratings
            elo_query = session.run("""
                UNWIND $player_ids AS pid
                MATCH (p:Player {id: pid})-[:HAS_ELO]->(e:ELO)
                WHERE e.date <= $cutoff_date
                WITH p, e ORDER BY e.date DESC
                WITH p, COLLECT(e)[0] as latest_elo
                RETURN p.id as player_id,
                       latest_elo.overall as elo_overall,
                       latest_elo.hard as elo_hard,
                       latest_elo.clay as elo_clay,
                       latest_elo.grass as elo_grass
            """, player_ids=player_ids, cutoff_date=cutoff_date)
            
            elo_ratings = {}
            for record in elo_query:
                if record['elo_overall'] is not None:
                    elo_ratings[record['player_id']] = {
                        'elo_overall': record['elo_overall'] or 1500,
                        'elo_hard': record['elo_hard'] or 1500,
                        'elo_clay': record['elo_clay'] or 1500,
                        'elo_grass': record['elo_grass'] or 1500}
            
            # get Glicko2 ratings
            glicko_query = session.run("""
                UNWIND $player_ids AS pid
                MATCH (p:Player {id: pid})-[:HAS_GLICKO2]->(g:Glicko2)
                WHERE g.date <= $cutoff_date
                WITH p, g ORDER BY g.date DESC
                WITH p, COLLECT(g)[0] as latest_glicko
                RETURN p.id as player_id,
                       latest_glicko.rating_overall as glicko_overall,
                       latest_glicko.rating_hard as glicko_hard,
                       latest_glicko.rating_clay as glicko_clay,
                       latest_glicko.rating_grass as glicko_grass,
                       latest_glicko.rd_overall as rd_overall,
                       latest_glicko.rd_hard as rd_hard,
                       latest_glicko.rd_clay as rd_clay,
                       latest_glicko.rd_grass as rd_grass
            """, player_ids=player_ids, cutoff_date=cutoff_date)
            
            glicko_ratings = {}
            for record in glicko_query:
                if record['glicko_overall'] is not None:
                    glicko_ratings[record['player_id']] = {
                        'glicko_overall': record['glicko_overall'] or 1500,
                        'glicko_hard': record['glicko_hard'] or 1500,
                        'glicko_clay': record['glicko_clay'] or 1500,
                        'glicko_grass': record['glicko_grass'] or 1500,
                        'rd_overall': record['rd_overall'] or 350,
                        'rd_hard': record['rd_hard'] or 350,
                        'rd_clay': record['rd_clay'] or 350,
                        'rd_grass': record['rd_grass'] or 350
                    }
            
            return elo_ratings, glicko_ratings
    
    def get_h2h_matches(self, player_ids, start_date, end_date):
        # get all H2H matches between selected players in date range (52w period = year before tournament)
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p1:Player)-[r1:WON|LOST]->(m:Match)<-[r2:WON|LOST]-(p2:Player)
                MATCH (m)-[:PLAYED_IN]->(t:Tournament)
                WHERE p1.id IN $player_ids AND p2.id IN $player_ids
                AND p1.id < p2.id  // Avoid duplicates
                AND t.date >= $start_date AND t.date < $end_date
                AND NOT m.score CONTAINS 'W/O'  // Exclude walkovers
                RETURN 
                    p1.id as player1_id,
                    p2.id as player2_id,
                    CASE WHEN type(r1) = 'WON' THEN 1 ELSE 0 END as player1_won,
                    t.surface as surface,
                    t.date as match_date,
                    m.id as match_id
                ORDER BY t.date
            """, player_ids=player_ids, start_date=start_date, end_date=end_date)
            
            matches = []
            for record in result:
                matches.append({
                    'player1_id': record['player1_id'],
                    'player2_id': record['player2_id'],
                    'player1_won': record['player1_won'],
                    'surface': record['surface'],
                    'match_date': record['match_date'],
                    'match_id': record['match_id']
                })
            
            return pd.DataFrame(matches)
    
    def calculate_deviations(self, players_df, h2h_df, elo_ratings, glicko_ratings):
        # calc deviation matrices from expected results
        # create player index mapping
        player_ids = players_df['player_id'].tolist()
        player_id_to_idx = {pid: i for i, pid in enumerate(player_ids)}
        n_players = len(player_ids)
        
        # init matrices
        elo_deviation_matrix = np.zeros((n_players, n_players))
        glicko_deviation_matrix = np.zeros((n_players, n_players))
        match_count_matrix = np.zeros((n_players, n_players))
        
        # ELO probability function
        def elo_expected(rating1, rating2):
            return 1.0 / (1 + 10 ** ((rating2 - rating1) / 400.0))
        
        # glicko2 probability function
        def glicko2_expected(rating1, rating2, rd1, rd2):
            # Combined RD - proper Glicko2 formula
            combined_rd = math.sqrt(rd1 * rd1 + rd2 * rd2)
            # g function with combined RD
            g = 1 / math.sqrt(1 + 3 * (combined_rd / math.pi) ** 2 / 400)
            # win prob
            return 1 / (1 + 10 ** (-g * (rating1 - rating2) / 400))
        
        # group H2H results by player pair
        h2h_summary = h2h_df.groupby(['player1_id', 'player2_id']).agg({'player1_won': ['sum', 'count']}).reset_index()
        h2h_summary.columns = ['player1_id', 'player2_id', 'p1_wins', 'total_matches']
        h2h_summary['p1_win_rate'] = h2h_summary['p1_wins'] / h2h_summary['total_matches']
        
        print(f"  Found {len(h2h_summary)} unique player pairings")
        print(f"  Total H2H matches: {h2h_summary['total_matches'].sum()}")
        
        # calc deviations
        for _, row in h2h_summary.iterrows():
            p1_id = row['player1_id']
            p2_id = row['player2_id']
            
            if p1_id not in player_id_to_idx or p2_id not in player_id_to_idx: continue
            
            idx1 = player_id_to_idx[p1_id]
            idx2 = player_id_to_idx[p2_id]
            
            actual_p1_win_rate = row['p1_win_rate']
            n_matches = row['total_matches']
            
            # store match counts
            match_count_matrix[idx1, idx2] = n_matches
            match_count_matrix[idx2, idx1] = n_matches
            
            # calc ELO deviation
            if p1_id in elo_ratings and p2_id in elo_ratings:
                elo1 = elo_ratings[p1_id]['elo_overall']
                elo2 = elo_ratings[p2_id]['elo_overall']
                expected_elo = elo_expected(elo1, elo2)
                
                deviation_elo = actual_p1_win_rate - expected_elo
                elo_deviation_matrix[idx1, idx2] = deviation_elo
                elo_deviation_matrix[idx2, idx1] = -deviation_elo
            
            # calc Glicko2 deviation
            if p1_id in glicko_ratings and p2_id in glicko_ratings:
                g1 = glicko_ratings[p1_id]['glicko_overall']
                g2 = glicko_ratings[p2_id]['glicko_overall']
                rd1 = glicko_ratings[p1_id]['rd_overall']
                rd2 = glicko_ratings[p2_id]['rd_overall']
                expected_glicko = glicko2_expected(g1, g2, rd1, rd2)
                
                deviation_glicko = actual_p1_win_rate - expected_glicko
                glicko_deviation_matrix[idx1, idx2] = deviation_glicko
                glicko_deviation_matrix[idx2, idx1] = -deviation_glicko
        
        # diagnostics
        print(f"  ELO deviation range: [{elo_deviation_matrix.min():.3f}, {elo_deviation_matrix.max():.3f}]")
        print(f"  Glicko deviation range: [{glicko_deviation_matrix.min():.3f}, {glicko_deviation_matrix.max():.3f}]")
        print(f"  Max matches between players: {match_count_matrix.max():.0f}")
        
        return elo_deviation_matrix, glicko_deviation_matrix, match_count_matrix
    
    def apply_nmf(self, deviation_matrix, match_count_matrix):
        # apply NMF to extract style factors
        # shift matrix to be non-negative
        min_val = deviation_matrix.min()
        matrix_shifted = deviation_matrix - min_val
        
        print(f"  Shifted matrix range: [{matrix_shifted.min():.3f}, {matrix_shifted.max():.3f}]")
        
        # weight by square root of match counts (gives less extreme weighting)
        weight_matrix = np.sqrt(match_count_matrix + 1)
        weighted_matrix = matrix_shifted * weight_matrix
        
        # apply NMF
        nmf = NMF(n_components=self.N_COMPONENTS, 
                  init='nndsvd', #better for sparse data like h2hs
                  max_iter=500, 
                  random_state=42)
        
        style_factors = nmf.fit_transform(weighted_matrix)
        
        # normalize to [0, 1]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        style_factors_normalized = scaler.fit_transform(style_factors)
        
        # calc reconstruction error
        reconstruction = nmf.inverse_transform(style_factors)
        error = np.mean((weighted_matrix - reconstruction) ** 2)
        print(f"  NMF reconstruction error: {error:.4f}")
        
        return style_factors_normalized
    
    def process_tournament(self, tournament_row):
        # process a single tournament & generate style factors
        print(f"\n{'='*60}")
        print(f"Processing {tournament_row['tournament_name']} {tournament_row['year']}")
        print(f"{'='*60}")
        print(f"Tournament date: {tournament_row['tournament_date']}")
        print(f"Rating cutoff date: {tournament_row['rating_cutoff_date']}")
        print(f"H2H period: {tournament_row['h2h_start_date']} to {tournament_row['h2h_end_date']}")
        
        # 1: get top players by TOUR-LEVEL WINS
        print("\nStep 1: Getting top players by tour-level wins...")
        players_df = self.get_top_players(
            tournament_row['h2h_start_date'],
            tournament_row['h2h_end_date'])
        print(f"  Found {len(players_df)} players")
        
        if len(players_df) < 50:
            print("  WARNING: Not enough players with wins in this period")
            return None
        
        # show top 10 players
        print("\n  Top 10 players by wins:")
        for i, row in players_df.head(10).iterrows():
            print(f"    {i+1}. {row['name']}: {row['wins']} wins")
        
        player_ids = players_df['player_id'].tolist()
        
        # 2: get ratings at cutoff date
        print("\nStep 2: Getting ELO and Glicko2 ratings...")
        elo_ratings, glicko_ratings = self.get_ratings_at_date(
            player_ids,
            tournament_row['rating_cutoff_date'])
        print(f"  Found ELO ratings for {len(elo_ratings)} players")
        print(f"  Found Glicko2 ratings for {len(glicko_ratings)} players")
        
        # 3: get H2H matches
        print("\nStep 3: Getting H2H matches...")
        h2h_df = self.get_h2h_matches(
            player_ids,
            tournament_row['h2h_start_date'],
            tournament_row['h2h_end_date'])
        print(f"  Found {len(h2h_df)} total H2H matches")
        
        if len(h2h_df) < 100:
            print("  WARNING: Not enough H2H matches for reliable factors")
            return None
        
        # 4: calc deviation matrices
        print("\nStep 4: Calculating deviation matrices...")
        elo_dev, glicko_dev, match_counts = self.calculate_deviations(
            players_df, h2h_df, elo_ratings, glicko_ratings
        )
        
        # 5: apply NMF
        print("\nStep 5: Applying NMF...")
        print("  ELO factors:")
        elo_factors = self.apply_nmf(elo_dev, match_counts)
        print("  Glicko2 factors:")
        glicko_factors = self.apply_nmf(glicko_dev, match_counts)
        
        # create output dataframes
        elo_df = pd.DataFrame(elo_factors, columns=[f'elo_factor_{i+1}' for i in range(self.N_COMPONENTS)])
        elo_df['player_id'] = player_ids
        elo_df['name'] = players_df['name'].values
        
        glicko_df = pd.DataFrame(glicko_factors, columns=[f'glicko_factor_{i+1}' for i in range(self.N_COMPONENTS)])
        glicko_df['player_id'] = player_ids
        glicko_df['name'] = players_df['name'].values
        
        # reorder columns
        elo_cols = ['player_id', 'name'] + [f'elo_factor_{i+1}' for i in range(self.N_COMPONENTS)]
        elo_df = elo_df[elo_cols]
        
        glicko_cols = ['player_id', 'name'] + [f'glicko_factor_{i+1}' for i in range(self.N_COMPONENTS)]
        glicko_df = glicko_df[glicko_cols]
        
        print("\nStyle factors generated successfully!")
        
        # show sample of factors
        print("\nSample of ELO factors (top 5 players):")
        print(elo_df.head())
        
        return {
            'elo_factors': elo_df,
            'glicko_factors': glicko_df,
            'metadata': {
                'n_players': len(players_df),
                'n_h2h_matches': len(h2h_df),
                'n_elo_ratings': len(elo_ratings),
                'n_glicko_ratings': len(glicko_ratings)
            }
        }

def main():
    # load tournament list
    tournaments_df = pd.read_csv('tournaments_to_analyze.csv')
    
    # filter to valid tournaments
    valid_tournaments = tournaments_df[tournaments_df['tournament_date'].notna()]
    print(f"Loaded {len(valid_tournaments)} tournaments to analyze")
    
    # create output directory
    output_dir = Path('style_factors')
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # initialize generator
    generator = StyleFactorGenerator()
    
    try:
        # process each tournament
        for _, tournament in valid_tournaments.iterrows():
            # generate filename prefix
            filename_prefix = f"{tournament['tournament_name'].replace(' ', '_')}_{tournament['year']}"
            
            # check if already exists
            elo_file = output_dir / f"style_factors_elo_{filename_prefix}.csv"
            glicko_file = output_dir / f"style_factors_glicko_{filename_prefix}.csv"
            
            if elo_file.exists() and glicko_file.exists():
                print(f"\nSkipping {tournament['tournament_name']} {tournament['year']} - already exists")
                continue
            
            # process tournament
            factors = generator.process_tournament(tournament)
            
            if factors:
                # save to CSV
                factors['elo_factors'].to_csv(elo_file, index=False)
                factors['glicko_factors'].to_csv(glicko_file, index=False)
                print(f"\nSaved style factors:")
                print(f"  {elo_file.name}")
                print(f"  {glicko_file.name}")
                
                # save metadata
                metadata_file = output_dir / f"metadata_{filename_prefix}.txt"
                with open(metadata_file, 'w') as f:
                    f.write(f"Tournament: {tournament['tournament_name']} {tournament['year']}\n")
                    f.write(f"Tournament Date: {tournament['tournament_date']}\n")
                    f.write(f"Rating Cutoff: {tournament['rating_cutoff_date']}\n")
                    f.write(f"H2H Period: {tournament['h2h_start_date']} to {tournament['h2h_end_date']}\n")
                    f.write(f"\nStatistics:\n")
                    for key, value in factors['metadata'].items():
                        f.write(f"  {key}: {value}\n")
            else:
                print(f"Failed to generate factors for {tournament['tournament_name']} {tournament['year']}")
        
        print("\n" + "="*60)
        print("STYLE FACTOR GENERATION COMPLETE!")
        print("="*60)
        
    finally:
        generator.close()

if __name__ == "__main__":
    main()