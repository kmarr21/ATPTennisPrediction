# import libraries
import os
import pandas as pd
from neo4j import GraphDatabase
import glob
import numpy as np
from tqdm import tqdm  #progress bars

# Neo4j connection parameters
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "RolandGarros2195!"

#path
DATA_DIR = "/Users/kierstenmarr/desktop/atp_match_data"

class TennisDataImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")
    
    # import players
    def import_players(self, file_path):
        print("Importing players...")
        
        #read player data
        players_df = pd.read_csv(file_path)
        
        # process data
        # convert DOB to integer if not already
        if 'dob' in players_df.columns:
            players_df['dob'] = players_df['dob'].fillna(0).astype(int)
        if 'height' in players_df.columns:
            players_df['height'] = players_df['height'].fillna(0).astype(float)
        
        # batch importing to protect
        batch_size = 1000
        total_batches = len(players_df) // batch_size + (1 if len(players_df) % batch_size > 0 else 0)
        
        for i in tqdm(range(total_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(players_df))
            batch = players_df.iloc[start_idx:end_idx]
            
            # Cypher param list!
            player_data = [
                {
                    "id": int(row['player_id']),
                    "first_name": str(row['name_first']),
                    "last_name": str(row['name_last']),
                    "hand": str(row['hand']) if not pd.isna(row['hand']) else None,
                    "dob": int(row['dob']) if row['dob'] > 0 else None,
                    "country": str(row['ioc']) if not pd.isna(row['ioc']) else None,
                    "height": float(row['height']) if not pd.isna(row['height']) and row['height'] > 0 else None,
                    "wikidata_id": str(row['wikidata_id']) if 'wikidata_id' in row and not pd.isna(row['wikidata_id']) else None
                }
                for _, row in batch.iterrows()]
            
            # run the query
            with self.driver.session() as session:
                session.run("""
                UNWIND $players AS player
                MERGE (p:Player {id: player.id})
                SET p.first_name = player.first_name,
                    p.last_name = player.last_name,
                    p.hand = player.hand,
                    p.dob = player.dob,
                    p.country = player.country,
                    p.height = player.height,
                    p.wikidata_id = player.wikidata_id
                """, players=player_data)
        
        print(f"Imported {len(players_df)} players.")
    
    # import player rankings
    def import_rankings(self, file_paths):
        print("Importing rankings...")
        
        #process each ranking file
        for file_path in file_paths:
            rankings_df = pd.read_csv(file_path)
            print(f"Processing {file_path}...")
            
            # batching
            batch_size = 5000
            total_batches = len(rankings_df) // batch_size + (1 if len(rankings_df) % batch_size > 0 else 0)
            
            for i in tqdm(range(total_batches)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(rankings_df))
                batch = rankings_df.iloc[start_idx:end_idx]
                
                # Cypher param list
                ranking_data = [
                    {
                        "player_id": int(row['player']),
                        "date": int(row['ranking_date']),
                        "rank": int(row['rank']),
                        "points": float(row['points']) if not pd.isna(row['points']) else 0.0
                    }
                    for _, row in batch.iterrows()
                ]
                
                # Execute Cypher query
                with self.driver.session() as session:
                    session.run("""
                    UNWIND $rankings AS ranking
                    MATCH (p:Player {id: ranking.player_id})
                    MERGE (r:Ranking {player_id: ranking.player_id, date: ranking.date})
                    SET r.rank = ranking.rank,
                        r.points = ranking.points
                    MERGE (p)-[:HAS_RANKING]->(r)
                    """, rankings=ranking_data)
        
        print("Rankings imported.")

    def import_matches(self, file_paths):
        print("Importing matches...")
        
        for file_path in file_paths:
            print(f"Processing {file_path}...")
            matches_df = pd.read_csv(file_path)
            
            # unique tournament nodes FIRST
            tournaments = matches_df[['tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level', 'tourney_date']].drop_duplicates()
            
            # process the tournaments
            tournament_data = [
                {
                    "id": str(row['tourney_id']),
                    "name": str(row['tourney_name']),
                    "surface": str(row['surface']) if not pd.isna(row['surface']) else "",
                    "draw_size": int(row['draw_size']) if not pd.isna(row['draw_size']) else 0,
                    "level": str(row['tourney_level']) if not pd.isna(row['tourney_level']) else "",
                    "date": int(row['tourney_date'])
                }
                for _, row in tournaments.iterrows()
            ]
            
            with self.driver.session() as session:
                session.run("""
                UNWIND $tournaments AS tournament
                MERGE (t:Tournament {id: tournament.id})
                SET t.name = tournament.name,
                    t.surface = tournament.surface,
                    t.draw_size = tournament.draw_size,
                    t.level = tournament.level,
                    t.date = tournament.date
                """, tournaments=tournament_data)
            
            # batching of matches
            batch_size = 1000
            total_batches = len(matches_df) // batch_size + (1 if len(matches_df) % batch_size > 0 else 0)
            
            for i in tqdm(range(total_batches)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(matches_df))
                batch = matches_df.iloc[start_idx:end_idx]
                
                # creating match data
                match_data = []
                for _, row in batch.iterrows():
                    # create unique match ID
                    match_id = f"{row['tourney_id']}-{row['match_num']}"
                    
                    # match properties (handles null values)
                    match_props = {
                        "id": match_id,
                        "match_num": int(row['match_num']),
                        "score": str(row['score']) if not pd.isna(row['score']) else "",
                        "best_of": int(row['best_of']) if not pd.isna(row['best_of']) else 0,
                        "round": str(row['round']) if not pd.isna(row['round']) else "",
                        "minutes": float(row['minutes']) if not pd.isna(row['minutes']) else 0.0,
                        "tourney_id": str(row['tourney_id'])
                    }
                    
                    # adds stats if they exist!
                    stats_fields = [
                        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 
                        'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 
                        'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
                    ]
                    
                    for field in stats_fields:
                        if field in row and not pd.isna(row[field]):
                            match_props[field] = float(row[field])
                        else:
                            match_props[field] = 0.0
                    
                    # create WINNER relationship
                    winner_props = {
                        "rank": float(row['winner_rank']) if not pd.isna(row['winner_rank']) else 0.0,
                        "rank_points": float(row['winner_rank_points']) if not pd.isna(row['winner_rank_points']) else 0.0,
                        "age": float(row['winner_age']) if not pd.isna(row['winner_age']) else 0.0
                    }
                    
                    #seed and entry separately (can be omitted if NULL)
                    if 'winner_seed' in row and not pd.isna(row['winner_seed']):
                        winner_props["seed"] = float(row['winner_seed'])
                        
                    if 'winner_entry' in row and not pd.isna(row['winner_entry']):
                        winner_props["entry"] = str(row['winner_entry'])
                    
                    # LOSER reltaionship
                    loser_props = {
                        "rank": float(row['loser_rank']) if not pd.isna(row['loser_rank']) else 0.0,
                        "rank_points": float(row['loser_rank_points']) if not pd.isna(row['loser_rank_points']) else 0.0,
                        "age": float(row['loser_age']) if not pd.isna(row['loser_age']) else 0.0}
                    
                    #seed and entry separately (can be omitted if NULL)
                    if 'loser_seed' in row and not pd.isna(row['loser_seed']):
                        loser_props["seed"] = float(row['loser_seed'])
                        
                    if 'loser_entry' in row and not pd.isna(row['loser_entry']):
                        loser_props["entry"] = str(row['loser_entry'])
                    
                    match_data.append({
                        "match": match_props,
                        "winner": {
                            "player_id": int(row['winner_id']),
                            "match_id": match_id,
                            "props": winner_props
                        },
                        "loser": {
                            "player_id": int(row['loser_id']),
                            "match_id": match_id,
                            "props": loser_props
                        }
                    })
                
                # add matches and relationships cypher query
                with self.driver.session() as session:
                    session.run("""
                    UNWIND $matches AS match_data
                    
                    // create match
                    MERGE (m:Match {id: match_data.match.id})
                    SET m += match_data.match
                    
                    WITH m, match_data
                    
                    // Connect match to tournament
                    MATCH (t:Tournament {id: match_data.match.tourney_id})
                    MERGE (m)-[:PLAYED_IN]->(t)
                    
                    WITH m, match_data
                    
                    // connect winner to match
                    MATCH (w:Player {id: match_data.winner.player_id})
                    CALL {
                        WITH w, m, match_data
                        MERGE (w)-[r:WON]->(m)
                        SET r.rank = match_data.winner.props.rank,
                            r.rank_points = match_data.winner.props.rank_points,
                            r.age = match_data.winner.props.age
                        
                        // set properties only if they exist
                        WITH r, match_data.winner.props as props
                        WHERE props.seed IS NOT NULL
                        SET r.seed = props.seed
                    }
                    
                    CALL {
                        WITH w, m, match_data
                        MATCH (w)-[r:WON]->(m)
                        WITH r, match_data.winner.props as props
                        WHERE props.entry IS NOT NULL
                        SET r.entry = props.entry
                    }
                    
                    WITH m, match_data
                    
                    // connect loser to match
                    MATCH (l:Player {id: match_data.loser.player_id})
                    CALL {
                        WITH l, m, match_data
                        MERGE (l)-[r:LOST]->(m)
                        SET r.rank = match_data.loser.props.rank,
                            r.rank_points = match_data.loser.props.rank_points,
                            r.age = match_data.loser.props.age
                        
                        // set properties only if they exist
                        WITH r, match_data.loser.props as props
                        WHERE props.seed IS NOT NULL
                        SET r.seed = props.seed
                    }
                    
                    CALL {
                        WITH l, m, match_data
                        MATCH (l)-[r:LOST]->(m)
                        WITH r, match_data.loser.props as props
                        WHERE props.entry IS NOT NULL
                        SET r.entry = props.entry
                    }
                    """, matches=match_data)
            
        print("Matches imported.")

# main exec func
def import_tennis_data(data_dir, neo4j_uri, neo4j_user, neo4j_password):
    # init importer
    importer = TennisDataImporter(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # ideally maybe want to clear the database first, since this is the inital import (can uncomment below)
        #importer.clear_database()
        print("Skipping database clearing to avoid memory issues. Using MERGE operations to prevent duplicates.")
        
        # import players
        players_file = os.path.join(data_dir, "atp_players.csv")
        if os.path.exists(players_file):
            importer.import_players(players_file)
        else:
            print(f"Players file not found: {players_file}")
        
        # import rankings
        rankings_files = [
            os.path.join(data_dir, "atp_rankings_00s.csv"),
            os.path.join(data_dir, "atp_rankings_10s.csv"),
            os.path.join(data_dir, "atp_rankings_20s.csv"),
            os.path.join(data_dir, "atp_rankings_current.csv")
        ]
        existing_rankings_files = [f for f in rankings_files if os.path.exists(f)]
        if existing_rankings_files:
            importer.import_rankings(existing_rankings_files)
        else:
            print("No ranking files found.")
        
        # import match data
        atp_matches_pattern = os.path.join(data_dir, "atp_matches_*.csv")
        qual_chall_matches_pattern = os.path.join(data_dir, "atp_matches_qual_chall_*.csv")
        
        atp_match_files = glob.glob(atp_matches_pattern)
        qual_chall_match_files = glob.glob(qual_chall_matches_pattern)
        
        # filter out futures matches (don't want these)
        atp_match_files = [f for f in atp_match_files if "futures" not in f.lower()]
        
        # filter for only 2000 onwards
        atp_match_files = [f for f in atp_match_files if int(os.path.basename(f).split("_")[-1].split('.')[0]) >= 2000]
        qual_chall_match_files = [f for f in qual_chall_match_files if int(os.path.basename(f).split("_")[-1].split('.')[0]) >= 2000]
        
        # import ATP matches
        if atp_match_files:
            print(f"Importing {len(atp_match_files)} ATP match files...")
            importer.import_matches(atp_match_files)
        else:
            print("No ATP match files found.")
        
        # import Qualifies and Challenger matches
        if qual_chall_match_files:
            print(f"Importing {len(qual_chall_match_files)} Qualification/Challenger match files...")
            importer.import_matches(qual_chall_match_files)
        else:
            print("No Qualification/Challenger match files found.")
        
        print("Data import completed successfully!")
        
    finally:
        # close Neo4j connection
        importer.close()

#run import process
if __name__ == "__main__":
    import_tennis_data(
        DATA_DIR,
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD
    )