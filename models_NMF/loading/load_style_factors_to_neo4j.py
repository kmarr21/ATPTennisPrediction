#!/usr/bin/env python3

# script to load StyleFactors into neo4j database
#   Reads style factors from style_factors_GAT directory and creates StyleFactor nodes in Neo4j with proper relationships to players
#   HEAP-SAFE VERSION: processes in small batches with garbage collection

from neo4j import GraphDatabase
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm
import gc  # For garbage collection
import time

class StyleFactorLoader:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="put_your_password_here"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.style_factors_dir = Path("style_factors_GAT")
        
    def close(self):
        self.driver.close()
    
    def parse_metadata(self, metadata_file):
        # get rating cutoff date from metdata file
        metadata = {}
        with open(metadata_file, 'r') as f:
            for line in f:
                if 'Rating Cutoff:' in line:
                    # get date like "20040315" from "Rating Cutoff: 20040315"
                    date_match = re.search(r'(\d{8})', line)
                    if date_match:
                        metadata['rating_cutoff'] = int(date_match.group(1))
                elif 'Tournament:' in line:
                    # get tournament name & year
                    parts = line.replace('Tournament:', '').strip().rsplit(' ', 1)
                    if len(parts) == 2:
                        metadata['tournament_name'] = parts[0]
                        metadata['year'] = int(parts[1])
        return metadata
    
    def create_style_factor_nodes(self, year_dir):
        # process all tournaments in a year directory

        # fet all metadata files in this year
        metadata_files = list(year_dir.glob("metadata_*.txt"))
        
        for idx, metadata_file in enumerate(metadata_files, 1):
            # parse metadata
            metadata = self.parse_metadata(metadata_file)
            if not metadata.get('rating_cutoff'):
                print(f"Warning: No rating cutoff found in {metadata_file}")
                continue
            
            # construct CSV filenames
            base_name = metadata_file.stem.replace('metadata_', '')
            elo_file = year_dir / f"style_factors_elo_{base_name}.csv"
            glicko_file = year_dir / f"style_factors_glicko_{base_name}.csv"
            
            if not elo_file.exists() or not glicko_file.exists():
                print(f"Warning: Missing CSV files for {base_name}")
                continue
            
            # load style factors
            elo_df = pd.read_csv(elo_file)
            glicko_df = pd.read_csv(glicko_file)
            
            print(f"\nTournament {idx}/{len(metadata_files)}: {metadata['tournament_name']} {metadata['year']}")
            print(f"  Rating cutoff: {metadata['rating_cutoff']}")
            print(f"  Players: {len(elo_df)}")
            
            # create nodes in neo4j
            self.batch_create_nodes(elo_df, glicko_df, metadata)
            
            # MEMORY MANAGEMENT: clean up after each tournament
            del elo_df, glicko_df
            gc.collect()
            
            # small pause to let db catch up
            time.sleep(0.5)
    
    def batch_create_nodes(self, elo_df, glicko_df, metadata):
        # create StyleFactor nodes for all players in a tournament
        
        # merge dfs on player_id and name (both have same cols)
        merged_df = pd.merge(elo_df, glicko_df, on=['player_id', 'name'], suffixes=('_elo', '_glicko'))
        
        # DEBUG: check cols
        # print(f"  Columns after merge: {merged_df.columns.tolist()}")
        
        # BATCH SIZE FOR HEAP SAFETY
        BATCH_SIZE = 20  # process only 20 players at a time
        
        # prep all data first
        all_data = []
        for _, row in merged_df.iterrows():
            # since both CSVs have same column names, after merge I have:
            # elo_factor_1_elo, elo_factor_1_glicko OR just elo_factor_1, glicko_factor_1
            # check what we actually have....
            all_data.append({
                'player_id': row['player_id'],
                'player_name': row['name'],  # just 'name', not 'name_elo'
                'elo_f1': float(row['elo_factor_1']),
                'elo_f2': float(row['elo_factor_2']),
                'elo_f3': float(row['elo_factor_3']),
                'elo_f4': float(row['elo_factor_4']),
                'elo_f5': float(row['elo_factor_5']),
                'glicko_f1': float(row['glicko_factor_1']),
                'glicko_f2': float(row['glicko_factor_2']),
                'glicko_f3': float(row['glicko_factor_3']),
                'glicko_f4': float(row['glicko_factor_4']),
                'glicko_f5': float(row['glicko_factor_5'])
            })
        
        # process in small batches
        total_created = 0
        for i in range(0, len(all_data), BATCH_SIZE):
            batch_data = all_data[i:i + BATCH_SIZE]
            
            with self.driver.session() as session:
                # create StyleFactor nodes for this batch
                create_query = """
                UNWIND $batch as row
                CREATE (sf:StyleFactor {
                    player_id: row.player_id,
                    tournament_name: $tournament_name,
                    year: $year,
                    date: $date,
                    elo_factors: [row.elo_f1, row.elo_f2, row.elo_f3, row.elo_f4, row.elo_f5],
                    glicko_factors: [row.glicko_f1, row.glicko_f2, row.glicko_f3, row.glicko_f4, row.glicko_f5],
                    player_name: row.player_name
                })
                """
                
                session.run(create_query, 
                           batch=batch_data,
                           tournament_name=metadata['tournament_name'],
                           year=metadata['year'],
                           date=metadata['rating_cutoff'])
                
                # create relationships for this batch
                relationship_query = """
                UNWIND $batch as row
                MATCH (p:Player {id: row.player_id})
                MATCH (sf:StyleFactor {
                    player_id: row.player_id, 
                    tournament_name: $tournament_name,
                    year: $year
                })
                CREATE (p)-[:HAS_STYLE_FACTORS {date: $date}]->(sf)
                """
                
                batch_player_ids = [{'player_id': d['player_id']} for d in batch_data]
                session.run(relationship_query,
                           batch=batch_player_ids,
                           tournament_name=metadata['tournament_name'],
                           year=metadata['year'],
                           date=metadata['rating_cutoff'])
                
                total_created += len(batch_data)
                print(f"    Batch {i//BATCH_SIZE + 1}: Created {len(batch_data)} nodes (Total: {total_created}/{len(all_data)})")
        
        print(f"  Completed: {total_created} StyleFactor nodes created")
    
    def create_indexes(self):
        # create indices for efficient querying
        with self.driver.session() as session:
            # create index on StyleFactor date for temporal queries
            session.run("CREATE INDEX style_factor_date IF NOT EXISTS FOR (sf:StyleFactor) ON (sf.date)")
            # create index on player_id for lookups
            session.run("CREATE INDEX style_factor_player IF NOT EXISTS FOR (sf:StyleFactor) ON (sf.player_id)")
            # composite index for player+date queries
            session.run("""
                CREATE INDEX style_factor_composite IF NOT EXISTS 
                FOR (sf:StyleFactor) ON (sf.player_id, sf.date)""")
            
            print("Created indexes for StyleFactor nodes")
    
    def verify_load(self):
        # verify data was loaded correctly
        with self.driver.session() as session:
            # count total StyleFactor nodes
            result = session.run("MATCH (sf:StyleFactor) RETURN count(sf) as total")
            total = result.single()['total']
            
            # count by year
            year_counts = session.run("""
                MATCH (sf:StyleFactor)
                RETURN sf.year as year, count(sf) as count
                ORDER BY year""")
            
            print(f"\nVerification Results:")
            print(f"Total StyleFactor nodes: {total}")
            print("\nBy year:")
            for record in year_counts:
                print(f"  {record['year']}: {record['count']} nodes")
            
            # sample query
            sample = session.run("""
                MATCH (p:Player)-[:HAS_STYLE_FACTORS]->(sf:StyleFactor)
                RETURN p.first_name + ' ' + p.last_name as player, 
                       sf.tournament_name as tournament,
                       sf.year as year,
                       sf.elo_factors[0] as first_elo_factor
                LIMIT 5
            """)
            
            print("\nSample relationships:")
            for record in sample:
                print(f"  {record['player']} - {record['tournament']} {record['year']}: {record['first_elo_factor']:.3f}")

def main():
    loader = StyleFactorLoader()
    
    try:
        print("="*70)
        print("LOADING STYLE FACTORS INTO NEO4J")
        print("="*70)
        
        # create indices first
        loader.create_indexes()
        
        # process each year
        years = sorted([d for d in loader.style_factors_dir.iterdir() if d.is_dir()])
        
        for year_dir in years:
            year = year_dir.name
            print(f"\n{'='*50}")
            print(f"Processing Year: {year}")
            print('='*50)
            
            loader.create_style_factor_nodes(year_dir)
        
        # verify load
        loader.verify_load()
        
        print("\n" + "="*70)
        print("STYLE FACTOR LOADING COMPLETE!")
        print("="*70)
        
    finally:
        loader.close()

if __name__ == "__main__":
    main()