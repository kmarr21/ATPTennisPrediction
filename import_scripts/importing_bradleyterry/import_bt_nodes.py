#!/usr/bin/env python3

# script to import Bradley-Terry strength nodes
"""
- pre-computes and stores weekly Bradley-Terry player strengths
- uses the paper's approach with MLE estimation
"""

import os
import sys
from neo4j import GraphDatabase
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Neo4j connection parameters
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "put_your_password_here"

# params
IMPORT_HALFLIFE = 240 # days: chose middle of paper's range
IMPORT_SURFACE_WEIGHT = 0.25 #weight for non-matching surfaces

class BradleyTerryImporter:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        print("=" * 70)
        print("BRADLEY-TERRY NODE IMPORTER")
        print("=" * 70)
        print("Connected to Neo4j database.")
        print(f"Using halflife: {IMPORT_HALFLIFE} days")
        print(f"Using surface weight: {IMPORT_SURFACE_WEIGHT}")
    
    def close(self):
        self.driver.close()
    
    def clear_existing_bt_nodes(self):
        # cleary any existing BT nodes
        print("\nClearing existing Bradley-Terry nodes...")
        
        with self.driver.session() as session:
            # check count first
            result = session.run("MATCH (bt:BradleyTerry) RETURN count(bt) as count")
            count = result.single()['count']
            
            if count > 0:
                print(f"Found {count:,} existing BT nodes to delete...")
                
                # delete in batches
                batch_size = 10000
                deleted = 0
                
                while True:
                    result = session.run("""
                        MATCH (bt:BradleyTerry)
                        WITH bt LIMIT $batch_size
                        DETACH DELETE bt
                        RETURN count(bt) as deleted
                    """, batch_size=batch_size)
                    
                    batch_deleted = result.single()['deleted']
                    deleted += batch_deleted
                    
                    if batch_deleted == 0:
                        break
                    
                    if deleted % 50000 == 0:
                        print(f"  Deleted {deleted:,} nodes...")
                
                print(f"Cleared {deleted:,} BT nodes.")
            else:
                print("No existing BT nodes found.")
    
    def get_all_matches_before_date(self, before_date, days_back=365*3):
        # get matches in time window before specified date
        start_date = before_date - days_back * 10000 // 365  # rough conversion...
        
        with self.driver.session() as session:
            query = """
            MATCH (winner:Player)-[:WON]->(m:Match)<-[:LOST]-(loser:Player)
            MATCH (m)-[:PLAYED_IN]->(t:Tournament)
            WHERE t.date < $before_date 
            AND t.date >= $start_date
            AND m.score IS NOT NULL
            AND NOT m.score CONTAINS 'W/O'
            RETURN 
                winner.id as winner_id,
                loser.id as loser_id,
                m.score as score,
                t.date as match_date,
                t.surface as surface
            ORDER BY t.date DESC
            """
            
            result = session.run(query, before_date=before_date, start_date=start_date)
            return list(result)
    
    def parse_score(self, score_str):
        # parse score to extract total games won by each player
        if not score_str or 'W/O' in score_str or 'DEF' in score_str: return 0, 0
        if 'RET' in score_str: score_str = score_str.replace(' RET', '').replace('RET', '')
        winner_games = 0
        loser_games = 0
        sets = score_str.strip().split()
        
        for set_score in sets:
            # remove tiebreak notation
            if '(' in set_score: set_score = set_score.split('(')[0]
            if '[' in set_score: set_score = set_score.split('[')[0]
            
            if '-' in set_score:
                try:
                    games = set_score.split('-')
                    w_games = int(games[0])
                    l_games = int(games[1])
                    winner_games += w_games
                    loser_games += l_games
                except (ValueError, IndexError):
                    continue
        
        return winner_games, loser_games
    
    def normalize_surface(self, surface):
        # normalize surface names
        if not surface: return 'hard'
        
        surface_lower = surface.lower()
        if 'carpet' in surface_lower: return 'hard'
        elif 'clay' in surface_lower: return 'clay'
        elif 'grass' in surface_lower: return 'grass'
        else: return 'hard'
    
    def calculate_surface_weight(self, match_surface, target_surface):
        # calc weight based on surface similarity
        match_surf = self.normalize_surface(match_surface)
        target_surf = self.normalize_surface(target_surface)
        
        if match_surf == target_surf: return 1.0
        else: return IMPORT_SURFACE_WEIGHT
    
    def estimate_strengths_for_surface(self, matches, snapshot_date, surface):
        # estim. player strengths for specific surface using MLE
        # build player list and match data
        players = set()
        match_data = []
        
        for match in matches:
            players.add(match['winner_id'])
            players.add(match['loser_id'])
            
            # parse score
            games_w, games_l = self.parse_score(match['score'])
            if games_w == 0 and games_l == 0:
                continue
            
            #calc time decay
            days_ago = (snapshot_date - match['match_date']) / 10000 * 365
            time_weight = np.exp(-days_ago / IMPORT_HALFLIFE)
            
            # calc surface weight
            surf_weight = self.calculate_surface_weight(match['surface'], surface)
            
            # combined weight
            weight = time_weight * surf_weight
            
            match_data.append({
                'winner': match['winner_id'],
                'loser': match['loser_id'],
                'games_w': games_w,
                'games_l': games_l,
                'weight': weight})
        
        if not match_data or len(players) < 2: return {}
        
        # create player index
        player_list = sorted(list(players))
        player_idx = {p: i for i, p in enumerate(player_list)}
        n_players = len(player_list)
        
        # init log strengths
        log_strengths = np.zeros(n_players)
        
        def neg_log_likelihood(log_alpha):
            #neg log-likelihood for Bradley-Terry model
            nll = 0
            
            for match in match_data:
                i = player_idx[match['winner']]
                j = player_idx[match['loser']]
                gi = match['games_w']
                gj = match['games_l']
                w = match['weight']
                
                # use logsumexp (for stability)
                log_sum = np.logaddexp(log_alpha[i], log_alpha[j])
                
                # ...to negative log-likelihood
                nll -= w * (gi * (log_alpha[i] - log_sum) + gj * (log_alpha[j] - log_sum))
            
            # + small regularization to prevent extreme values
            nll += 0.001 * np.sum(log_alpha ** 2)
            
            return nll
        
        # optimize
        result = minimize(neg_log_likelihood, log_strengths, method='L-BFGS-B', options={'maxiter': 100})
        
        if result.success or result.fun < 1e10: #accept if reasonably converged
            # convert to regular scale
            strengths = np.exp(result.x)
            # normalize to geometric mean of 1
            strengths = strengths / np.exp(np.mean(np.log(strengths)))
            return {player_list[i]: float(strengths[i]) for i in range(n_players)}
        else: return {}
    
    def compute_weekly_snapshot(self, snapshot_date):
        #compute Bradley-Terry strengths for all surfaces at a given date
        #get historical matches
        matches = self.get_all_matches_before_date(snapshot_date)
        
        if not matches:return []
        
        # compute strengths for each surface
        surfaces = ['overall', 'hard', 'clay', 'grass']
        all_strengths = {}
        
        for surface in surfaces:
            strengths = self.estimate_strengths_for_surface(matches, snapshot_date, surface)
            all_strengths[surface] = strengths
        
        # create node data for each player
        bt_nodes = []
        players = set()
        for surface_strengths in all_strengths.values():
            players.update(surface_strengths.keys())
        
        for player_id in players:
            # only create node if player has at least one strength!!
            if any(player_id in all_strengths[s] for s in surfaces):
                node_data = {
                    'player_id': player_id,
                    'date': snapshot_date,
                    'strength_overall': all_strengths['overall'].get(player_id, 1.0),
                    'strength_hard': all_strengths['hard'].get(player_id, 1.0),
                    'strength_clay': all_strengths['clay'].get(player_id, 1.0),
                    'strength_grass': all_strengths['grass'].get(player_id, 1.0),
                    'matches_used': len(matches)}
                bt_nodes.append(node_data)
        
        return bt_nodes
    
    def save_snapshot(self, bt_nodes):
        #ave BT snapshot to database
        if not bt_nodes: return 0
        
        with self.driver.session() as session:
            query = """
            UNWIND $nodes AS node_data
            CREATE (bt:BradleyTerry {
                player_id: node_data.player_id,
                date: node_data.date,
                strength_overall: node_data.strength_overall,
                strength_hard: node_data.strength_hard,
                strength_clay: node_data.strength_clay,
                strength_grass: node_data.strength_grass,
                matches_used: node_data.matches_used
            })
            WITH bt, node_data
            MATCH (p:Player {id: node_data.player_id})
            CREATE (p)-[:HAS_BT]->(bt)
            """
            
            # batching
            batch_size = 1000
            for i in range(0, len(bt_nodes), batch_size):
                batch = bt_nodes[i:i+batch_size]
                session.run(query, nodes=batch)
        
        return len(bt_nodes)
    
    def get_mondays_in_range(self, start_date, end_date):
        #get all Monday dates between start & end
        mondays = []
        
        # convert to datetime
        start = datetime.strptime(str(start_date), '%Y%m%d')
        end = datetime.strptime(str(end_date), '%Y%m%d')
        
        # find first Monday
        current = start
        while current.weekday() != 0: # 0 = Monday
            current += timedelta(days=1)
        
        # collect all Mondays
        while current <= end:
            mondays.append(int(current.strftime('%Y%m%d')))
            current += timedelta(weeks=1)
        
        return mondays
    
    def import_bradley_terry_nodes(self):
        # main import func
        print("\nStarting Bradley-Terry strength calculation...")
        print("This will take some time as we compute MLE for each week...")
        
        # clear existing nodes
        self.clear_existing_bt_nodes()
        
        # get date range
        start_date = 20000101
        end_date = 20241231
        
        # get all Mondays
        all_mondays = self.get_mondays_in_range(start_date, end_date)
        print(f"\nProcessing {len(all_mondays)} weekly snapshots from {start_date} to {end_date}")
        
        total_nodes = 0
        
        # process each Monday
        with tqdm(total=len(all_mondays), desc="Computing weekly BT strengths") as pbar:
            for monday in all_mondays:
                # compute strengths for this week
                bt_nodes = self.compute_weekly_snapshot(monday)
                
                # save to db
                if bt_nodes:
                    count = self.save_snapshot(bt_nodes)
                    total_nodes += count
                
                # update progress bar with year info
                year = monday // 10000
                pbar.set_description(f"Year {year}: Processing week {monday}")
                pbar.update(1)
        
        print(f"\nBRADLEY-TERRY IMPORT COMPLETE!")
        print(f"Created {total_nodes:,} BT strength records")
    
    def verify_import(self):
        # verify import + show sample data
        print("\n" + "=" * 70)
        print("VERIFICATION: Sample Bradley-Terry Strengths")
        print("=" * 70)
        
        with self.driver.session() as session:
            # get most recent date
            result = session.run("""
                MATCH (bt:BradleyTerry)
                RETURN max(bt.date) as latest_date, 
                       min(bt.date) as earliest_date,
                       count(DISTINCT bt.date) as n_weeks
            """)
            
            record = result.single()
            if record and record['latest_date']:
                print(f"Date range: {record['earliest_date']} to {record['latest_date']}")
                print(f"Total weeks with data: {record['n_weeks']}")
                
                # show top players by overall strength at end
                print(f"\nTop 10 players by overall strength (as of {record['latest_date']}):")
                
                result = session.run("""
                    MATCH (p:Player)-[:HAS_BT]->(bt:BradleyTerry)
                    WHERE bt.date = $date
                    RETURN p.first_name + ' ' + p.last_name as name,
                           bt.strength_overall as overall,
                           bt.strength_hard as hard,
                           bt.strength_clay as clay,
                           bt.strength_grass as grass
                    ORDER BY bt.strength_overall DESC
                    LIMIT 10
                """, date=record['latest_date'])
                
                print(f"\n{'Rank':<5} {'Player':<20} {'Overall':<10} {'Hard':<10} {'Clay':<10} {'Grass':<10}")
                print("-" * 75)
                
                for i, r in enumerate(result, 1):
                    print(f"{i:<5} {r['name']:<20} {r['overall']:<10.3f} "
                          f"{r['hard']:<10.3f} {r['clay']:<10.3f} {r['grass']:<10.3f}")
            else:
                print("No BT data found!")

def main():
    importer = BradleyTerryImporter()
    try:
        # run import
        importer.import_bradley_terry_nodes()
        # verify
        importer.verify_import()
        
    except Exception as e:
        print(f"Error during import: {e}")
        import traceback
        traceback.print_exc()
    finally:
        importer.close()

if __name__ == "__main__":
    main()