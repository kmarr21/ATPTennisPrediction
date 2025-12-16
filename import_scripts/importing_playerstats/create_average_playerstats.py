#!/usr/bin/env python3

# script to create avg PlayerStats nodes for top 10/20/50/100/200 players based on weekly ATP rankings at each point in time

from neo4j import GraphDatabase
import numpy as np
from datetime import datetime
from tqdm import tqdm
import gc

class AverageStatsCreator:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="put_your_password_here"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.top_n_groups = [10, 20, 50, 100, 200]
        
    def close(self):
        self.driver.close()
    
    def get_unique_dates(self):
        # get all unique dates that have PlayerStats
        with self.driver.session() as session:
            result = session.run("""
                MATCH (ps:PlayerStats)
                RETURN DISTINCT ps.date as date
                ORDER BY date
            """)
            dates = [record['date'] for record in result]
            print(f"Found {len(dates)} unique dates with PlayerStats")
            return dates
    
    def delete_existing_average_nodes(self):
        # SAFETLY delete only existing avg playerstats nodes
        with self.driver.session() as session:
            # Count them first
            count_result = session.run("""
                MATCH (ps:PlayerStats)
                WHERE ps.is_average = true OR ps.player_id STARTS WITH 'avg_top_'
                RETURN count(ps) as count
            """)
            count = count_result.single()['count']
            
            if count > 0:
                print(f"\nDeleting {count} existing average nodes...")
                # delete ONLY nodes with is_average flag or avg_top_ player_id
                session.run("""
                    MATCH (ps:PlayerStats)
                    WHERE ps.is_average = true OR ps.player_id STARTS WITH 'avg_top_'
                    DELETE ps
                """)
                print(f"  Deleted {count} average nodes")
                
                # verify no regular playerstats were deleted...
                verify_result = session.run("""
                    MATCH (ps:PlayerStats)
                    RETURN count(ps) as remaining
                """)
                remaining = verify_result.single()['remaining']
                print(f"  Remaining PlayerStats nodes: {remaining:,}")
            else:
                print("\nNo existing average nodes to delete")
    
    def get_top_players_at_date(self, date, top_n):
        # get top N players by ranking at/before the given date
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Player)-[:HAS_RANKING]->(r:Ranking)
                WHERE r.date <= $date AND r.rank <= $top_n
                WITH p.id as player_id, r.rank as ranking, r.date as rank_date
                ORDER BY rank_date DESC, ranking
                WITH player_id, COLLECT({rank: ranking, date: rank_date})[0] as latest_ranking
                WHERE latest_ranking.rank <= $top_n
                RETURN player_id
                ORDER BY latest_ranking.rank
                LIMIT $top_n
            """, date=date, top_n=top_n)
            
            player_ids = [record['player_id'] for record in result]
            return player_ids
    
    def get_player_stats_for_averaging(self, date, player_ids):
        # get all 52w stats for specified players at given date
        if not player_ids: return []
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (ps:PlayerStats)
                WHERE ps.date = $date AND ps.player_id IN $player_ids
                RETURN ps
            """, date=date, player_ids=player_ids)
            
            stats_list = []
            for record in result:
                if record['ps']:
                    stats_dict = dict(record['ps'])
                    # Only keep 52w stats for averaging
                    stats_52w = {k: v for k, v in stats_dict.items() 
                                if 'stats_52w_' in k and v is not None}
                    if stats_52w: stats_list.append(stats_52w)
            
            return stats_list
    
    def calculate_average_stats(self, stats_list):
        # calc avg of all numeric 52w stats
        if not stats_list: return {}
        
        # collect all unique stat keys
        all_keys = set()
        for stats in stats_list:
            all_keys.update(stats.keys())
        
        # calc avgs
        avg_stats = {}
        for key in all_keys:
            values = []
            for stats in stats_list:
                if key in stats and stats[key] is not None:
                    # only include numeric values!!
                    if isinstance(stats[key], (int, float)):
                        values.append(stats[key])
            
            if values: avg_stats[key] = np.mean(values)
        
        return avg_stats
    
    def create_average_node(self, date, top_n, avg_stats, num_players):
        """Create a single average PlayerStats node"""
        if not avg_stats:
            return False
        
        with self.driver.session() as session:
            # create node with a special player_id
            player_id = f"avg_top_{top_n}"
            
            # build props string
            props = {
                'player_id': player_id,
                'date': date,
                'is_average': True,
                'top_n': top_n,
                'num_players_averaged': num_players
            }
            props.update(avg_stats)
            
            # create node
            result = session.run("""
                CREATE (ps:PlayerStats $props)
                RETURN ps
            """, props=props)
            
            return result.single() is not None
    
    def process_date(self, date):
        # process all top N groups for a single date
        results = {}
        
        for top_n in self.top_n_groups:
            # get top N players at this date
            player_ids = self.get_top_players_at_date(date, top_n)
            
            if len(player_ids) < min(10, top_n): #need at least 10 players or top_n if smaller......
                results[top_n] = {'status': 'skipped', 'reason': f'Only {len(player_ids)} players found'}
                continue
            
            # get the stats
            stats_list = self.get_player_stats_for_averaging(date, player_ids)
            
            if not stats_list:
                results[top_n] = {'status': 'skipped', 'reason': 'No stats found'}
                continue
            
            # calc avgs
            avg_stats = self.calculate_average_stats(stats_list)
            
            # create node
            success = self.create_average_node(date, top_n, avg_stats, len(stats_list))
            
            if success:
                results[top_n] = {
                    'status': 'created',
                    'num_players': len(stats_list),
                    'num_stats': len(avg_stats)
                }
            else:
                results[top_n] = {'status': 'failed'}
        
        return results
    
    def create_indexes(self):
        # create indices for efficiency
        with self.driver.session() as session:
            #index for finding avg. nodes
            session.run("""
                CREATE INDEX avg_stats_lookup IF NOT EXISTS 
                FOR (ps:PlayerStats) ON (ps.is_average, ps.top_n, ps.date)
            """)
            print("Created indexes for average PlayerStats")
    
    def run(self, start_date=None, end_date=None):
        # main exec
        print("="*70)
        print("CREATING AVERAGE PLAYERSTATS NODES")
        print("="*70)
        
        # DELETE EXISTING AVERAGE NODES FIRST
        self.delete_existing_average_nodes()
        
        # create indices
        self.create_indexes()
        
        #get all dates
        all_dates = self.get_unique_dates()
        
        # filter dates if specified
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]
        
        print(f"Processing {len(all_dates)} dates from {all_dates[0]} to {all_dates[-1]}")
        print(f"Creating averages for top: {self.top_n_groups}")
        
        # process each date
        total_created = 0
        total_skipped = 0
        
        # process in batches for memory management
        BATCH_SIZE = 52 # process 1year at a time
        
        for i in range(0, len(all_dates), BATCH_SIZE):
            batch_dates = all_dates[i:i+BATCH_SIZE]
            print(f"\nProcessing batch {i//BATCH_SIZE + 1}/{(len(all_dates)-1)//BATCH_SIZE + 1}")
            
            for date in tqdm(batch_dates, desc=f"Dates {batch_dates[0]}-{batch_dates[-1]}"):
                results = self.process_date(date)
                
                for top_n, result in results.items():
                    if result['status'] == 'created':
                        total_created += 1
                    elif result['status'] == 'skipped':
                        total_skipped += 1
            
            # clean up memory
            gc.collect()
        
        print(f"\n{'='*70}")
        print(f"COMPLETE")
        print(f"{'='*70}")
        print(f"Total nodes created: {total_created}")
        print(f"Total skipped: {total_skipped}")
        
        # verify
        self.verify_creation()
    
    def verify_creation(self):
        # verify avg nodes were actually created
        with self.driver.session() as session:
            result = session.run("""
                MATCH (ps:PlayerStats)
                WHERE ps.is_average = true
                RETURN ps.top_n as top_n, count(ps) as count
                ORDER BY top_n
            """)
            
            print("\nVerification - Average nodes created:")
            for record in result:
                print(f"  Top {record['top_n']:3}: {record['count']:,} nodes")
            
            # sample check
            result = session.run("""
                MATCH (ps:PlayerStats)
                WHERE ps.is_average = true AND ps.top_n = 10
                RETURN ps.date as date, ps.stats_52w_win_pct as win_pct
                ORDER BY date DESC
                LIMIT 5
            """)
            
            print("\nSample Top-10 average win rates:")
            for record in result:
                if record['win_pct']:
                    print(f"  {record['date']}: {record['win_pct']:.1%} win rate")

def main():
    creator = AverageStatsCreator()
    
    try:
        # #1: specify date range to process
        # creator.run(start_date=20200101, end_date=20201231)
        
        # #2: process all dates
        creator.run()
        
    finally:
        creator.close()

if __name__ == "__main__":
    main()