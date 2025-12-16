#!/usr/bin/env python3

# script to implement Elo as close to TennisAbstract implementation as possible
"""
Following Tennis Abstract's methodology:
- K = 250 / ((matches + 5)^0.4)
- Absence penalty: 100 points (8-10 weeks), 150 points (30-52 weeks)
- Post-return K multiplier: 1.5x declining to 1x over 20 matches
- Grand Slam bonus: 1.1x
- Surface: 50/50 blend of overall and surface ratings for expected score
- December excluded as offseason
- Stores last_match_date in ELO nodes for prediction model use
"""

import os
import sys
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Neo4j connection parameters
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "put_your_password_here"

class TennisAbstractELO:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Elo params
        self.STARTING_RATING = 1500
        
        # for tracking player data:
        self.player_data = {}
        
        print("=" * 70)
        print("TENNIS ABSTRACT ELO IMPLEMENTATION")
        print("=" * 70)
        print("Connected to Neo4j database.")
    
    def close(self):
        self.driver.close()
    
    def clear_existing_elo(self):
        # clear existing Elo rating nodes + relationships
        print("\nClearing existing ELO ratings...")
        
        with self.driver.session() as session:
            check_query = """
            MATCH (e:ELO)
            RETURN count(e) as count
            """
            result = session.run(check_query)
            count = result.single()['count']
            
            if count > 0:
                print(f"Found {count:,} existing ELO nodes to delete...")
                batch_size = 10000
                deleted = 0
                while True:
                    result = session.run("""
                        MATCH (e:ELO)
                        WITH e LIMIT $batch_size
                        DETACH DELETE e
                        RETURN count(e) as deleted
                    """, batch_size=batch_size)
                    
                    batch_deleted = result.single()['deleted']
                    deleted += batch_deleted
                    if batch_deleted == 0:
                        break
                    if deleted % 50000 == 0:
                        print(f"  Deleted {deleted:,} nodes...")
                
                print(f"Cleared {deleted:,} ELO nodes.")
            else:
                print("No existing ELO nodes found.")
    
    def get_all_matches_chronological(self):
        # get all matches in chronological order
        with self.driver.session() as session:
            query = """
            MATCH (winner:Player)-[:WON]->(m:Match)<-[:LOST]-(loser:Player)
            MATCH (m)-[:PLAYED_IN]->(t:Tournament)
            WHERE t.date >= 20000101
            RETURN 
                winner.id as winner_id,
                loser.id as loser_id,
                t.date as match_date,
                t.surface as surface,
                t.level as tournament_level,
                m.round as round,
                m.best_of as best_of,
                m.match_num as match_num
            ORDER BY t.date, m.match_num
            """
            
            result = session.run(query)
            matches = list(result)
            print(f"Found {len(matches):,} matches to process")
            return matches
    
    def calculate_k_factor(self, matches_played):
        # TennisAbstract k-factor fomula: K = 250 / ((matches_played + 5)^0.4)
        return 250 / ((matches_played + 5) ** 0.4)
    
    def calculate_expected_score(self, rating_a, rating_b):
        # calc epxected score for player A vs. player B
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def normalize_surface(self, surface):
        #normalize surface names
        if surface is None or surface == '':
            return 'hard'
        
        surface_lower = surface.lower()
        if 'hard' in surface_lower or 'carpet' in surface_lower: return 'hard'
        elif 'clay' in surface_lower: return 'clay'
        elif 'grass' in surface_lower: return 'grass'
        else: return 'hard'
    
    def initialize_player(self, player_id):
        # initialize new players ratings
        if player_id not in self.player_data:
            self.player_data[player_id] = {
                'overall': self.STARTING_RATING,
                'hard': self.STARTING_RATING,
                'clay': self.STARTING_RATING,
                'grass': self.STARTING_RATING,
                'matches_played': 0,
                'matches_played_surface': {'hard': 0, 'clay': 0, 'grass': 0},
                'surface_initialized': {'hard': False, 'clay': False, 'grass': False},
                'last_match_date': None,
                'matches_since_return': 20, #start at 20 (no boost)
                'last_layoff_date': None,
                'cumulative_layoff_weeks': 0
            }
    
    def calculate_weeks_off(self, last_date, current_date):
        #calc weeks off (excluding dec offseason)
        last_dt = datetime.strptime(str(last_date), '%Y%m%d')
        curr_dt = datetime.strptime(str(current_date), '%Y%m%d')
        
        # if break spans dec, subtract those weeks
        weeks_off = 0
        temp_date = last_dt
        
        while temp_date < curr_dt:
            next_week = temp_date + timedelta(weeks=1)
            if next_week > curr_dt:
                next_week = curr_dt
            
            # don't count weeks in dec
            if temp_date.month != 12 and next_week.month != 12:
                weeks_off += (next_week - temp_date).days / 7
            elif temp_date.month != 12 and next_week.month == 12:
                # partial week before dec
                dec_start = datetime(next_week.year, 12, 1)
                weeks_off += (dec_start - temp_date).days / 7
            elif temp_date.month == 12 and next_week.month != 12:
                # partial week after dec
                jan_start = datetime(next_week.year, 1, 1)
                weeks_off += (next_week - jan_start).days / 7
            # if both in Ddec, don't count
            
            temp_date = next_week
        
        return weeks_off
    
    def calculate_absence_penalty(self, weeks_off):
        # absence penalty: 
        # - < 8 weeks: no penalty
        # - 8-10 weeks: 100 points
        # - 10-52 weeks: linear interpolation from 100 to 150
        # - > 52 weeks: 150 points
        if weeks_off < 8:
            return 0
        elif weeks_off <= 10:
            return 100
        elif weeks_off <= 52:
            # linear from 100 to 150 over 10-52 weeks
            return 100 + (weeks_off - 10) * (50 / 42)
        else:
            return 150
    
    def apply_absence_penalty(self, player_id, current_date):
        # apply absence penalty (for multiple layoffs within 2 years: combine lengths)
        player = self.player_data[player_id]
        
        if player['last_match_date'] is None:
            return False, 0
        
        weeks_off = self.calculate_weeks_off(player['last_match_date'], current_date)
        
        if weeks_off < 8:
            return False, 0
        
        # check for multiple layoffs w/in 2 years
        penalty = 0
        if player['last_layoff_date'] is not None:
            last_layoff = datetime.strptime(str(player['last_layoff_date']), '%Y%m%d')
            curr_date_dt = datetime.strptime(str(current_date), '%Y%m%d')
            
            if (curr_date_dt - last_layoff).days < 730:  # w/in 2 years
                # combine layoffs
                combined_weeks = player['cumulative_layoff_weeks'] + weeks_off
                combined_penalty = self.calculate_absence_penalty(combined_weeks)
                previous_penalty = self.calculate_absence_penalty(player['cumulative_layoff_weeks'])
                penalty = combined_penalty - previous_penalty
                player['cumulative_layoff_weeks'] = combined_weeks
            else:
                # new independent layoff
                penalty = self.calculate_absence_penalty(weeks_off)
                player['cumulative_layoff_weeks'] = weeks_off
        else:
            # 1st layoffs
            penalty = self.calculate_absence_penalty(weeks_off)
            player['cumulative_layoff_weeks'] = weeks_off
        
        if penalty > 0:
            # apply penalty ONLY to overall rating
            player['overall'] -= penalty
            player['overall'] = max(1000, player['overall']) #floor at 1000
            
            # reset matches since return for k-factor boost
            player['matches_since_return'] = 0
            player['last_layoff_date'] = current_date
            
            return True, penalty
        
        return False, 0
    
    def get_post_layoff_k_multiplier(self, matches_since_return):
        # post-layoff k multiplier: 
        # - 1.5x for first match back
        # - Linear decline to 1x over 20 matches
        if matches_since_return >= 20: return 1.0
        else: return 1.5 - (0.5 * matches_since_return / 20)
    
    def update_elo(self, winner_id, loser_id, surface, tournament_level, match_date):
        # update elo ratings after a match
        # initialize players if needed
        self.initialize_player(winner_id)
        self.initialize_player(loser_id)
        
        # check for absences and apply penalties
        winner_had_layoff, winner_penalty = self.apply_absence_penalty(winner_id, match_date)
        loser_had_layoff, loser_penalty = self.apply_absence_penalty(loser_id, match_date)
        
        # get player data
        winner = self.player_data[winner_id]
        loser = self.player_data[loser_id]
        
        # normalize surface
        surface_norm = self.normalize_surface(surface)
        
        # initialize surface ratings to overall if first time on surface
        # TennisAbstract method: surface ratings start at overall rating when first played
        if not winner['surface_initialized'][surface_norm]:
            winner[surface_norm] = winner['overall']
            winner['surface_initialized'][surface_norm] = True
        
        if not loser['surface_initialized'][surface_norm]:
            loser[surface_norm] = loser['overall']
            loser['surface_initialized'][surface_norm] = True
        
        # store ORIGINAL ratings before any updates
        winner_overall_orig = winner['overall']
        loser_overall_orig = loser['overall']
        winner_surface_orig = winner[surface_norm]
        loser_surface_orig = loser[surface_norm]
        
        # calculate base k-factors
        k_winner_base = self.calculate_k_factor(winner['matches_played'])
        k_loser_base = self.calculate_k_factor(loser['matches_played'])
        
        # apply Grand Slam bonus (only tournament adjustment in TennisAbstract)
        if tournament_level == 'G':
            k_winner_base *= 1.1
            k_loser_base *= 1.1
        
        # apply post-layoff k multiplier
        k_winner = k_winner_base * self.get_post_layoff_k_multiplier(winner['matches_since_return'])
        k_loser = k_loser_base * self.get_post_layoff_k_multiplier(loser['matches_since_return'])
        
        #update OVERALL ratings (overall vs overall)
        winner_expected_overall = self.calculate_expected_score(winner_overall_orig, loser_overall_orig)
        winner['overall'] += k_winner * (1 - winner_expected_overall)
        loser['overall'] += k_loser * (0 - (1 - winner_expected_overall))
        
        # update SURFACE ratings
        # TennisAbstract: Use 50/50 blend for expected score calculation
        winner_blended = 0.5 * winner_overall_orig + 0.5 * winner_surface_orig
        loser_blended = 0.5 * loser_overall_orig + 0.5 * loser_surface_orig
        winner_expected_surface = self.calculate_expected_score(winner_blended, loser_blended)
        
        # surface k-factors based on surface-specific match counts
        k_winner_surface = self.calculate_k_factor(winner['matches_played_surface'][surface_norm])
        k_loser_surface = self.calculate_k_factor(loser['matches_played_surface'][surface_norm])
        
        #Grand Slam bonus for surface ratings too
        if tournament_level == 'G':
            k_winner_surface *= 1.1
            k_loser_surface *= 1.1
        
        # apply post-layoff multiplier to surface k too
        k_winner_surface *= self.get_post_layoff_k_multiplier(winner['matches_since_return'])
        k_loser_surface *= self.get_post_layoff_k_multiplier(loser['matches_since_return'])
        
        # update surface ratings
        winner[surface_norm] += k_winner_surface * (1 - winner_expected_surface)
        loser[surface_norm] += k_loser_surface * (0 - (1 - winner_expected_surface))
        
        # update match counts
        winner['matches_played'] += 1
        loser['matches_played'] += 1
        winner['matches_played_surface'][surface_norm] += 1
        loser['matches_played_surface'][surface_norm] += 1
        
        # update dates and matches since return
        winner['last_match_date'] = match_date
        loser['last_match_date'] = match_date
        winner['matches_since_return'] = min(20, winner['matches_since_return'] + 1)
        loser['matches_since_return'] = min(20, loser['matches_since_return'] + 1)
    
    def get_mondays_in_range(self, start_date, end_date):
        # get all Monday dates betw/ start and end
        mondays = []
        
        start = datetime.strptime(str(start_date), '%Y%m%d')
        end = datetime.strptime(str(end_date), '%Y%m%d')
        
        # find first Monday
        current = start
        while current.weekday() != 0: # 0 == Monday
            current += timedelta(days=1)
        
        # collect all Mondays
        while current <= end:
            mondays.append(int(current.strftime('%Y%m%d')))
            current += timedelta(weeks=1)
        
        return mondays
    
    def save_weekly_snapshot(self, snapshot_date):
        # save Elo ratings snapshot for a given date
        elo_nodes = []
        
        # only save players who have played at least one match
        for player_id, data in self.player_data.items():
            if data['matches_played'] > 0:
                elo_nodes.append({
                    'player_id': player_id,
                    'date': snapshot_date,
                    'overall': round(data['overall'], 1),
                    'hard': round(data['hard'], 1),
                    'clay': round(data['clay'], 1),
                    'grass': round(data['grass'], 1),
                    'matches_played': data['matches_played'],
                    'matches_since_return': data['matches_since_return'],
                    'last_match_date': data['last_match_date'] if data['last_match_date'] else 0
                })
        
        if elo_nodes:
            with self.driver.session() as session:
                query = """
                UNWIND $nodes AS node_data
                CREATE (e:ELO {
                    player_id: node_data.player_id,
                    date: node_data.date,
                    overall: node_data.overall,
                    hard: node_data.hard,
                    clay: node_data.clay,
                    grass: node_data.grass,
                    matches_played: node_data.matches_played,
                    matches_since_return: node_data.matches_since_return,
                    last_match_date: node_data.last_match_date
                })
                WITH e, node_data
                MATCH (p:Player {id: node_data.player_id})
                CREATE (p)-[:HAS_ELO]->(e)
                """
                
                # batching!
                batch_size = 1000
                for i in range(0, len(elo_nodes), batch_size):
                    batch = elo_nodes[i:i+batch_size]
                    session.run(query, nodes=batch)
        
        return len(elo_nodes)
    
    def import_elo_ratings(self):
        # main import function
        print("\nStarting Tennis Abstract ELO calculation...")
        print("Key features:")
        print("- K-factor: 250 / ((matches + 5)^0.4)")
        print("- Grand Slam bonus: 1.1x")
        print("- Absence penalty: 100-150 points based on weeks off")
        print("- December excluded as offseason")
        print("- Post-layoff K boost: 1.5x declining over 20 matches")
        print("- Surface: 50/50 blend for expected scores")
        print("- Storing last_match_date in ELO nodes")
        
        # clear existing ELO nodes
        self.clear_existing_elo()
        
        # get all matches
        matches = self.get_all_matches_chronological()
        
        if not matches:
            print("No matches found!")
            return
        
        # process matches + create weekly snapshots
        print("\nProcessing matches and creating weekly snapshots...")
        
        total_matches = len(matches)
        matches_processed = 0
        total_snapshots = 0
        
        #get date range
        first_date = matches[0]['match_date']
        last_date = matches[-1]['match_date']
        
        #get all Mondays (weekly snapshots)
        all_mondays = self.get_mondays_in_range(first_date, last_date)
        print(f"Creating snapshots for {len(all_mondays)} weeks")
        
        # process matches week by week
        monday_idx = 0
        
        with tqdm(total=total_matches, desc="Processing matches") as pbar:
            for match in matches:
                # update ELO for this match
                self.update_elo(
                    match['winner_id'],
                    match['loser_id'],
                    match['surface'],
                    match['tournament_level'],
                    match['match_date']
                )
                
                matches_processed += 1
                pbar.update(1)
                
                # check if should save snapshot
                while (monday_idx < len(all_mondays) and 
                       match['match_date'] >= all_mondays[monday_idx]):
                    # save snapshot for this Monday
                    snapshot_count = self.save_weekly_snapshot(all_mondays[monday_idx])
                    total_snapshots += snapshot_count
                    
                    if monday_idx % 52 == 0: # yearly update
                        year = all_mondays[monday_idx] // 10000
                        active_players = len([p for p in self.player_data.values() 
                                            if p['matches_played'] > 0])
                        pbar.set_description(f"Year {year}: {active_players} active players")
                    
                    monday_idx += 1
        
        # save final snapshot (if needed)
        if monday_idx < len(all_mondays):
            snapshot_count = self.save_weekly_snapshot(all_mondays[-1])
            total_snapshots += snapshot_count
        
        print(f"\nTENNIS ABSTRACT ELO IMPORT COMPLETE!")
        print(f"Processed {matches_processed:,} matches")
        print(f"Created {total_snapshots:,} ELO rating records")
        print(f"Active players: {len([p for p in self.player_data.values() if p['matches_played'] > 0]):,}")
    
    def verify_and_show_top_players(self):
        # verify import + show current top ACTIVE players
        print("\n" + "=" * 80)
        print("TENNIS ABSTRACT ELO - Top 20 Active Players (December 2024)")
        print("=" * 80)
        
        with self.driver.session() as session:
            # get most recent date
            result = session.run("""
                MATCH (e:ELO)
                RETURN max(e.date) as latest_date""")
            latest_date = result.single()['latest_date']
            
            if not latest_date:
                print("No ELO data found!")
                return
            
            # calc cutoff date for active players (played in last 365 days)
            from datetime import datetime, timedelta
            latest = datetime.strptime(str(latest_date), '%Y%m%d')
            cutoff = latest - timedelta(days=365)
            cutoff_date = int(cutoff.strftime('%Y%m%d'))
            
            print(f"Latest ratings date: {latest_date}")
            print(f"Showing players active since: {cutoff_date}")
            print("\n{:<4} {:<20} {:<8} {:<8} {:<8} {:<8} {:<8}".format("Rank", "Player", "Overall", "Hard", "Clay", "Grass", "Matches"))
            print("-" * 76)
            
            # get top 20 ACTIVE players by overall ELO
            result = session.run("""
                MATCH (p:Player)-[:HAS_ELO]->(e:ELO)
                WHERE e.date = $date
                WITH p, e
                MATCH (p)-[:WON|LOST]->(m:Match)-[:PLAYED_IN]->(t:Tournament)
                WHERE t.date >= $cutoff
                WITH p, e, max(t.date) as last_match
                RETURN DISTINCT p.first_name + ' ' + p.last_name as name,
                       e.overall as overall,
                       e.hard as hard,
                       e.clay as clay,
                       e.grass as grass,
                       e.matches_played as matches,
                       last_match
                ORDER BY e.overall DESC
                LIMIT 20
            """, date=latest_date, cutoff=cutoff_date)
            
            for i, record in enumerate(result, 1):
                print(f"{i:<4} {record['name']:<20} {record['overall']:<8.1f} "
                      f"{record['hard']:<8.1f} {record['clay']:<8.1f} "
                      f"{record['grass']:<8.1f} {record['matches']:<8}")

def main():
    importer = TennisAbstractELO()
    try:
        # import ELO ratings
        importer.import_elo_ratings()
        # show verify
        importer.verify_and_show_top_players()
        
    except Exception as e:
        print(f"Error during import: {e}")
        import traceback
        traceback.print_exc()
    finally:
        importer.close()

if __name__ == "__main__":
    main()