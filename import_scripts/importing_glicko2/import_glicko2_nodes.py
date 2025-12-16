#!/usr/bin/env python3

# script to create a Glicko2 rating system for tennis
"""
Based on Professor Mark Glickman's Glicko-2 system with tennis-specific adaptations.
Creates bi-weekly rating snapshots with rating, deviation, and volatility.

References:
- Glickman, M.E. (2012). "Example of the Glicko-2 system" 
  http://www.glicko.net/glicko/glicko2.pdf
- Glickman, M.E. (2001). "Dynamic paired comparison models with stochastic variances"
  Journal of Applied Statistics, 28(6), 673-689

Tennis adaptations:
- Bi-weekly rating periods to capture tournament dynamics
- Surface-specific RD initialization
- December off-season handling
- Grand Slam weighting for informativeness
"""

import os
import sys
import argparse
from neo4j import GraphDatabase
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

class Glicko2Importer:
    def __init__(self, tau=0.4, rating_period_days=14, rd_decay=30, 
                 node_suffix="", surface_rd_init=200):
        """
        Args:
            tau: sys constant controlling volatility (0.3-1.2)
                 - 0.3 = ratings change slowly (stable)
                 - 0.5 = Glickman's default
                 - 1.2 = ratings change quickly (volatile)
            rating_period_days: days per rating period (7, 14, or 30)
            rd_decay: RD growth per period when inactive
            node_suffix: suffix for node labels (for variants)
            surface_rd_init: onitial RD when playing new surface
        """
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # config parameters
        self.TAU = tau
        self.RATING_PERIOD_DAYS = rating_period_days
        self.RD_DECAY = rd_decay
        self.NODE_SUFFIX = node_suffix # e.g., "_stable" or "_volatile"
        self.SURFACE_RD_INIT = surface_rd_init # tennis-specific
        
        # fixed Glicko-2 Parameters (from Glickman 2012)
        self.MU = 0.0  # initial rating (on Glicko-2 scale)
        self.PHI = 2.014761  # initial RD: 350/173.7178 for Glicko-2 scale
        self.SIGMA = 0.06  # initial volatility (Glickman recommendation)
        self.EPSILON = 0.000001  # convergence tolerance for Illinois algorithm
        
        # conversion constants (Glickman 2012, equation 2)
        self.SCALE = 173.7178  # converts between Glicko-1 and Glicko-2
        self.BASE_RATING = 1500  # center of rating scale
        
        # tracking player data
        self.player_data = {}
        
        # tracking rating periods
        self.rating_periods = []
        self.current_period_matches = defaultdict(lambda: defaultdict(list))
        
        # surface tracking
        self.surfaces = ['hard', 'clay', 'grass']
        
        # node label for this variant
        self.node_label = f"Glicko2{self.NODE_SUFFIX}"
        self.relationship_type = f"HAS_GLICKO2{self.NODE_SUFFIX}"
        
        print("=" * 70)
        print("GLICKO-2 RATING SYSTEM FOR TENNIS")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  TAU (volatility): {self.TAU}")
        print(f"  Rating period: {self.RATING_PERIOD_DAYS} days")
        print(f"  RD decay rate: {self.RD_DECAY} points/period")
        print(f"  Surface RD init: {self.SURFACE_RD_INIT}")
        print(f"  Node label: {self.node_label}")
        print("Connected to Neo4j database.")
    
    def close(self):
        self.driver.close()
    
    def clear_existing_glicko2(self):
        # clear any existing Glicko2 rating nodes for this variant
        print(f"\nClearing existing {self.node_label} nodes...")
        
        with self.driver.session() as session:
            result = session.run(f"MATCH (g:{self.node_label}) RETURN count(g) as count")
            count = result.single()['count']
            
            if count > 0:
                print(f"Found {count:,} existing {self.node_label} nodes to delete...")
                
                batch_size = 10000
                deleted = 0
                
                while True:
                    result = session.run(f"""
                        MATCH (g:{self.node_label})
                        WITH g LIMIT $batch_size
                        DETACH DELETE g
                        RETURN count(g) as deleted
                    """, batch_size=batch_size)
                    
                    batch_deleted = result.single()['deleted']
                    deleted += batch_deleted
                    
                    if batch_deleted == 0:
                        break
                    
                    if deleted % 50000 == 0:
                        print(f"  Deleted {deleted:,} nodes...")
                
                print(f"Cleared {deleted:,} {self.node_label} nodes.")
            else:
                print(f"No existing {self.node_label} nodes found.")
    
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
    
    def glicko_to_glicko2(self, rating, rd):
        # convert from Glicko scale to Glicko2 scale (Glickman 2012, eq. 2)
        mu = (rating - self.BASE_RATING) / self.SCALE
        phi = rd / self.SCALE
        return mu, phi
    
    def glicko2_to_glicko(self, mu, phi):
        # convert from Glicko2 scale to Glicko scale
        rating = self.SCALE * mu + self.BASE_RATING
        rd = self.SCALE * phi
        return rating, rd
    
    def g_function(self, phi):
        #gravity function (Glickman 2012, eq. 3)
        return 1 / math.sqrt(1 + 3 * phi * phi / (math.pi * math.pi))
    
    def e_function(self, mu, mu_j, phi_j):
        #expec score function (Glickman 2012, eq. 4)
        return 1 / (1 + math.exp(-self.g_function(phi_j) * (mu - mu_j)))
    
    def compute_variance(self, g_values, e_values):
        # compute variance of performance (Glickman 2012, eq. 5)
        variance = 0
        for g, e in zip(g_values, e_values):
            variance += g * g * e * (1 - e)
        return 1 / variance if variance > 0 else 1e10
    
    def compute_delta(self, variance, g_values, scores, e_values):
        # compute performance rating difference (Glickman 2012, eq. 6)
        delta = 0
        for g, s, e in zip(g_values, scores, e_values):
            delta += g * (s - e)
        return variance * delta
    
    def compute_volatility(self, sigma, delta, phi, variance):
        # compute new volatility using Illinois algorithm (Glickman 2012, Section 5.1)
        a = math.log(sigma * sigma)
        
        def f(x):
            ex = math.exp(x)
            term1 = ex * (delta * delta - phi * phi - variance - ex)
            term2 = 2 * (phi * phi + variance + ex) ** 2
            term3 = x - a
            term4 = self.TAU * self.TAU
            return term1 / term2 - term3 / term4
        
        # find bracket for root
        A = a
        if delta * delta > phi * phi + variance:
            B = math.log(delta * delta - phi * phi - variance)
        else:
            k = 1
            while f(a - k * self.TAU) < 0:
                k += 1
            B = a - k * self.TAU
        
        # Illinois algorithm for root finding
        fa = f(A)
        fb = f(B)
        
        while abs(B - A) > self.EPSILON:
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            
            if fc * fb <= 0:
                A = B
                fa = fb
            else:
                fa = fa / 2
            
            B = C
            fb = fc
        
        return math.exp(A / 2)
    
    def calculate_weeks_off(self, last_date, current_date):
        # calc periods off, excuding dec (~tennis off-season)
        # similar as our Elo approach!
        last_dt = datetime.strptime(str(last_date), '%Y%m%d')
        curr_dt = datetime.strptime(str(current_date), '%Y%m%d')
        
        periods_off = 0
        temp_date = last_dt
        
        while temp_date < curr_dt:
            next_period = temp_date + timedelta(days=self.RATING_PERIOD_DAYS)
            if next_period > curr_dt:
                next_period = curr_dt
            
            # don't count dec periods
            if temp_date.month != 12 and next_period.month != 12:
                periods_off += 1
            elif temp_date.month != 12 and next_period.month == 12:
                # partial period before dec
                periods_off += 0.5
            elif temp_date.month == 12 and next_period.month != 12:
                # partial period after dec
                periods_off += 0.5
            
            temp_date = next_period
        
        return periods_off
    
    def apply_rd_decay(self, phi, periods_inactive):
        # apply RD decay for periods of inactivity
        # tennis-specific: slower decay during off-season

        # convert decay rate to Glicko2 scale
        c_sq = (self.RD_DECAY / self.SCALE) ** 2
        
        # calc new RD with decay
        new_phi = math.sqrt(phi * phi + periods_inactive * c_sq)
        
        #cap at initial RD
        return min(new_phi, self.PHI)
    
    def initialize_player(self, player_id):
        #initialize a new player's ratings
        if player_id not in self.player_data:
            self.player_data[player_id] = {
                'overall': {'mu': self.MU, 'phi': self.PHI, 'sigma': self.SIGMA},
                'hard': {'mu': self.MU, 'phi': self.PHI, 'sigma': self.SIGMA},
                'clay': {'mu': self.MU, 'phi': self.PHI, 'sigma': self.SIGMA},
                'grass': {'mu': self.MU, 'phi': self.PHI, 'sigma': self.SIGMA},
                'surface_initialized': {'hard': False, 'clay': False, 'grass': False},
                'last_period_played': None,
                'total_matches': 0,
                'surface_matches': defaultdict(int),
                'inactive_periods': 0}
    
    def normalize_surface(self, surface):
        #normalize surface names
        if surface is None or surface == '': return 'hard'
        
        surface_lower = surface.lower()
        if 'hard' in surface_lower or 'carpet' in surface_lower: return 'hard'
        elif 'clay' in surface_lower: return 'clay'
        elif 'grass' in surface_lower: return 'grass'
        else: return 'hard'
    
    def get_rating_period(self, match_date):
        #get the rating period for a given date
        date_obj = datetime.strptime(str(match_date), '%Y%m%d')
        
        # calc period start (align to Mondays)
        days_since_epoch = (date_obj - datetime(2000, 1, 3)).days # Jan 3, 2000 was Monday
        period_number = days_since_epoch // self.RATING_PERIOD_DAYS
        
        period_start = datetime(2000, 1, 3) + timedelta(days=period_number * self.RATING_PERIOD_DAYS)
        period_end = period_start + timedelta(days=self.RATING_PERIOD_DAYS - 1)
        
        return period_number, period_start, period_end
    
    def store_match_for_period(self, winner_id, loser_id, surface, tournament_level, match_date):
        #store match for batch processing in rating period
        #init players if needed
        self.initialize_player(winner_id)
        self.initialize_player(loser_id)
        
        # get rating period
        period_num, _, _ = self.get_rating_period(match_date)
        
        # normalize surface
        surface_norm = self.normalize_surface(surface)
        
        # store match for both players
        match_data = {
            'surface': surface_norm,
            'tournament_level': tournament_level,
            'date': match_date
        }
        
        # winner's perspective
        self.current_period_matches[period_num][winner_id].append({
            **match_data,
            'opponent_id': loser_id,
            'score': 1  # Won
        })
        
        # loser's perspective
        self.current_period_matches[period_num][loser_id].append({
            **match_data,
            'opponent_id': winner_id,
            'score': 0 #lost
        })
    
    def process_rating_period(self, period_num):
        #process all matches in a rating period using true Glicko2 batch update
        #THIS is the key difference from sequential updates
        matches_by_player = self.current_period_matches.get(period_num, {})
        
        if not matches_by_player: return
        
        # process each player who played in this period
        for player_id, matches in matches_by_player.items():
            player = self.player_data[player_id]
            
            # check for inactivity + apply RD decay
            if player['last_period_played'] is not None:
                periods_inactive = period_num - player['last_period_played'] - 1
                if periods_inactive > 0:
                    # apply decay to all ratings
                    for rating_type in ['overall'] + self.surfaces:
                        player[rating_type]['phi'] = self.apply_rd_decay(
                            player[rating_type]['phi'], periods_inactive
                        )
            
            # process each rating type
            for rating_type in ['overall'] + self.surfaces:
                if rating_type == 'overall':
                    # use all matches for overall rating
                    relevant_matches = matches
                else:
                    # for surface-specific, only use matches on that surface
                    relevant_matches = [m for m in matches if m['surface'] == rating_type]
                    
                    if not relevant_matches:
                        # no matches on this surface:> mild decay
                        player[rating_type]['phi'] = min(
                            player[rating_type]['phi'] * 1.02, 
                            self.PHI
                        )
                        continue
                    
                    # tennis-specific: init surface rating if first time
                    if not player['surface_initialized'][rating_type]:
                        # start surface rating at overall but w/ higher uncertainty
                        player[rating_type]['mu'] = player['overall']['mu']
                        player[rating_type]['phi'] = self.SURFACE_RD_INIT / self.SCALE
                        player['surface_initialized'][rating_type] = True
                
                # get current ratings
                mu = player[rating_type]['mu']
                phi = player[rating_type]['phi']
                sigma = player[rating_type]['sigma']
                
                #calc updates based on all matches in period
                g_values = []
                e_values = []
                scores = []
                weights = [] #for tournament importance
                
                for match in relevant_matches:
                    opp_id = match['opponent_id']
                    if opp_id not in self.player_data:
                        self.initialize_player(opp_id)
                    
                    opp_data = self.player_data[opp_id][rating_type]
                    opp_mu = opp_data['mu']
                    opp_phi = opp_data['phi']
                    
                    g_values.append(self.g_function(opp_phi))
                    e_values.append(self.e_function(mu, opp_mu, opp_phi))
                    scores.append(match['score'])
                    
                    # tennis-specific: Grand Slams are more informative
                    weight = 1.0
                    if match['tournament_level'] == 'G':
                        weight = 1.2 # Grand Slams count 20% more
                    elif match['tournament_level'] == 'M':
                        weight = 1.1 # Masters count 10% more
                    elif match['tournament_level'] == 'C':
                        weight = 0.7 #Challengers count 30% less
                    weights.append(weight)
                
                # apply weights to g_values
                g_values = [g * w for g, w in zip(g_values, weights)]
                
                # compute new values using Glicko2 formulas
                variance = self.compute_variance(g_values, e_values)
                delta = self.compute_delta(variance, g_values, scores, e_values)
                
                # update volatility (key part of Glicko2)
                sigma_prime = self.compute_volatility(sigma, delta, phi, variance)
                
                # pre-rating RD (Glickman 2012, Step 6)
                phi_star = math.sqrt(phi * phi + sigma_prime * sigma_prime)
                
                # new RD (Glickman 2012, Step 7)
                phi_prime = 1 / math.sqrt(1 / (phi_star * phi_star) + 1 / variance)
                
                # new rating (Glickman 2012, Step 7)
                mu_change = phi_prime * phi_prime * sum(g * (s - e) for g, s, e in zip(g_values, scores, e_values))
                mu_prime = mu + mu_change
                
                # store updated values
                player[rating_type]['mu'] = mu_prime
                player[rating_type]['phi'] = phi_prime
                player[rating_type]['sigma'] = sigma_prime
            
            # update tracking
            player['total_matches'] += len(matches)
            for match in matches: player['surface_matches'][match['surface']] += 1
            player['last_period_played'] = period_num
    
    def get_mondays_in_range(self, start_date, end_date):
        # get all monday dates for weekly snapshots
        mondays = []
        
        start = datetime.strptime(str(start_date), '%Y%m%d')
        end = datetime.strptime(str(end_date), '%Y%m%d')
        
        current = start
        while current.weekday() != 0: # find first Monday
            current += timedelta(days=1)
        
        while current <= end:
            mondays.append(int(current.strftime('%Y%m%d')))
            current += timedelta(weeks=1)
        
        return mondays
    
    def save_weekly_snapshot(self, snapshot_date):
        # save Glicko2 ratings for all active players
        glicko2_nodes = []
        
        for player_id, data in self.player_data.items():
            #ONLY requirement: player must have played at least one match ever
            # (same as ELO implementation:> keep all players with any history)
            if data['total_matches'] == 0:
                continue
            
            # convert to Glicko scale for storage
            overall_rating, overall_rd = self.glicko2_to_glicko(data['overall']['mu'], data['overall']['phi'])
            hard_rating, hard_rd = self.glicko2_to_glicko(data['hard']['mu'], data['hard']['phi'])
            clay_rating, clay_rd = self.glicko2_to_glicko(data['clay']['mu'], data['clay']['phi'])
            grass_rating, grass_rd = self.glicko2_to_glicko(data['grass']['mu'], data['grass']['phi'])
            
            node_data = {
                'player_id': player_id,
                'date': snapshot_date,
                'rating_overall': round(overall_rating, 1),
                'rating_hard': round(hard_rating, 1),
                'rating_clay': round(clay_rating, 1),
                'rating_grass': round(grass_rating, 1),
                'rd_overall': round(overall_rd, 1),
                'rd_hard': round(hard_rd, 1),
                'rd_clay': round(clay_rd, 1),
                'rd_grass': round(grass_rd, 1),
                'volatility_overall': round(data['overall']['sigma'], 4),
                'volatility_hard': round(data['hard']['sigma'], 4),
                'volatility_clay': round(data['clay']['sigma'], 4),
                'volatility_grass': round(data['grass']['sigma'], 4),
                'total_matches': data['total_matches']}
            
            glicko2_nodes.append(node_data)
        
        # save to db w/ dynamic labels
        if glicko2_nodes:
            with self.driver.session() as session:
                query = f"""
                UNWIND $nodes AS node_data
                CREATE (g:{self.node_label} {{
                    player_id: node_data.player_id,
                    date: node_data.date,
                    rating_overall: node_data.rating_overall,
                    rating_hard: node_data.rating_hard,
                    rating_clay: node_data.rating_clay,
                    rating_grass: node_data.rating_grass,
                    rd_overall: node_data.rd_overall,
                    rd_hard: node_data.rd_hard,
                    rd_clay: node_data.rd_clay,
                    rd_grass: node_data.rd_grass,
                    volatility_overall: node_data.volatility_overall,
                    volatility_hard: node_data.volatility_hard,
                    volatility_clay: node_data.volatility_clay,
                    volatility_grass: node_data.volatility_grass,
                    total_matches: node_data.total_matches
                }})
                WITH g, node_data
                MATCH (p:Player {{id: node_data.player_id}})
                CREATE (p)-[:{self.relationship_type}]->(g)
                """
                
                # BATCHING
                batch_size = 1000
                for i in range(0, len(glicko2_nodes), batch_size):
                    batch = glicko2_nodes[i:i+batch_size]
                    session.run(query, nodes=batch)
        
        return len(glicko2_nodes)
    
    def import_glicko2_ratings(self):
        # main import func using true rating periods
        print("\nStarting Glicko-2 rating calculation...")
        print("Using TRUE rating periods (batch processing)")
        print("Ensures meaningful volatility calculations...")
        
        # clear existing nodes
        self.clear_existing_glicko2()
        
        # get all matches
        matches = self.get_all_matches_chronological()
        
        if not matches:
            print("No matches found!")
            return
        
        print("\nProcessing matches in rating periods...")
        
        # get date range
        first_date = matches[0]['match_date']
        last_date = matches[-1]['match_date']
        
        # determine all rating periods
        first_period, _, _ = self.get_rating_period(first_date)
        last_period, _, _ = self.get_rating_period(last_date)
        total_periods = last_period - first_period + 1
        
        print(f"Processing {total_periods} rating periods ({self.RATING_PERIOD_DAYS}-day periods)")
        
        # store all matches by period
        print("Organizing matches by period...")
        for match in tqdm(matches, desc="Organizing matches"):
            self.store_match_for_period(
                match['winner_id'],
                match['loser_id'],
                match['surface'],
                match['tournament_level'],
                match['match_date']
            )
        
        # process each period + create weekly snapshots
        all_mondays = self.get_mondays_in_range(first_date, last_date)
        print(f"Creating {len(all_mondays)} weekly snapshots...")
        
        total_snapshots = 0
        current_period = first_period - 1
        monday_idx = 0
        
        with tqdm(total=len(all_mondays), desc="Creating snapshots") as pbar:
            for monday in all_mondays:
                monday_period, _, _ = self.get_rating_period(monday)
                
                # process any periods not processed yet
                while current_period < monday_period:
                    current_period += 1
                    if current_period in self.current_period_matches:
                        # process this period's matches
                        self.process_rating_period(current_period)
                        
                        # show progress!!
                        period_date = datetime(2000, 1, 3) + timedelta(
                            days=current_period * self.RATING_PERIOD_DAYS
                        )
                        num_players = len(self.current_period_matches[current_period])
                        if current_period % 26 == 0: # update every ~year
                            pbar.set_description(
                                f"Period {period_date.strftime('%Y-%m')}: "
                                f"{num_players} active players"
                            )
                
                # save snapshot for this Monday
                snapshot_count = self.save_weekly_snapshot(monday)
                total_snapshots += snapshot_count
                pbar.update(1)
        
        print(f"\nGLICKO-2 IMPORT COMPLETE!")
        print(f"Created {total_snapshots:,} rating records")
        print(f"Active players: {len([p for p in self.player_data.values() if p['total_matches'] > 0]):,}")
        print(f"Node type: {self.node_label}")
    
    def verify_and_show_top_players(self):
        # verify import + show current top players
        print("\n" + "=" * 80)
        print(f"GLICKO-2 RATINGS - Top 20 Players ({self.node_label})")
        print("=" * 80)
        
        with self.driver.session() as session:
            # get most recent date
            result = session.run(f"""
                MATCH (g:{self.node_label})
                RETURN max(g.date) as latest_date""")
            latest_date = result.single()['latest_date']
            
            if not latest_date:
                print(f"No {self.node_label} data found!")
                return
            
            # calc cutoff for active players
            from datetime import datetime, timedelta
            latest = datetime.strptime(str(latest_date), '%Y%m%d')
            cutoff = latest - timedelta(days=365)
            cutoff_date = int(cutoff.strftime('%Y%m%d'))
            
            print(f"Latest ratings date: {latest_date}")
            print(f"Configuration: TAU={self.TAU}, Period={self.RATING_PERIOD_DAYS} days")
            print(f"Showing players active since: {cutoff_date}")
            print("\n{:<4} {:<20} {:<8} {:<6} {:<8} {:<8} {:<8} {:<8}".format("Rank", "Player", "Rating", "±RD", "Hard", "Clay", "Grass", "Vol"))
            print("-" * 86)
            
            # get top players
            result = session.run(f"""
                MATCH (p:Player)-[:{self.relationship_type}]->(g:{self.node_label})
                WHERE g.date = $date
                WITH p, g
                MATCH (p)-[:WON|LOST]->(m:Match)-[:PLAYED_IN]->(t:Tournament)
                WHERE t.date >= $cutoff
                WITH p, g, max(t.date) as last_match
                RETURN DISTINCT p.first_name + ' ' + p.last_name as name,
                       g.rating_overall as rating,
                       g.rd_overall as rd,
                       g.rating_hard as hard,
                       g.rating_clay as clay,
                       g.rating_grass as grass,
                       g.volatility_overall as volatility,
                       g.total_matches as matches
                ORDER BY g.rating_overall DESC
                LIMIT 20
            """, date=latest_date, cutoff=cutoff_date)
            
            for i, record in enumerate(result, 1):
                # format conf int (CI)
                rating = record['rating']
                rd = record['rd']
                conf_interval = f"±{rd:.0f}"
                
                print(f"{i:<4} {record['name']:<20} {rating:<8.1f} {conf_interval:<6} "
                      f"{record['hard']:<8.1f} {record['clay']:<8.1f} "
                      f"{record['grass']:<8.1f} {record['volatility']:<8.4f}")
            
            print("\nNote: Rating ±2*RD gives 95% confidence interval")
            print("Volatility measures consistency (lower = more consistent)")

def main():
    parser = argparse.ArgumentParser(description='Import Glicko-2 ratings for tennis')
    parser.add_argument('--tau', type=float, default=0.4, help='System constant (0.3=stable, 0.5=default, 1.2=volatile)')
    parser.add_argument('--period', type=int, default=14, help='Rating period in days (7, 14, or 30)')
    parser.add_argument('--rd-decay', type=float, default=30, help='RD growth per period when inactive')
    parser.add_argument('--surface-rd', type=float, default=200, help='Initial RD when playing new surface')
    parser.add_argument('--suffix', type=str, default='', help='Node label suffix (e.g., "_stable" or "_volatile")')
    
    args = parser.parse_args()
    
    # validate params
    if args.tau < 0.3 or args.tau > 1.2: print("Warning: TAU should be between 0.3 and 1.2")
    if args.period not in [7, 14, 30]: print("Warning: Period typically 7, 14, or 30 days")
    
    importer = Glicko2Importer(
        tau=args.tau,
        rating_period_days=args.period,
        rd_decay=args.rd_decay,
        node_suffix=args.suffix,
        surface_rd_init=args.surface_rd)
    
    try:
        # import ratings
        importer.import_glicko2_ratings()
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