# script to create weekly player statistics nodes in Neo4j

# imports 
import pandas as pd
from neo4j import GraphDatabase
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import re
from tqdm import tqdm
import gc  #for garbage collection

# Neo4j connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "put_your_password_here"

class WeeklyStatsImporter:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.player_rankings_cache = {}
        
    def close(self):
        self.driver.close()
    
    def get_monday_of_week(self, date):
        # convert date to Monday of that week
        if isinstance(date, int): date = str(date)
        dt = datetime.strptime(date, '%Y%m%d')
        days_since_monday = dt.weekday()
        monday = dt - timedelta(days=days_since_monday)
        return int(monday.strftime('%Y%m%d'))
    
    def parse_score(self, score):
        # enchanced score parsing to find straight sets + close matches
        if not score or pd.isna(score): return None
        sets = score.strip().split()
        sets_won = 0
        sets_lost = 0
        games_won = 0
        games_lost = 0
        tiebreaks_won = 0
        tiebreaks_lost = 0
        
        for set_score in sets:
            if 'RET' in set_score or 'W/O' in set_score or 'DEF' in set_score:
                continue
                
            # handle tiebreak notation like 7-6(5)
            tiebreak_match = re.match(r'(\d+)-(\d+)(?:\((\d+)\))?', set_score)
            if tiebreak_match:
                winner_games = int(tiebreak_match.group(1))
                loser_games = int(tiebreak_match.group(2))
                
                games_won += winner_games
                games_lost += loser_games
                
                if winner_games > loser_games:
                    sets_won += 1
                    if winner_games == 7 and loser_games == 6:
                        tiebreaks_won += 1
                else:
                    sets_lost += 1
                    if loser_games == 7 and winner_games == 6:
                        tiebreaks_lost += 1
        
        #determine if straight sets
        is_straight_sets = False
        if sets_won == 2 and sets_lost == 0:  # Best of 3
            is_straight_sets = True
        elif sets_won == 3 and sets_lost == 0:  # Best of 5
            is_straight_sets = True
        
        # determine if deciding set (3 in bo3 or 5 in bo5)
        total_sets = sets_won + sets_lost
        is_deciding_set = (total_sets == 3 and max(sets_won, sets_lost) == 2) or \
                         (total_sets == 5 and max(sets_won, sets_lost) == 3)
        
        # close match (games difference <= 3)
        is_close_match = abs(games_won - games_lost) <= 3
            
        return {
            'sets_won': sets_won,
            'sets_lost': sets_lost,
            'games_won': games_won,
            'games_lost': games_lost,
            'tiebreaks_won': tiebreaks_won,
            'tiebreaks_lost': tiebreaks_lost,
            'is_straight_sets': is_straight_sets,
            'is_deciding_set': is_deciding_set,
            'is_close_match': is_close_match
        }
    
    def calculate_match_stats(self, match_data, player_id, is_winner):
        # calc comprehensive stats for a single match
        stats = {
            'date': match_data['tournament_date'],
            'surface': match_data['surface'],
            'player_id': player_id,
            'is_winner': is_winner
        }
        
        #parse score
        score_info = self.parse_score(match_data['score'])
        if score_info: stats.update(score_info)
        
        # get serve/return stats based on winner/loser
        if is_winner:
            prefix = 'w_'
            opp_prefix = 'l_'
        else:
            prefix = 'l_'
            opp_prefix = 'w_'
        
        # basic stats
        stats['ace'] = match_data.get(f'{prefix}ace', 0) or 0
        stats['df'] = match_data.get(f'{prefix}df', 0) or 0
        stats['svpt'] = match_data.get(f'{prefix}svpt', 0) or 0
        stats['1stIn'] = match_data.get(f'{prefix}1stIn', 0) or 0
        stats['1stWon'] = match_data.get(f'{prefix}1stWon', 0) or 0
        stats['2ndWon'] = match_data.get(f'{prefix}2ndWon', 0) or 0
        stats['SvGms'] = match_data.get(f'{prefix}SvGms', 0) or 0
        stats['bpSaved'] = match_data.get(f'{prefix}bpSaved', 0) or 0
        stats['bpFaced'] = match_data.get(f'{prefix}bpFaced', 0) or 0
        
        # opponent's stats (for return calculations!!)
        stats['opp_svpt'] = match_data.get(f'{opp_prefix}svpt', 0) or 0
        stats['opp_1stIn'] = match_data.get(f'{opp_prefix}1stIn', 0) or 0
        stats['opp_1stWon'] = match_data.get(f'{opp_prefix}1stWon', 0) or 0
        stats['opp_2ndWon'] = match_data.get(f'{opp_prefix}2ndWon', 0) or 0
        stats['opp_SvGms'] = match_data.get(f'{opp_prefix}SvGms', 0) or 0
        stats['opp_bpSaved'] = match_data.get(f'{opp_prefix}bpSaved', 0) or 0
        stats['opp_bpFaced'] = match_data.get(f'{opp_prefix}bpFaced', 0) or 0
        
        # minutes
        stats['minutes'] = match_data.get('minutes', 0) or 0
        
        # get op ranking for upset calcs
        stats['opponent_id'] = match_data['loser_id'] if is_winner else match_data['winner_id']
        
        return stats
    
    def calculate_window_stats(self, matches, window):
        # calc comprehensive stats for a time window
        if not matches: return {f'stats_{window}_matches_played': 0}
        
        stats = {
            f'stats_{window}_matches_played': len(matches),
            f'stats_{window}_wins': sum(1 for m in matches if m['is_winner']),
            f'stats_{window}_losses': sum(1 for m in matches if not m['is_winner'])
        }
        
        # win %
        stats[f'stats_{window}_win_pct'] = stats[f'stats_{window}_wins'] / len(matches) if matches else -1
        
        # totals aggregated
        total_ace = sum(m.get('ace', 0) for m in matches)
        total_df = sum(m.get('df', 0) for m in matches)
        total_svpt = sum(m.get('svpt', 0) for m in matches)
        total_1stIn = sum(m.get('1stIn', 0) for m in matches)
        total_1stWon = sum(m.get('1stWon', 0) for m in matches)
        total_2ndWon = sum(m.get('2ndWon', 0) for m in matches)
        total_SvGms = sum(m.get('SvGms', 0) for m in matches)
        total_bpSaved = sum(m.get('bpSaved', 0) for m in matches)
        total_bpFaced = sum(m.get('bpFaced', 0) for m in matches)
        
        # op totals (for return stats)
        total_opp_svpt = sum(m.get('opp_svpt', 0) for m in matches)
        total_opp_1stIn = sum(m.get('opp_1stIn', 0) for m in matches)
        total_opp_1stWon = sum(m.get('opp_1stWon', 0) for m in matches)
        total_opp_2ndWon = sum(m.get('opp_2ndWon', 0) for m in matches)
        total_opp_SvGms = sum(m.get('opp_SvGms', 0) for m in matches)
        total_opp_bpSaved = sum(m.get('opp_bpSaved', 0) for m in matches)
        total_opp_bpFaced = sum(m.get('opp_bpFaced', 0) for m in matches)
        
        # serving stats
        stats[f'stats_{window}_ace_pct'] = total_ace / total_svpt if total_svpt > 0 else -1
        stats[f'stats_{window}_df_pct'] = total_df / total_svpt if total_svpt > 0 else -1
        stats[f'stats_{window}_first_serve_pct'] = total_1stIn / total_svpt if total_svpt > 0 else -1
        
        # first serve won % (first serve goes in)
        stats[f'stats_{window}_first_serve_won_pct'] = total_1stWon / total_1stIn if total_1stIn > 0 else -1
        
        # second serve won %
        second_serves = total_svpt - total_1stIn
        stats[f'stats_{window}_second_serve_won_pct'] = total_2ndWon / second_serves if second_serves > 0 else -1
        
        # overall serve points won
        stats[f'stats_{window}_serve_points_won_pct'] = (total_1stWon + total_2ndWon) / total_svpt if total_svpt > 0 else -1
        
        # service games
        games_broken = total_bpFaced - total_bpSaved
        games_held = total_SvGms - games_broken if total_SvGms > 0 else 0
        stats[f'stats_{window}_service_games_held_pct'] = games_held / total_SvGms if total_SvGms > 0 else -1
        stats[f'stats_{window}_bp_saved_pct'] = total_bpSaved / total_bpFaced if total_bpFaced > 0 else -1
        stats[f'stats_{window}_bp_faced_per_game'] = total_bpFaced / total_SvGms if total_SvGms > 0 else -1
        
        # raw break point numbers (serve)
        stats[f'stats_{window}_bp_faced_total'] = total_bpFaced
        stats[f'stats_{window}_bp_saved_total'] = total_bpSaved
        
        # return stats
        return_points_won = total_opp_svpt - (total_opp_1stWon + total_opp_2ndWon)
        stats[f'stats_{window}_return_points_won_pct'] = return_points_won / total_opp_svpt if total_opp_svpt > 0 else -1
        
        # first return won %
        first_returns_won = total_opp_1stIn - total_opp_1stWon
        stats[f'stats_{window}_first_return_won_pct'] = first_returns_won / total_opp_1stIn if total_opp_1stIn > 0 else -1
        
        # second return won %
        opp_second_serves = total_opp_svpt - total_opp_1stIn
        second_returns_won = opp_second_serves - total_opp_2ndWon if opp_second_serves > 0 else 0
        stats[f'stats_{window}_second_return_won_pct'] = second_returns_won / opp_second_serves if opp_second_serves > 0 else -1
        
        # return games
        return_games_broken = total_opp_bpFaced - total_opp_bpSaved
        stats[f'stats_{window}_return_games_broken_pct'] = return_games_broken / total_opp_SvGms if total_opp_SvGms > 0 else -1
        stats[f'stats_{window}_bp_converted_pct'] = return_games_broken / total_opp_bpFaced if total_opp_bpFaced > 0 else -1
        stats[f'stats_{window}_bp_created_per_return_game'] = total_opp_bpFaced / total_opp_SvGms if total_opp_SvGms > 0 else -1
        
        # raw break point numbers (return)
        stats[f'stats_{window}_bp_created_total'] = total_opp_bpFaced  # BPs created = opponent's BPs faced
        stats[f'stats_{window}_bp_converted_total'] = return_games_broken  # BPs converted = games broken
        
        # total points won %
        total_points_played = total_svpt + total_opp_svpt
        total_points_won = (total_1stWon + total_2ndWon) + return_points_won
        stats[f'stats_{window}_total_points_won_pct'] = total_points_won / total_points_played if total_points_played > 0 else -1
        
        # efficiency stats
        # service game efficiency (points per service game: lower= better)
        stats[f'stats_{window}_service_game_efficiency'] = total_svpt / total_SvGms if total_SvGms > 0 else -1
        
        # return game impact (opponent's points per return game: lower = better)
        stats[f'stats_{window}_return_game_impact'] = total_opp_svpt / total_opp_SvGms if total_opp_SvGms > 0 else -1
        
        # efficiency ratio (opponent efficiency / your efficiency: higher = better)
        if total_SvGms > 0 and total_opp_SvGms > 0:
            your_efficiency = total_svpt / total_SvGms
            opp_efficiency = total_opp_svpt / total_opp_SvGms
            stats[f'stats_{window}_efficiency_ratio'] = opp_efficiency / your_efficiency if your_efficiency > 0 else -1
        else:
            stats[f'stats_{window}_efficiency_ratio'] = -1
        
        # dominance metrics
        total_games_won = sum(m.get('games_won', 0) for m in matches)
        total_games_lost = sum(m.get('games_lost', 0) for m in matches)
        stats[f'stats_{window}_games_ratio'] = total_games_won / total_games_lost if total_games_lost > 0 else -1
        
        total_sets_won = sum(m.get('sets_won', 0) for m in matches)
        total_sets_lost = sum(m.get('sets_lost', 0) for m in matches)
        stats[f'stats_{window}_sets_ratio'] = total_sets_won / total_sets_lost if total_sets_lost > 0 else -1
        
        # straight sets percentage (when winning)
        wins = [m for m in matches if m['is_winner']]
        straight_set_wins = sum(1 for m in wins if m.get('is_straight_sets', False))
        stats[f'stats_{window}_straight_sets_pct'] = straight_set_wins / len(wins) if wins else -1
        
        # tiebreak stats
        total_tb_won = sum(m.get('tiebreaks_won', 0) for m in matches)
        total_tb_lost = sum(m.get('tiebreaks_lost', 0) for m in matches)
        total_tb = total_tb_won + total_tb_lost
        stats[f'stats_{window}_tiebreak_pct'] = total_tb_won / total_tb if total_tb > 0 else -1
        
        # pressure situatations (12w and 52w only)
        if window in ['12w', '52w']:
            # deciding set %
            deciding_set_matches = [m for m in matches if m.get('is_deciding_set', False)]
            deciding_set_wins = sum(1 for m in deciding_set_matches if m['is_winner'])
            stats[f'stats_{window}_deciding_set_pct'] = deciding_set_wins / len(deciding_set_matches) if deciding_set_matches else -1
            
            # close match %
            close_matches = [m for m in matches if m.get('is_close_match', False)]
            close_wins = sum(1 for m in close_matches if m['is_winner'])
            stats[f'stats_{window}_close_match_pct'] = close_wins / len(close_matches) if close_matches else -1
            
            # upset stats (need opponent rankings)
            self.calculate_upset_stats(matches, window, stats)
        
        # ninutes (for fatigue tracking, 2w 4w 8w only)
        if window in ['2w', '4w', '8w']:
            stats[f'stats_{window}_minutes_total'] = sum(m.get('minutes', 0) for m in matches)
            stats[f'stats_{window}_minutes_avg'] = stats[f'stats_{window}_minutes_total'] / len(matches) if matches else -1
            stats[f'stats_{window}_matches_per_week'] = len(matches) / (int(window[:-1]) if window != '8w' else 8)
        
        return stats
    
    def calculate_upset_stats(self, matches, window, stats):
        # calc upset and defend rates
        upsets = []
        defends = []
        
        for match in matches:
            # get player and opponent rankings at match time
            player_id = match['player_id']
            opponent_id = match.get('opponent_id')
            match_date = match['date']
            
            player_ranking = self.get_ranking_at_date(player_id, match_date)
            opponent_ranking = self.get_ranking_at_date(opponent_id, match_date)
            
            if player_ranking and opponent_ranking:
                player_points = player_ranking.get('points', 0)
                opponent_points = opponent_ranking.get('points', 0)
                
                if player_points > 0 and opponent_points > 0:
                    if match['is_winner']:
                        if opponent_points > player_points:
                            # upset
                            magnitude = opponent_points / player_points - 1
                            upsets.append(magnitude)
                        elif player_points > opponent_points:
                            # defended
                            defends.append(1)
                    else:
                        if player_points > opponent_points:
                            # failed to defend
                            defends.append(0)
        
        #calc rates
        total_underdog = len(upsets) + sum(1 for m in matches if not m['is_winner'])
        stats[f'stats_{window}_upset_rate'] = len(upsets) / total_underdog if total_underdog > 0 else -1
        stats[f'stats_{window}_upset_avg_magnitude'] = np.mean(upsets) if upsets else -1
        stats[f'stats_{window}_defend_rate'] = np.mean(defends) if defends else -1
    
    def calculate_surface_stats(self, matches_52w, surface):
        # calc surface-specific stats
        surface_matches = [m for m in matches_52w if m.get('surface') == surface]
        
        if not surface_matches: return {}
        stats = {
            f'stats_52w_{surface}_matches': len(surface_matches),
            f'stats_52w_{surface}_win_pct': sum(1 for m in surface_matches if m['is_winner']) / len(surface_matches)}
        
        # surface-specific serve
        total_svpt = sum(m.get('svpt', 0) for m in surface_matches)
        total_serve_won = sum(m.get('1stWon', 0) + m.get('2ndWon', 0) for m in surface_matches)
        stats[f'stats_52w_{surface}_serve_pct'] = total_serve_won / total_svpt if total_svpt > 0 else -1
        
        # surface-specific return
        total_opp_svpt = sum(m.get('opp_svpt', 0) for m in surface_matches)
        total_return_won = sum(m.get('opp_svpt', 0) - (m.get('opp_1stWon', 0) + m.get('opp_2ndWon', 0)) for m in surface_matches)
        stats[f'stats_52w_{surface}_return_pct'] = total_return_won / total_opp_svpt if total_opp_svpt > 0 else -1
        
        return stats
    
    def calculate_streaks(self, matches_52w):
        # calc current streaks
        if not matches_52w: return {'streak_matches_won': 0, 'streak_matches_lost': 0, 'streak_tiebreaks': 0}
        
        # sort by date (most recent last)
        sorted_matches = sorted(matches_52w, key=lambda x: x['date'])
        
        # match streak
        match_streak = 0
        for match in reversed(sorted_matches):
            if match['is_winner']:
                if match_streak >= 0: match_streak += 1
                else: break
            else:
                if match_streak <= 0: match_streak -= 1
                else: break
        
        # tiebreak streak
        tb_streak = 0
        for match in reversed(sorted_matches):
            tb_won = match.get('tiebreaks_won', 0)
            tb_lost = match.get('tiebreaks_lost', 0)
            
            if tb_won > 0 or tb_lost > 0:
                if tb_won > tb_lost:
                    if tb_streak >= 0: tb_streak += 1
                    else: break
                else:
                    if tb_streak <= 0: tb_streak -= 1
                    else: break
        
        return {
            'streak_matches_won': max(0, match_streak),
            'streak_matches_lost': abs(min(0, match_streak)),
            'streak_tiebreaks': tb_streak}
    
    def get_ranking_at_date(self, player_id, date):
        # get player ranking at specific date(CACHED!!)
        cache_key = f"{player_id}_{date}"
        
        if cache_key in self.player_rankings_cache: return self.player_rankings_cache[cache_key]
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Player {id: $player_id})-[:HAS_RANKING]->(r:Ranking)
                WHERE r.date <= $date
                RETURN r.rank as rank, r.points as points
                ORDER BY r.date DESC
                LIMIT 1
            """, player_id=player_id, date=date)
            
            record = result.single()
            ranking = {'rank': record['rank'], 'points': record['points']} if record else None
            self.player_rankings_cache[cache_key] = ranking
            
            return ranking
    
    def get_player_batch_ids(self, batch_size=100):
        # get player IDs in bathces (avoids loading everything at once)
        with self.driver.session() as session:
            #get players who have actually played matches
            result = session.run("""
                MATCH (p:Player)-[:WON|LOST]->()
                WITH DISTINCT p.id as player_id
                RETURN collect(player_id) as player_ids
            """)
            
            all_player_ids = result.single()['player_ids']
            print(f"Found {len(all_player_ids)} players with matches")
            
            # Yield batches
            for i in range(0, len(all_player_ids), batch_size):
                yield all_player_ids[i:i+batch_size]
    
    def load_matches_for_players(self, player_ids):
        # loads matches for specific batch of players
        player_matches = defaultdict(list)
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Match)<-[:WON]-(winner:Player)
                WHERE winner.id IN $player_ids
                MATCH (m)<-[:LOST]-(loser:Player)
                MATCH (m)-[:PLAYED_IN]->(t:Tournament)
                RETURN 
                    m.id as match_id,
                    winner.id as winner_id,
                    loser.id as loser_id,
                    t.date as tournament_date,
                    t.surface as surface,
                    m.score as score,
                    m.best_of as best_of,
                    m.minutes as minutes,
                    m.w_ace as w_ace, m.w_df as w_df,
                    m.w_svpt as w_svpt, m.w_1stIn as w_1stIn,
                    m.w_1stWon as w_1stWon, m.w_2ndWon as w_2ndWon,
                    m.w_SvGms as w_SvGms, m.w_bpSaved as w_bpSaved,
                    m.w_bpFaced as w_bpFaced,
                    m.l_ace as l_ace, m.l_df as l_df,
                    m.l_svpt as l_svpt, m.l_1stIn as l_1stIn,
                    m.l_1stWon as l_1stWon, m.l_2ndWon as l_2ndWon,
                    m.l_SvGms as l_SvGms, m.l_bpSaved as l_bpSaved,
                    m.l_bpFaced as l_bpFaced
                    
                UNION
                
                MATCH (m:Match)<-[:LOST]-(loser:Player)
                WHERE loser.id IN $player_ids
                MATCH (m)<-[:WON]-(winner:Player)
                MATCH (m)-[:PLAYED_IN]->(t:Tournament)
                RETURN 
                    m.id as match_id,
                    winner.id as winner_id,
                    loser.id as loser_id,
                    t.date as tournament_date,
                    t.surface as surface,
                    m.score as score,
                    m.best_of as best_of,
                    m.minutes as minutes,
                    m.w_ace as w_ace, m.w_df as w_df,
                    m.w_svpt as w_svpt, m.w_1stIn as w_1stIn,
                    m.w_1stWon as w_1stWon, m.w_2ndWon as w_2ndWon,
                    m.w_SvGms as w_SvGms, m.w_bpSaved as w_bpSaved,
                    m.w_bpFaced as w_bpFaced,
                    m.l_ace as l_ace, m.l_df as l_df,
                    m.l_svpt as l_svpt, m.l_1stIn as l_1stIn,
                    m.l_1stWon as l_1stWon, m.l_2ndWon as l_2ndWon,
                    m.l_SvGms as l_SvGms, m.l_bpSaved as l_bpSaved,
                    m.l_bpFaced as l_bpFaced
            """, player_ids=player_ids)
            
            for record in result:
                match_data = dict(record)
                
                #process for winner
                if match_data['winner_id'] in player_ids:
                    winner_stats = self.calculate_match_stats(match_data, match_data['winner_id'], is_winner=True)
                    player_matches[match_data['winner_id']].append(winner_stats)
                
                #process for loser
                if match_data['loser_id'] in player_ids:
                    loser_stats = self.calculate_match_stats(match_data, match_data['loser_id'], is_winner=False)
                    player_matches[match_data['loser_id']].append(loser_stats)
        
        return player_matches
    
    def create_weekly_stats_for_player(self, player_id, player_matches):
        # create ALL weekly stats nodes for a player
        if not player_matches: return []
        
        # sort matches by date
        sorted_matches = sorted(player_matches, key=lambda x: x['date'])
        
        # find all Mondays from first match to end of 2024 (end of data)
        first_date = sorted_matches[0]['date']
        last_date = 20241231
        
        current_monday = self.get_monday_of_week(str(first_date))
        end_monday = self.get_monday_of_week(str(last_date))
        
        weekly_nodes = []
        
        while current_monday <= end_monday:
            # get matches in diff windows
            current_dt = datetime.strptime(str(current_monday), '%Y%m%d')
            
            date_2w_ago = int((current_dt - timedelta(weeks=2)).strftime('%Y%m%d'))
            date_4w_ago = int((current_dt - timedelta(weeks=4)).strftime('%Y%m%d'))
            date_8w_ago = int((current_dt - timedelta(weeks=8)).strftime('%Y%m%d'))
            date_12w_ago = int((current_dt - timedelta(weeks=12)).strftime('%Y%m%d'))
            date_52w_ago = int((current_dt - timedelta(weeks=52)).strftime('%Y%m%d'))
            
            # filter matches for each window
            matches_2w = [m for m in sorted_matches if date_2w_ago < m['date'] <= current_monday]
            matches_4w = [m for m in sorted_matches if date_4w_ago < m['date'] <= current_monday]
            matches_8w = [m for m in sorted_matches if date_8w_ago < m['date'] <= current_monday]
            matches_12w = [m for m in sorted_matches if date_12w_ago < m['date'] <= current_monday]
            matches_52w = [m for m in sorted_matches if date_52w_ago < m['date'] <= current_monday]
            
            # skip if no matches in 52w (year)
            if not matches_52w:
                # move to next Monday
                current_dt = current_dt + timedelta(weeks=1)
                current_monday = int(current_dt.strftime('%Y%m%d'))
                continue
            
            # create node data
            node_data = {
                'player_id': player_id,
                'date': current_monday}
            
            # + window stats
            node_data.update(self.calculate_window_stats(matches_2w, '2w'))
            node_data.update(self.calculate_window_stats(matches_4w, '4w'))
            node_data.update(self.calculate_window_stats(matches_8w, '8w'))
            node_data.update(self.calculate_window_stats(matches_12w, '12w'))
            node_data.update(self.calculate_window_stats(matches_52w, '52w'))
            
            # + surface-specific stats (52w only)
            for surface in ['Hard', 'Clay', 'Grass']:
                node_data.update(self.calculate_surface_stats(matches_52w, surface))
            
            # + streak stats
            node_data.update(self.calculate_streaks(matches_52w))
            
            weekly_nodes.append(node_data)
            
            # move to next Monday
            current_dt = current_dt + timedelta(weeks=1)
            current_monday = int(current_dt.strftime('%Y%m%d'))
        
        return weekly_nodes
    
    def import_stats_nodes_batch(self, nodes_batch):
        #import batch of stats nodes to neo4j
        with self.driver.session() as session:
            # create nodes & relationships in one query (more efficient)
            session.run("""
                UNWIND $nodes AS node
                CREATE (s:PlayerStats)
                SET s = node
                WITH s
                MATCH (p:Player {id: s.player_id})
                CREATE (p)-[:HAS_STATS]->(s)
            """, nodes=nodes_batch)
    
    def run_import(self, batch_size=100):
        #main import func; UPDATED w/ better memory management
        print("=" * 80)
        print("ENHANCED WEEKLY PLAYER STATS IMPORT (v3)")
        print("=" * 80)
        print(f"Processing players in batches of {batch_size}")
        
        # clear any existing PlayerStats nodes first (safety measure for re-runs)
        print("\nClearing existing PlayerStats nodes...")
        with self.driver.session() as session:
            result = session.run("MATCH (ps:PlayerStats) RETURN COUNT(ps) as count")
            existing_count = result.single()['count']
            if existing_count > 0:
                print(f"Found {existing_count:,} existing nodes, removing...")
                session.run("MATCH (ps:PlayerStats) DETACH DELETE ps")
            else:
                print("No existing PlayerStats nodes found")
        
        total_nodes_created = 0
        batch_num = 0
        
        # process players in BATCHES
        for player_batch_ids in self.get_player_batch_ids(batch_size):
            batch_num += 1
            print(f"\n--- Processing batch {batch_num} ({len(player_batch_ids)} players) ---")
            
            # load matches for batch of players
            player_matches = self.load_matches_for_players(player_batch_ids)
            
            # create weekly stats for each player
            batch_nodes = []
            for player_id, matches in player_matches.items():
                weekly_nodes = self.create_weekly_stats_for_player(player_id, matches)
                batch_nodes.extend(weekly_nodes)
            
            # import nodes in smaller sub-batches
            if batch_nodes:
                print(f"  Creating {len(batch_nodes)} PlayerStats nodes...")
                for i in range(0, len(batch_nodes), 500):
                    sub_batch = batch_nodes[i:i+500]
                    self.import_stats_nodes_batch(sub_batch)
                    total_nodes_created += len(sub_batch)
            
            # CLEAR MEMORY after each batch
            del player_matches
            del batch_nodes
            gc.collect()
            
            print(f"  Total nodes created so far: {total_nodes_created:,}")
        
        print(f"\n{'='*80}")
        print(f"IMPORT COMPLETE!")
        print(f"Total PlayerStats nodes created: {total_nodes_created:,}")
        
        # verify import
        with self.driver.session() as session:
            result = session.run("""
                MATCH (ps:PlayerStats)
                RETURN COUNT(ps) as total,
                       MIN(ps.date) as earliest_date,
                       MAX(ps.date) as latest_date
            """)
            record = result.single()
            print(f"\nVerification:")
            print(f"  Total nodes in database: {record['total']:,}")
            print(f"  Date range: {record['earliest_date']} to {record['latest_date']}")

if __name__ == "__main__":
    importer = WeeklyStatsImporter()
    try:
        # use batch_size=100 for safer memory usage
        importer.run_import(batch_size=100)
    finally:
        importer.close()