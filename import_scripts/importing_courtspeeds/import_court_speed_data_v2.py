# script to import court speed and ace% data to tournament nodes in neo4j
# reads from court_speeds_YYYY.csv files and adds properties to matching tournaments
# UPDATED: handling of tournaments which may start Dec one year, Jan the next!

import os
import pandas as pd
from neo4j import GraphDatabase
import glob
from difflib import SequenceMatcher
from tqdm import tqdm

# Neo4j connection parameters
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "put_your_password_here"

# path to court speed data --> ADJUST FOR DATA
COURT_SPEED_DIR = "../../../court_speed"

class CourtSpeedImporter:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.tournament_cache = {}
        self.stats = {
            'total_processed': 0,
            'matched': 0,
            'failed': 0,
            'failed_tournaments': []
        }
    
    def close(self):
        self.driver.close()
    
    def load_tournaments_for_year(self, year):
        # load all tournaments for a given year into cache
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Tournament)
                WHERE t.date >= $start_date AND t.date < $end_date
                RETURN t.id as id, t.name as name, t.date as date
            """, start_date=int(f"{year}0101"), end_date=int(f"{year+1}0101"))
            
            tournaments = {}
            for record in result:
                tournaments[record['name'].lower()] = {
                    'id': record['id'],
                    'name': record['name'],
                    'date': record['date']
                }
            return tournaments
    
    def load_tournaments_for_december(self, year):
        # load tournaments from Dec of given year
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Tournament)
                WHERE t.date >= $start_date AND t.date < $end_date
                RETURN t.id as id, t.name as name, t.date as date
            """, start_date=int(f"{year}1201"), end_date=int(f"{year+1}0101"))
            
            tournaments = {}
            for record in result:
                # only include if AFTER DEC 25 --> likely year spanning and issue tournament
                date_str = str(record['date'])
                day = int(date_str[6:8]) if len(date_str) >= 8 else 0
                if day >= 25:
                    tournaments[record['name'].lower()] = {
                        'id': record['id'],
                        'name': record['name'],
                        'date': record['date']
                    }
            return tournaments
    
    def fuzzy_match_tournament(self, tournament_name, year):
        # find best matching tournament using fuzzy matching
        tournament_lower = tournament_name.lower().strip()
        
        # . ... these tournaments often start in late Dec of the previous year, so probably want to check all of these
        early_year_tournaments = ['doha', 'adelaide', 'brisbane', 'chennai', 'auckland', 'pune']
        
        # ...check if this is an early-year tournament
        is_early_year = any(early_tourn in tournament_lower for early_tourn in early_year_tournaments)
        
        if is_early_year:
            # IMPORTANT: For year X, I want:
            # - the tournament that starts in late Dec of year X-1 (for year X's edition)
            # - NOT the tournament in jan of year X (that was year X's edition)
            # - NOT the tournament in Dec of year X (that's year X+1's edition)
            
            # FIRST, try Dec of PREVIOUS year (this is likely the correct match?)
            prev_year = year - 1
            dec_key = f"dec_{prev_year}"
            if dec_key not in self.tournament_cache:
                self.tournament_cache[dec_key] = self.load_tournaments_for_december(prev_year)
                print(f"    Loaded {len(self.tournament_cache[dec_key])} December {prev_year} tournaments")
            
            # check dec of previous year for exact match
            if tournament_lower in self.tournament_cache[dec_key]:
                matched_data = self.tournament_cache[dec_key][tournament_lower]
                print(f"    [V] Cross-year match: {tournament_name} {year} → {matched_data['name']} (Dec {prev_year})")
                return matched_data['id']
            
            # try fuzzy matching in Dec of previous year
            for cached_name, cached_data in self.tournament_cache[dec_key].items():
                # check for partial match
                if tournament_lower in cached_name or cached_name in tournament_lower:
                    print(f"    [V] Cross-year fuzzy match: {tournament_name} {year} → {cached_data['name']} (Dec {prev_year})")
                    return cached_data['id']
                
                # check common variations.....
                if ('brisbane' in tournament_lower and 'brisbane' in cached_name) or \
                   ('adelaide' in tournament_lower and 'adelaide' in cached_name) or \
                   ('chennai' in tournament_lower and 'chennai' in cached_name) or \
                   ('doha' in tournament_lower and 'doha' in cached_name) or \
                   ('auckland' in tournament_lower and 'auckland' in cached_name):
                    print(f"    [V] Cross-year fuzzy match: {tournament_name} {year} → {cached_data['name']} (Dec {prev_year})")
                    return cached_data['id']
            
            # if NOT in dec of prev year, try Jan of current year?
            # (in case it didn't span years this particular year)
            # load only Jan tournaments for current year
            jan_key = f"jan_{year}"
            if jan_key not in self.tournament_cache:
                with self.driver.session() as session:
                    result = session.run("""
                        MATCH (t:Tournament)
                        WHERE t.date >= $start_date AND t.date < $end_date
                        RETURN t.id as id, t.name as name, t.date as date
                    """, start_date=int(f"{year}0101"), end_date=int(f"{year}0120"))
                    
                    jan_tournaments = {}
                    for record in result:
                        jan_tournaments[record['name'].lower()] = {
                            'id': record['id'],
                            'name': record['name'],
                            'date': record['date']
                        }
                    self.tournament_cache[jan_key] = jan_tournaments
                    if jan_tournaments:
                        print(f"    Loaded {len(jan_tournaments)} January {year} tournaments")
            
            # check jan of current year
            if jan_key in self.tournament_cache:
                if tournament_lower in self.tournament_cache[jan_key]:
                    matched_data = self.tournament_cache[jan_key][tournament_lower]
                    print(f"    [V] January match: {tournament_name} {year} → {matched_data['name']} (Jan {year})")
                    return matched_data['id']
                
                # try fuzzy matching in jan
                for cached_name, cached_data in self.tournament_cache[jan_key].items():
                    if tournament_lower in cached_name or cached_name in tournament_lower:
                        print(f"    [V] January fuzzy match: {tournament_name} {year} → {cached_data['name']} (Jan {year})")
                        return cached_data['id']
        
        else:
            # for non-early-year tournaments, use standard matching in current year
            # FIRST make sure year is loaded
            if year not in self.tournament_cache:
                self.tournament_cache[year] = self.load_tournaments_for_year(year)
            
            # TRY exact match in the given year
            if tournament_lower in self.tournament_cache[year]:
                return self.tournament_cache[year][tournament_lower]['id']
        
        # common name variations to handle (for all tournaments)
        name_mappings = {
            'indian wells': 'indian wells masters',
            'miami': 'miami masters',
            'monte carlo': 'monte carlo masters',
            'rome': 'rome masters',
            'madrid': 'madrid masters',
            'canada': 'canada masters',
            'cincinnati': 'cincinnati masters',
            'shanghai': 'shanghai masters',
            'paris': 'paris masters',
            'australian open': 'australian open',
            'roland garros': 'roland garros',
            'french open': 'roland garros',
            'wimbledon': 'wimbledon',
            'us open': 'us open',
            'atp finals': 'atp finals',
            'tour finals': 'tour finals',
            'masters cup': 'tour finals',
            'olympics': 'olympics',
            'olympic games': 'olympics'
        }
        
        # check if tournament name matches any mappings
        for key, value in name_mappings.items():
            if key in tournament_lower:
                mapped_name = value
                if year in self.tournament_cache and mapped_name in self.tournament_cache[year]:
                    return self.tournament_cache[year][mapped_name]['id']
        
        # if no exact or mapped match, use fuzzy matching (current year)
        if year in self.tournament_cache:
            best_match = None
            best_score = 0
            
            for cached_name, cached_data in self.tournament_cache[year].items():
                # skip dec tournaments for early-year tournament names
                # (like: we don't want Brisbane 2002 to match December 2002 Brisbane)
                if is_early_year:
                    date_str = str(cached_data['date'])
                    month = int(date_str[4:6]) if len(date_str) >= 6 else 0
                    if month == 12:
                        continue  # skip dec tournaments for early-year names
                
                # calc similarity score
                score = SequenceMatcher(None, tournament_lower, cached_name).ratio()
                
                # Boost score for partial matches
                if tournament_lower in cached_name or cached_name in tournament_lower:
                    score += 0.2
                
                # handling for masters tournaments
                if 'masters' in tournament_lower and 'masters' in cached_name: score += 0.1
                
                if score > best_score and score > 0.8: #threshold of 0.8 for matches
                    best_score = score
                    best_match = cached_data['id']
            
            if best_match:
                return best_match
        
        return None
    
    def update_tournament_node(self, tournament_id, ace_pct, surface_speed):
        #update tournament node with ace% and surface speed
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Tournament {id: $tournament_id})
                SET t.ace_pct = $ace_pct,
                    t.surface_speed = $surface_speed
                RETURN t.name as name
            """, tournament_id=tournament_id, ace_pct=ace_pct, surface_speed=surface_speed)
            
            record = result.single()
            return record['name'] if record else None
    
    def find_tournament_name_by_id(self, tournament_id, year):
        # find tournament name from ID by cache
        # check current year cache
        if year in self.tournament_cache:
            for name, data in self.tournament_cache[year].items():
                if data['id'] == tournament_id: return data['name']
        
        # check dec cache
        dec_key = f"dec_{year-1}"
        if dec_key in self.tournament_cache:
            for name, data in self.tournament_cache[dec_key].items():
                if data['id'] == tournament_id: return data['name']
        
        return None
    
    def process_court_speed_file(self, filepath):
        # process a single court speed CSV file
        filename = os.path.basename(filepath) # get year from filename
        year = int(filename.replace('court_speeds_', '').replace('.csv', ''))
        
        print(f"\nProcessing {filename} (Year: {year})")
        
        # load tournaments for this year if not cached
        if year not in self.tournament_cache:
            self.tournament_cache[year] = self.load_tournaments_for_year(year)
            print(f"  Loaded {len(self.tournament_cache[year])} tournaments for {year}")
        
        # read file
        df = pd.read_csv(filepath)
        print(f"  Found {len(df)} tournament entries")
        
        # add Matched column if it doesn't exist
        if 'Matched' not in df.columns: df['Matched'] = 'no'
        
        # process each roW . . . 
        matched_count = 0
        failed_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing {year}"):
            tournament_name = row['Tournament']
            ace_pct = row['Ace%']
            surface_speed = row['Surface Speed']
            
            # skip if already matched (in case we're re-running)
            if 'Matched' in df.columns and df.at[idx, 'Matched'] == 'yes':
                matched_count += 1
                continue
            
            # find matching tournament
            tournament_id = self.fuzzy_match_tournament(tournament_name, year)
            
            if tournament_id:
                #get actual matched tournament name
                matched_name = self.find_tournament_name_by_id(tournament_id, year)
                
                # print mapping if names don't match exactly
                if matched_name and matched_name.lower() != tournament_name.lower():
                    print(f"    Fuzzy matched: '{tournament_name}' → '{matched_name}'")
                
                #  update the tournament node
                updated_name = self.update_tournament_node(tournament_id, ace_pct, surface_speed)
                if updated_name:
                    df.at[idx, 'Matched'] = 'yes'
                    matched_count += 1
                    self.stats['matched'] += 1
                else:
                    df.at[idx, 'Matched'] = 'no'
                    failed_count += 1
                    self.stats['failed'] += 1
                    self.stats['failed_tournaments'].append(f"{year}: {tournament_name}")
            else:
                df.at[idx, 'Matched'] = 'no'
                failed_count += 1
                self.stats['failed'] += 1
                self.stats['failed_tournaments'].append(f"{year}: {tournament_name}")
        
        self.stats['total_processed'] += len(df)
        
        # save updated CSV (has MATCHED col now)
        df.to_csv(filepath, index=False)
        
        print(f"  Results: {matched_count} matched, {failed_count} failed")
        
        # show failed tournaments for this year
        if failed_count > 0:
            print(f"  Failed to match:")
            for idx, row in df[df['Matched'] == 'no'].iterrows():
                print(f"    - {row['Tournament']}")
    
    def run_import(self):
        # main import
        print("=" * 80)
        print("COURT SPEED AND ACE% IMPORT")
        print("=" * 80)
        
        # DEBUG: show current directory + target directory
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for files in: {os.path.abspath(COURT_SPEED_DIR)}")
        
        # find all court speed CSV files
        csv_pattern = os.path.join(COURT_SPEED_DIR, "court_speeds_*.csv")
        csv_files = sorted(glob.glob(csv_pattern))
        
        if not csv_files:
            print(f"No court speed files found in {COURT_SPEED_DIR}")
            print(f"Looking for files matching pattern: {csv_pattern}")
            
            # DEBUG: check if directory exists
            if os.path.exists(COURT_SPEED_DIR):
                print(f"Directory exists. Files in directory:")
                try:
                    files = os.listdir(COURT_SPEED_DIR)
                    for f in files[:10]: # show first 10 files
                        print(f"  - {f}")
                except Exception as e:
                    print(f"Error listing directory: {e}")
            else: print(f"Directory does not exist: {COURT_SPEED_DIR}")
            return
        
        print(f"Found {len(csv_files)} court speed files to process")
        
        # process each file
        for filepath in csv_files:
            try: self.process_court_speed_file(filepath)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
        
        # print summary stats
        print("\n" + "=" * 80)
        print("IMPORT SUMMARY")
        print("=" * 80)
        print(f"Total tournament entries processed: {self.stats['total_processed']}")
        print(f"Successfully matched: {self.stats['matched']}")
        print(f"Failed to match: {self.stats['failed']}")
        
        if self.stats['failed'] > 0:
            print(f"\nSuccess rate: {(self.stats['matched'] / self.stats['total_processed']) * 100:.1f}%")
            
            # sve failed tournaments to file for review
            failed_file = "failed_court_speed_matches.txt"
            with open(failed_file, 'w') as f:
                f.write("Failed Tournament Matches\n")
                f.write("=" * 50 + "\n")
                for tournament in sorted(self.stats['failed_tournaments']): f.write(f"{tournament}\n")
            print(f"\nFailed matches saved to: {failed_file}")
        else: print("\nAll tournaments successfully matched!")
        
        # verify import by checking a few tournaments
        print("\n" + "-" * 50)
        print("Verification: Checking Updated Tournaments:")
        print("-" * 50)
        
        with self.driver.session() as session:
            # check specific tournaments that were failing before
            test_tournaments = [
                ("Adelaide", 2003),
                ("Chennai", 2003),
                ("Doha", 2003),
                ("Brisbane", 2014),
                ("Doha", 2014)
            ]
            
            print("\nChecking previously failed tournaments:")
            for tourn_name, tourn_year in test_tournaments:
                # check BOTH 1) current year and 2) Dec of previous year
                result = session.run("""
                    MATCH (t:Tournament)
                    WHERE (t.date >= $dec_start AND t.date < $jan_end)
                    AND toLower(t.name) CONTAINS toLower($tournament_name)
                    AND t.ace_pct IS NOT NULL
                    RETURN t.name as name, t.date as date, t.ace_pct as ace_pct, 
                           t.surface_speed as surface_speed
                    LIMIT 1
                """, 
                dec_start=int(f"{tourn_year-1}1225"),
                jan_end=int(f"{tourn_year}0115"),
                tournament_name=tourn_name)
                
                record = result.single()
                if record:
                    date_str = str(record['date'])
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    print(f"  [V] {tourn_name} {tourn_year}: Found as {record['name']} on {formatted_date}")
                    print(f"    Ace% = {record['ace_pct']:.2f}, Speed = {record['surface_speed']:.2f}")
                else:
                    print(f"  ✗ {tourn_name} {tourn_year}: Not updated")
            
            print("\nSample of all updated tournaments:")
            result = session.run("""
                MATCH (t:Tournament)
                WHERE t.ace_pct IS NOT NULL AND t.surface_speed IS NOT NULL
                RETURN t.name as name, t.date as date, t.ace_pct as ace_pct, 
                       t.surface_speed as surface_speed
                ORDER BY t.date DESC
                LIMIT 5
            """)
            
            for record in result:
                date_str = str(record['date'])
                year = date_str[:4]
                print(f"  {year} {record['name']}: Ace% = {record['ace_pct']:.2f}, Speed = {record['surface_speed']:.2f}")

def main():
    importer = CourtSpeedImporter()
    
    try: importer.run_import()
    except Exception as e:
        print(f"Error during import: {e}")
        raise
    finally: importer.close()
    
    print("\nImport process completed!")

if __name__ == "__main__":
    main()