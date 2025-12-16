#!/usr/bin/env python3

# script to generate tournament list CSV
#   creates a CSV of tournaments with detailed information for analysis

from neo4j import GraphDatabase
import pandas as pd
from datetime import datetime

def get_tournaments(driver):
    #get list of specified tournaments from db
    
    # tournaments wanted:
    tournaments_to_analyze = [
        # Grand Slams
        ('Wimbledon', 2024), ('Wimbledon', 2019), ('Wimbledon', 2016), ('Wimbledon', 2012), ('Wimbledon', 2008),
        ('Australian Open', 2024), ('Australian Open', 2019), ('Australian Open', 2016), ('Australian Open', 2012), ('Australian Open', 2008),
        ('Roland Garros', 2024), ('Roland Garros', 2019), ('Roland Garros', 2016), ('Roland Garros', 2012), ('Roland Garros', 2008),
        ('US Open', 2024), ('US Open', 2019), ('US Open', 2016), ('US Open', 2012), ('US Open', 2008),
        
        # Masters 1000
        ('Indian Wells Masters', 2024), ('Indian Wells Masters', 2019), ('Indian Wells Masters', 2016), ('Indian Wells Masters', 2012), ('Indian Wells Masters', 2008),
        ('Miami Masters', 2024), ('Miami Masters', 2019), ('Miami Masters', 2016), ('Miami Masters', 2012), ('Miami Masters', 2008),
        ('Rome Masters', 2024), ('Rome Masters', 2019), ('Rome Masters', 2016), ('Rome Masters', 2012), ('Rome Masters', 2008),
        ('Cincinnati Masters', 2024), ('Cincinnati Masters', 2019), ('Cincinnati Masters', 2016), ('Cincinnati Masters', 2012), ('Cincinnati Masters', 2008),
    ]
    
    tournament_data = []
    
    with driver.session() as session:
        for tourn_name, year in tournaments_to_analyze:
            # get tournament info from db w/ more details
            query = """
            MATCH (t:Tournament)
            WHERE t.name = $name 
            AND t.date >= $year_start 
            AND t.date <= $year_end
            WITH t
            // Count matches to determine draw size
            OPTIONAL MATCH (m:Match)-[:PLAYED_IN]->(t)
            WITH t, count(m) as total_matches
            // Get a sample match to check best_of
            OPTIONAL MATCH (m2:Match)-[:PLAYED_IN]->(t)
            WITH t, total_matches, COLLECT(m2.best_of)[0] as sample_best_of
            RETURN t.id as tournament_id,
                   t.name as name,
                   t.date as date,
                   t.surface as surface,
                   t.level as level,
                   total_matches,
                   sample_best_of,
                   CASE 
                       WHEN total_matches >= 127 THEN 128
                       WHEN total_matches >= 95 THEN 96
                       WHEN total_matches >= 63 THEN 64
                       WHEN total_matches >= 55 THEN 56
                       WHEN total_matches >= 47 THEN 48
                       WHEN total_matches >= 31 THEN 32
                       WHEN total_matches >= 27 THEN 28
                       ELSE 32
                   END as draw_size
            ORDER BY t.date
            LIMIT 1
            """
            
            year_start = year * 10000 + 101
            year_end = year * 10000 + 1231
            
            result = session.run(query, name=tourn_name, year_start=year_start, year_end=year_end)
            record = result.single()
            
            if record:
                # calc Monday before tournament (for ratings cutoff)
                tourn_date = record['date']
                date_obj = datetime.strptime(str(tourn_date), '%Y%m%d')
                
                # find the Monday before (or same day if it's Monday)
                days_since_monday = date_obj.weekday()
                if days_since_monday == 0:
                    # it's Monday, use previous Monday
                    monday_before = date_obj.timestamp() - 7 * 24 * 3600
                else:
                    # go back to previous Monday
                    monday_before = date_obj.timestamp() - days_since_monday * 24 * 3600
                
                monday_date = datetime.fromtimestamp(monday_before)
                monday_int = int(monday_date.strftime('%Y%m%d'))
                
                # 52 weeks before for H2H analysis
                weeks_52_before = monday_date.timestamp() - 52 * 7 * 24 * 3600
                start_date = datetime.fromtimestamp(weeks_52_before)
                start_int = int(start_date.strftime('%Y%m%d'))
                
                # determine best_of based on tournament level and sample match
                if record['sample_best_of']:
                    best_of = record['sample_best_of']
                else:
                    # default based on tournament type
                    if record['level'] == 'G':  # grand Slam
                        best_of = 5
                    else:  # Masters and others
                        best_of = 3
                
                # refine draw size based on tournament type --> REFINE THIS??
                if record['level'] == 'G':  # grand Slams are always 128
                    draw_size = 128
                elif record['level'] in ['M', '1000']:  # masters can be 96, 64, or 56
                    if record['total_matches'] >= 95:
                        draw_size = 96
                    elif record['total_matches'] >= 55:
                        draw_size = 56
                    else:
                        draw_size = 64
                else:
                    draw_size = record['draw_size']
                
                tournament_data.append({
                    'tournament_name': record['name'],
                    'tournament_id': record['tournament_id'],
                    'year': year,
                    'tournament_date': tourn_date,
                    'surface': record['surface'],
                    'level': record['level'],
                    'best_of': best_of,
                    'draw_size': draw_size,
                    'total_matches': record['total_matches'],
                    'rating_cutoff_date': monday_int,  # monday before tournament
                    'h2h_start_date': start_int,  # 52 weeks before
                    'h2h_end_date': monday_int  # up to Monday before
                })
                
                print(f"Found: {record['name']:25s} {year} - ID: {record['tournament_id']}, "
                      f"Date: {tourn_date}, Surface: {record['surface']}, "
                      f"Best of: {best_of}, Draw: {draw_size}, Matches: {record['total_matches']}")
            else:
                print(f"NOT FOUND: {tourn_name} {year}")
                # still add a placeholder entry
                tournament_data.append({
                    'tournament_name': tourn_name,
                    'tournament_id': f"{tourn_name.replace(' ', '_')}_{year}",
                    'year': year,
                    'tournament_date': None,
                    'surface': None,
                    'level': None,
                    'best_of': None,
                    'draw_size': None,
                    'total_matches': 0,
                    'rating_cutoff_date': None,
                    'h2h_start_date': None,
                    'h2h_end_date': None
                })
    
    return pd.DataFrame(tournament_data)

def main():
    # connect to neo4j
    driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "put_your_password_here"))
    
    try:
        # get tournament data
        df = get_tournaments(driver)
        
        # sort by date (handling None values)
        df['sort_date'] = df['tournament_date'].fillna(0)
        df = df.sort_values(['year', 'sort_date'])
        df = df.drop('sort_date', axis=1)
        
        # save to CSV
        output_file = 'tournaments_to_analyze.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"Saved {len(df)} tournaments to {output_file}")
        print(f"{'='*60}")
        
        # print summary by year
        for year in sorted(df['year'].unique()):
            year_df = df[df['year'] == year]
            print(f"\n{year}:")
            for _, row in year_df.iterrows():
                if pd.notna(row['tournament_date']):
                    print(f"  {row['tournament_name']:25s} - {row['surface']:5s}, "
                          f"Best of {row['best_of']}, Draw: {row['draw_size']:3.0f}")
                else:
                    print(f"  {row['tournament_name']:25s} - NOT FOUND")
        
        # print stats
        found = df[df['tournament_date'].notna()]
        print(f"\nFound {len(found)} tournaments out of {len(df)} requested")
        
        if len(found) > 0:
            print("\nSurface distribution:")
            print(df['surface'].value_counts())
            
            print("\nBest of distribution:")
            print(df['best_of'].value_counts())
        
    finally:
        driver.close()

if __name__ == "__main__":
    main()