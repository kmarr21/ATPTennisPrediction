# script to retrieve and display all properties for the 2008 Wimbledon Final -- TESTER SCRIPT

from neo4j import GraphDatabase
import json

# Neo4j connection parameters
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "put_your_password_here"

def get_wimbledon_2008_final():
    # retrieve 2008 Wimbledon final match w/ all related nodes
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            # Query to find the 2008 Wimbledon Final
            # (know it's Federer vs. Nadal, F round)
            query = """
            MATCH (p1:Player)-[:WON|LOST]->(m:Match)<-[:WON|LOST]-(p2:Player)
            WHERE m.round = 'F' 
            AND m.tourney_id CONTAINS '2008-540'  // Wimbledon tournament ID pattern
            MATCH (m)-[:PLAYED_IN]->(t:Tournament)
            RETURN m, t, p1, p2
            """
            
            # alt query if the above doesn't work
            alternative_query = """
            MATCH (t:Tournament)
            WHERE t.name = 'Wimbledon' AND t.date >= 20080601 AND t.date <= 20080731
            MATCH (m:Match)-[:PLAYED_IN]->(t)
            WHERE m.round = 'F'
            MATCH (p1:Player)-[:WON|LOST]->(m)<-[:WON|LOST]-(p2:Player)
            RETURN m, t, p1, p2
            """
            
            print("=" * 80)
            print("2008 WIMBLEDON FINAL - COMPLETE NODE PROPERTIES")
            print("=" * 80)
            
            # try query #1:
            result = session.run(query)
            records = list(result)
            
            # if no results, try alt query:
            if not records:
                print("First query didn't find the match, trying alternative...")
                result = session.run(alternative_query)
                records = list(result)
            
            if records:
                record = records[0]
                match_node = record['m']
                tournament_node = record['t']
                player1_node = record['p1']
                player2_node = record['p2']
                
                # show MATCH NODE
                print("\n1. MATCH NODE PROPERTIES")
                print("-" * 40)
                match_props = dict(match_node)
                print(f"Total match properties: {len(match_props)}")
                print("\nAll properties:")
                for key in sorted(match_props.keys()):
                    value = match_props[key]
                    print(f"  {key}: {value}")
                
                # show TOURNAMENT NODE
                print("\n2. TOURNAMENT NODE PROPERTIES")
                print("-" * 40)
                tournament_props = dict(tournament_node)
                print(f"Total tournament properties: {len(tournament_props)}")
                print("\nAll properties:")
                for key in sorted(tournament_props.keys()):
                    value = tournament_props[key]
                    print(f"  {key}: {value}")
                
                # show PLAYER 1 NODE
                print("\n3. PLAYER 1 NODE PROPERTIES")
                print("-" * 40)
                player1_props = dict(player1_node)
                print(f"Name: {player1_props.get('first_name', '')} {player1_props.get('last_name', '')}")
                print(f"Total player properties: {len(player1_props)}")
                print("\nAll properties:")
                for key in sorted(player1_props.keys()):
                    value = player1_props[key]
                    print(f"  {key}: {value}")
                
                # show PLAYER 2 NODE
                print("\n4. PLAYER 2 NODE PROPERTIES")
                print("-" * 40)
                player2_props = dict(player2_node)
                print(f"Name: {player2_props.get('first_name', '')} {player2_props.get('last_name', '')}")
                print(f"Total player properties: {len(player2_props)}")
                print("\nAll properties:")
                for key in sorted(player2_props.keys()):
                    value = player2_props[key]
                    print(f"  {key}: {value}")
                
                # count stats
                print("\n" + "=" * 80)
                print("MATCH STATISTICS SUMMARY")
                print("=" * 80)
                
                # separate match stats from other properties
                winner_stats = [k for k in match_props.keys() if k.startswith('w_')]
                loser_stats = [k for k in match_props.keys() if k.startswith('l_')]
                other_props = [k for k in match_props.keys() if not k.startswith('w_') and not k.startswith('l_')]
                
                print(f"\nWinner statistics fields: {len(winner_stats)}")
                print(f"  {', '.join(sorted(winner_stats))}")
                
                print(f"\nLoser statistics fields: {len(loser_stats)}")
                print(f"  {', '.join(sorted(loser_stats))}")
                
                print(f"\nOther match properties: {len(other_props)}")
                print(f"  {', '.join(sorted(other_props))}")
                
                print(f"\nTOTAL match node properties: {len(match_props)}")
                
            else:
                print("Could not find the 2008 Wimbledon Final.")
                print("\nChecking what Wimbledon tournaments we have around 2008:")
                
                check_query = """
                MATCH (t:Tournament)
                WHERE t.name CONTAINS 'Wimbledon' 
                AND t.date >= 20070101 AND t.date <= 20090101
                RETURN t.name, t.date, t.id
                ORDER BY t.date
                """
                
                result = session.run(check_query)
                print("\nWimbledon tournaments 2007-2009:")
                for record in result:
                    print(f"  {record['t.name']}: Date={record['t.date']}, ID={record['t.id']}")
                
    finally:
        driver.close()

if __name__ == "__main__":
    get_wimbledon_2008_final()