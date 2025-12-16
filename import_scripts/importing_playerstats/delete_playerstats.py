# script to DELETE all PlayerStats nodes and their relationships (ONLY PlayerStats nodes)

from neo4j import GraphDatabase
import time

# Neo4j connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "put_your_password_here"

def delete_playerstats_nodes():
    # delete all playerstats nodes + relationships
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            # first, let's count how many we're deleting....
            print("=" * 80)
            print("PLAYERSTATS NODE DELETION")
            print("=" * 80)
            
            result = session.run("MATCH (ps:PlayerStats) RETURN COUNT(ps) as count")
            count = result.single()['count']
            
            if count == 0:
                print("No PlayerStats nodes found to delete.")
                return
            
            print(f"Found {count:,} PlayerStats nodes to delete")
            
            # CONFIRM
            response = input("\nAre you sure you want to delete ALL PlayerStats nodes? (yes/no): ")
            if response.lower() != 'yes':
                print("Deletion cancelled.")
                return
            
            print("\nDeleting PlayerStats nodes in batches...")
            start_time = time.time()
            
            # delete in batches to avoid memory issues
            batch_size = 10000
            total_deleted = 0
            
            while True:
                # delete batch
                result = session.run("""
                    MATCH (ps:PlayerStats)
                    WITH ps LIMIT $batch_size
                    DETACH DELETE ps
                    RETURN COUNT(ps) as deleted
                """, batch_size=batch_size)
                
                deleted = result.single()['deleted']
                total_deleted += deleted
                
                if deleted == 0: break
                
                print(f"  Deleted {total_deleted:,} / {count:,} nodes...")
            
            # verify deleted
            result = session.run("MATCH (ps:PlayerStats) RETURN COUNT(ps) as remaining")
            remaining = result.single()['remaining']
            
            elapsed = time.time() - start_time
            
            print(f"\n{'='*80}")
            print("DELETION COMPLETE")
            print(f"{'='*80}")
            print(f"Deleted: {total_deleted:,} nodes")
            print(f"Remaining: {remaining} nodes")
            print(f"Time taken: {elapsed:.1f} seconds")
            
            if remaining == 0:
                print("\n All PlayerStats nodes successfully deleted!")
            else:
                print(f"\n Warning: {remaining} nodes could not be deleted")
                
    finally:
        driver.close()

if __name__ == "__main__":
    delete_playerstats_nodes()