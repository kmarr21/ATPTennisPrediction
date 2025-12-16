#!/usr/bin/env python3

# script to delete ELO nodes and HAS_ELO relationships

from neo4j import GraphDatabase

# Neo4j connection parameters
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "put_your_password_here"

class ELODeleter:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("=" * 70)
        print("ELO NODE DELETION UTILITY")
        print("=" * 70)
        
    def close(self):
        self.driver.close()
    
    def delete_elo_nodes(self):
        # delete all ELO nodes + relationships
        with self.driver.session() as session:
            # check what's there
            result = session.run("""
                MATCH (e:ELO)
                RETURN count(e) as node_count""")
            node_count = result.single()['node_count']
            
            result = session.run("""
                MATCH ()-[r:HAS_ELO]->()
                RETURN count(r) as rel_count""")
            rel_count = result.single()['rel_count']
            
            if node_count == 0:
                print("No ELO nodes found.")
                return
            
            print(f"Found {node_count:,} ELO nodes")
            print(f"Found {rel_count:,} HAS_ELO relationships")
            
            response = input("\nDelete all ELO data? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Cancelled.")
                return
            
            print("\nDeleting...")
            batch_size = 10000
            total_deleted = 0
            
            while True:
                result = session.run("""
                    MATCH (e:ELO)
                    WITH e LIMIT $batch_size
                    DETACH DELETE e
                    RETURN count(e) as deleted
                """, batch_size=batch_size)
                
                deleted = result.single()['deleted']
                total_deleted += deleted
                
                if deleted == 0:
                    break
                
                if total_deleted % 50000 == 0:
                    print(f"  Deleted {total_deleted:,} nodes...")
            
            print(f"Deleted {total_deleted:,} ELO nodes")

def main():
    deleter = ELODeleter()
    try:
        deleter.delete_elo_nodes()
    finally:
        deleter.close()

if __name__ == "__main__":
    main()