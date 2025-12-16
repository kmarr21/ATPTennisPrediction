#!/usr/bin/env python3

# script to delete all Bradley-Terry nodes from the db

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "put_your_password_here"

class BTDeleter:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("=" * 70)
        print("BRADLEY-TERRY NODE DELETION UTILITY")
        print("=" * 70)
        
    def close(self):
        self.driver.close()
    
    def delete_bt_nodes(self):
        # delte all BradleyTerry nodes
        with self.driver.session() as session:
            # check what's there
            result = session.run("""
                MATCH (bt:BradleyTerry)
                RETURN count(bt) as node_count""")
            node_count = result.single()['node_count']
            
            if node_count == 0:
                print("No BradleyTerry nodes found.")
                return
            
            print(f"Found {node_count:,} BradleyTerry nodes")
            
            # check for relationships
            result = session.run("""
                MATCH (bt:BradleyTerry)-[r]-()
                RETURN count(DISTINCT r) as rel_count""")
            rel_count = result.single()['rel_count']
            
            if rel_count > 0:
                print(f"Found {rel_count:,} relationships to BradleyTerry nodes")
            
            response = input("\nDelete all Bradley-Terry data? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Cancelled.")
                return
            
            print("\nDeleting...")
            batch_size = 10000
            total_deleted = 0
            
            while True:
                # use DETACH DELETE to remove relationships too
                result = session.run("""
                    MATCH (bt:BradleyTerry)
                    WITH bt LIMIT $batch_size
                    DETACH DELETE bt
                    RETURN count(bt) as deleted
                """, batch_size=batch_size)
                
                deleted = result.single()['deleted']
                total_deleted += deleted
                
                if deleted == 0:
                    break
                
                if total_deleted % 50000 == 0:
                    print(f"  Deleted {total_deleted:,} nodes...")
            
            print(f"Deleted {total_deleted:,} BradleyTerry nodes")

def main():
    deleter = BTDeleter()
    try:
        deleter.delete_bt_nodes()
    finally:
        deleter.close()

if __name__ == "__main__":
    main()