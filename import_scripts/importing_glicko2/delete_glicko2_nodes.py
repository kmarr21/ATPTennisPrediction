#!/usr/bin/env python3

# script to delete all Glicko2 nodes from the database (all variants)

from neo4j import GraphDatabase
import sys

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "put_your_password_here"

class Glicko2Deleter:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("=" * 70)
        print("GLICKO-2 NODE DELETION UTILITY")
        print("=" * 70)
        
    def close(self):
        self.driver.close()
    
    def delete_glicko2_nodes(self, suffix=""):
        # delete Glicko2 nodes w/ and w/o suffix
        node_label = f"Glicko2{suffix}"
        
        with self.driver.session() as session:
            # check what's there
            result = session.run(f"""
                MATCH (g:{node_label})
                RETURN count(g) as node_count""")
            node_count = result.single()['node_count']
            
            if node_count == 0:
                print(f"No {node_label} nodes found.")
                return 0
            
            print(f"Found {node_count:,} {node_label} nodes")
            
            # check for relationships
            result = session.run(f"""
                MATCH (g:{node_label})-[r]-()
                RETURN count(DISTINCT r) as rel_count""")
            rel_count = result.single()['rel_count']
            
            if rel_count > 0:
                print(f"Found {rel_count:,} relationships to {node_label} nodes")
            
            response = input(f"\nDelete all {node_label} data? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Cancelled.")
                return 0
            
            print(f"\nDeleting {node_label} nodes...")
            batch_size = 10000
            total_deleted = 0
            
            while True:
                result = session.run(f"""
                    MATCH (g:{node_label})
                    WITH g LIMIT $batch_size
                    DETACH DELETE g
                    RETURN count(g) as deleted
                """, batch_size=batch_size)
                
                deleted = result.single()['deleted']
                total_deleted += deleted
                
                if deleted == 0:
                    break
                
                if total_deleted % 50000 == 0:
                    print(f"  Deleted {total_deleted:,} nodes...")
            
            print(f"Deleted {total_deleted:,} {node_label} nodes")
            return total_deleted
    
    def find_all_glicko2_variants(self):
        # find all Glicko2 node variants if there
        with self.driver.session() as session:
            # get all node labels
            result = session.run("CALL db.labels()")
            all_labels = [record[0] for record in result]
            
            # filter for Glicko2 variants
            glicko2_labels = [label for label in all_labels if label.startswith('Glicko2')]
            
            if glicko2_labels:
                print("\nFound Glicko2 variants:")
                for label in glicko2_labels:
                    result = session.run(f"MATCH (g:{label}) RETURN count(g) as count")
                    count = result.single()['count']
                    suffix = label[7:] if len(label) > 7 else ""
                    suffix_display = suffix if suffix else " (default)"
                    print(f"  {label}: {count:,} nodes{suffix_display}")
            else:
                print("\nNo Glicko2 nodes found in database")
            
            return glicko2_labels
    
    def delete_all_variants(self):
        # delete all Glicko2 vairants
        variants = self.find_all_glicko2_variants()
        
        if not variants: return
        
        response = input("\nDelete ALL Glicko2 variants? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Cancelled.")
            return
        
        total_deleted = 0
        for label in variants:
            suffix = label[7:] # remove "Glicko2" prefix
            deleted = self.delete_glicko2_nodes(suffix)
            total_deleted += deleted
        
        print(f"\nTotal deleted across all variants: {total_deleted:,} nodes")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--all': delete_all = True
    else: delete_all = False
    
    deleter = Glicko2Deleter()
    
    try:
        if delete_all:
            deleter.delete_all_variants()
        else:
            # check what variants exist
            variants = deleter.find_all_glicko2_variants()
            
            if not variants:
                print("No Glicko2 nodes to delete.")
                return
            
            if len(variants) == 1:
                # only one variant, delete it
                suffix = variants[0][7:] # remove "Glicko2" prefix
                deleter.delete_glicko2_nodes(suffix)
            else:
                # multiple variants, ask which one
                print("\nWhich variant to delete?")
                print("0: ALL variants")
                for i, label in enumerate(variants, 1):
                    suffix = label[7:] if len(label) > 7 else " (default)"
                    print(f"{i}: {label}{suffix}")
                
                choice = input("\nEnter choice (0 for all, or specific number): ").strip()
                
                try:
                    choice = int(choice)
                    if choice == 0:
                        deleter.delete_all_variants()
                    elif 1 <= choice <= len(variants):
                        suffix = variants[choice-1][7:]
                        deleter.delete_glicko2_nodes(suffix)
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Invalid input.")
    
    finally:
        deleter.close()

if __name__ == "__main__":
    print("Usage:")
    print("  python delete_glicko2_nodes.py         # Interactive mode")
    print("  python delete_glicko2_nodes.py --all   # Delete all variants")
    print()
    main()