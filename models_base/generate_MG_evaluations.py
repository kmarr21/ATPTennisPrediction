#!/usr/bin/env python3

# script to generaate tournament evaluations for all base models
"""
Generates prediction files for all baseline models on tournaments from tournaments_to_analyze.csv
Creates separate outputs for overall and surface-specific versions of ELO and Glicko2 models

Usage: python generate_MG_evaluations.py
Output: all files saved to model_outputs_M_G/ directory with standardized naming
"""

import subprocess
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# path to models directory
MODELS_DIR = os.path.expanduser("~/Desktop/base_models/models")

def run_command(command, description):
    # run command + handle errors
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print('='*70)
    
    try:
        result = subprocess.run(
            command,
            cwd=MODELS_DIR,
            capture_output=True,
            text=True,
            timeout=1200  #20 minute timeout per command
        )
        
        if result.returncode == 0:
            print(f"[V] SUCCESS: {description}")
            # print last few lines of output for confirmation
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                print("  Output (last 5 lines):")
                for line in lines[-5:]:
                    print(f"    {line}")
        else:
            print(f"  FAILED: {description}")
            print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"X TIMEOUT: {description} took longer than 20 minutes")
        return False
    except Exception as e:
        print(f"X ERROR: {description} - {str(e)}")
        return False
    
    return True

def generate_evaluations():
    # generate all eval files
    start_time = datetime.now()
    print("\n" + "="*70)
    print("TOURNAMENT EVALUATION FILE GENERATOR (FROM CSV)")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models directory: {MODELS_DIR}")
    
    # create output directory
    output_dir = Path(MODELS_DIR) / "model_outputs_M_G"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # load tournaments from CSV
    csv_path = "tournaments_to_analyze.csv"
    if not os.path.exists(csv_path):
        print(f"\n✗ ERROR: Tournament CSV not found: {csv_path}")
        sys.exit(1)
    
    tournaments_df = pd.read_csv(csv_path)
    
    # filter to tournaments with valid data
    valid_tournaments = tournaments_df[tournaments_df['tournament_date'].notna()].copy()
    
    print(f"\nLoaded {len(valid_tournaments)} valid tournaments from {csv_path}")
    print(f"Years: {valid_tournaments['year'].min()}-{valid_tournaments['year'].max()}")
    print(f"Tournaments: {len(valid_tournaments['tournament_name'].unique())} unique")
    print(f"Models: rankings (4 variants), elo (2 variants), glicko2 (2 variants), h2h")
    
    # check if models directory exists
    if not os.path.exists(MODELS_DIR):
        print(f"\n✗ ERROR: Models directory not found: {MODELS_DIR}")
        sys.exit(1)
    
    # track stats
    total_commands = 0
    successful_commands = 0
    failed_commands = []
    
    # group tournaments by name
    tournament_groups = valid_tournaments.groupby('tournament_name')
    
    for tournament_name, group in tournament_groups:
        print(f"\n{'#'*70}")
        print(f"# TOURNAMENT: {tournament_name}")
        print(f"{'#'*70}")
        
        # get surface (should be consistent for a tournament)
        surfaces = group['surface'].unique()
        if len(surfaces) > 1: print(f"  WARNING: Multiple surfaces found for {tournament_name}: {surfaces}")
        surface = surfaces[0].lower() if pd.notna(surfaces[0]) else 'hard'
        
        print(f"  Surface: {surface.upper()}")
        print(f"  Years: {sorted(group['year'].unique())}")
        
        # iterate through all years by tournament
        for _, tournament_row in group.iterrows():
            year = int(tournament_row['year'])
            tournament = tournament_row['tournament_name']
            
            print(f"\n--- {tournament} {year} ---")
            
            # 1. RANKINGS MODELS (4 variants)
            ranking_models = ['position', 'points']
            for model_type in ranking_models:
                command = [
                    'python3', 'rankings_simple.py',
                    '--batch-evaluate',
                    '--model', model_type,
                    '--year', str(year),
                    '--tournament', tournament
                ]
                total_commands += 1
                if run_command(command, f"Rankings {model_type} - {tournament} {year}"): successful_commands += 1
                else: failed_commands.append(f"Rankings {model_type} - {tournament} {year}")
            
            # Rankings Joint
            command = [
                'python3', 'rankings_joint.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament
            ]
            total_commands += 1
            if run_command(command, f"Rankings joint - {tournament} {year}"): successful_commands += 1
            else: failed_commands.append(f"Rankings joint - {tournament} {year}")
            
            # Rankings Enhanced
            command = [
                'python3', 'rankings_enhanced.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament
            ]
            total_commands += 1
            if run_command(command, f"Rankings enhanced - {tournament} {year}"): successful_commands += 1
            else: failed_commands.append(f"Rankings enhanced - {tournament} {year}")
            
            # 2. ELO: OVERALL
            command = [
                'python3', 'elo.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament
            ]
            total_commands += 1
            if run_command(command, f"ELO overall - {tournament} {year}"): successful_commands += 1
            else: failed_commands.append(f"ELO overall - {tournament} {year}")
            
            # 3. ELO: SURFACE-SPECIFIC
            command = [
                'python3', 'elo.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament,
                '--surface', surface
            ]
            total_commands += 1
            if run_command(command, f"ELO {surface} - {tournament} {year}"): successful_commands += 1
            else: failed_commands.append(f"ELO {surface} - {tournament} {year}")
            
            # 4. GLICKO2: OVERALL
            command = [
                'python3', 'glicko2.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament
            ]
            total_commands += 1
            if run_command(command, f"Glicko2 overall - {tournament} {year}"): successful_commands += 1
            else: failed_commands.append(f"Glicko2 overall - {tournament} {year}")
            
            # 5. GLICKO2: SURFACE-SPECIFIC
            command = [
                'python3', 'glicko2.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament,
                '--surface', surface
            ]
            total_commands += 1
            if run_command(command, f"Glicko2 {surface} - {tournament} {year}"): successful_commands += 1
            else: failed_commands.append(f"Glicko2 {surface} - {tournament} {year}")
            
            # 6. H2H
            command = [
                'python3', 'h2h_simple.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament
            ]
            total_commands += 1
            if run_command(command, f"H2H - {tournament} {year}"): successful_commands += 1
            else: failed_commands.append(f"H2H - {tournament} {year}")
    
    # final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("EVALUATION GENERATION COMPLETE")
    print("="*70)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"\nResults:")
    print(f"  Total commands: {total_commands}")
    print(f"  Successful: {successful_commands}")
    print(f"  Failed: {len(failed_commands)}")
    if total_commands > 0:
        print(f"  Success rate: {successful_commands/total_commands*100:.1f}%")
    
    if failed_commands:
        print(f"\nFailed commands:")
        for cmd in failed_commands: print(f"  - {cmd}")
    
    print(f"\nOutput files saved to: {output_dir}")
    
    # expected files calc
    n_tournaments = len(valid_tournaments)
    n_models = 9 # 4 rankings + 2 elo + 2 glicko2 + 1 h2h
    print(f"\nExpected number of files: {n_tournaments} tournaments × {n_models} model variants = {n_tournaments * n_models} files")
    
    # summary by tournament
    print("\nTournaments processed:")
    for tournament_name, group in tournament_groups:
        years = sorted(group['year'].unique())
        print(f"  {tournament_name}: {len(years)} years ({min(years)}-{max(years)})")

if __name__ == "__main__":
    try:
        generate_evaluations()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)