#!/usr/bin/env python3
"""
Generate Grand Slam Evaluation Files for All Models
====================================================

Generates prediction files for all baseline models on Grand Slam tournaments
from 2014-2024. Creates separate outputs for overall and surface-specific
versions of ELO and Glicko2 models.

Usage:
    python generate_grand_slam_evaluations.py

Output:
    All files saved to model_outputs/ directory with standardized naming
"""

import subprocess
import os
import sys
from datetime import datetime

# Grand Slam tournaments with their surfaces
GRAND_SLAMS = {
    'Australian Open': 'hard',
    'Roland Garros': 'clay',
    'Wimbledon': 'grass',
    'US Open': 'hard'
}

# Years to evaluate
YEARS = list(range(2014, 2025))  # 2014-2024

# Path to models directory
MODELS_DIR = os.path.expanduser("~/Desktop/base_models/models")

def run_command(command, description):
    """Run a command and handle errors"""
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
            timeout=300  # 5 minute timeout per command
        )
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: {description}")
            # Print last few lines of output for confirmation
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                print("  Output (last 5 lines):")
                for line in lines[-5:]:
                    print(f"    {line}")
        else:
            print(f"✗ FAILED: {description}")
            print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {description} took longer than 5 minutes")
        return False
    except Exception as e:
        print(f"✗ ERROR: {description} - {str(e)}")
        return False
    
    return True

def generate_evaluations():
    """Generate all evaluation files"""
    
    start_time = datetime.now()
    print("\n" + "="*70)
    print("GRAND SLAM EVALUATION FILE GENERATOR")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Output directory: {MODELS_DIR}/model_outputs/")
    print(f"\nGenerating evaluations for:")
    print(f"  Tournaments: {', '.join(GRAND_SLAMS.keys())}")
    print(f"  Years: {YEARS[0]}-{YEARS[-1]}")
    print(f"  Models: rankings (4 variants), elo (2 variants), glicko2 (2 variants), h2h")
    
    # Check if models directory exists
    if not os.path.exists(MODELS_DIR):
        print(f"\n✗ ERROR: Models directory not found: {MODELS_DIR}")
        sys.exit(1)
    
    # Track statistics
    total_commands = 0
    successful_commands = 0
    failed_commands = []
    
    # Iterate through each Grand Slam
    for tournament, surface in GRAND_SLAMS.items():
        print(f"\n{'#'*70}")
        print(f"# TOURNAMENT: {tournament} ({surface.upper()})")
        print(f"{'#'*70}")
        
        # Iterate through each year
        for year in YEARS:
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
                if run_command(command, f"Rankings {model_type} - {tournament} {year}"):
                    successful_commands += 1
                else:
                    failed_commands.append(f"Rankings {model_type} - {tournament} {year}")
            
            # Rankings Joint
            command = [
                'python3', 'rankings_joint.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament
            ]
            total_commands += 1
            if run_command(command, f"Rankings joint - {tournament} {year}"):
                successful_commands += 1
            else:
                failed_commands.append(f"Rankings joint - {tournament} {year}")
            
            # Rankings Enhanced
            command = [
                'python3', 'rankings_enhanced.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament
            ]
            total_commands += 1
            if run_command(command, f"Rankings enhanced - {tournament} {year}"):
                successful_commands += 1
            else:
                failed_commands.append(f"Rankings enhanced - {tournament} {year}")
            
            # 2. ELO - OVERALL
            command = [
                'python3', 'elo.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament
            ]
            total_commands += 1
            if run_command(command, f"ELO overall - {tournament} {year}"):
                successful_commands += 1
            else:
                failed_commands.append(f"ELO overall - {tournament} {year}")
            
            # 3. ELO - SURFACE-SPECIFIC
            command = [
                'python3', 'elo.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament,
                '--surface', surface
            ]
            total_commands += 1
            if run_command(command, f"ELO {surface} - {tournament} {year}"):
                successful_commands += 1
            else:
                failed_commands.append(f"ELO {surface} - {tournament} {year}")
            
            # 4. GLICKO2 - OVERALL
            command = [
                'python3', 'glicko2.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament
            ]
            total_commands += 1
            if run_command(command, f"Glicko2 overall - {tournament} {year}"):
                successful_commands += 1
            else:
                failed_commands.append(f"Glicko2 overall - {tournament} {year}")
            
            # 5. GLICKO2 - SURFACE-SPECIFIC
            command = [
                'python3', 'glicko2.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament,
                '--surface', surface
            ]
            total_commands += 1
            if run_command(command, f"Glicko2 {surface} - {tournament} {year}"):
                successful_commands += 1
            else:
                failed_commands.append(f"Glicko2 {surface} - {tournament} {year}")
            
            # 6. H2H
            command = [
                'python3', 'h2h_simple.py',
                '--batch-evaluate',
                '--year', str(year),
                '--tournament', tournament
            ]
            total_commands += 1
            if run_command(command, f"H2H - {tournament} {year}"):
                successful_commands += 1
            else:
                failed_commands.append(f"H2H - {tournament} {year}")
    
    # Final summary
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
    print(f"  Success rate: {successful_commands/total_commands*100:.1f}%")
    
    if failed_commands:
        print(f"\nFailed commands:")
        for cmd in failed_commands:
            print(f"  - {cmd}")
    
    print(f"\nOutput files saved to: {MODELS_DIR}/model_outputs/")
    print("\nExpected number of files: " + 
          f"{len(GRAND_SLAMS)} tournaments × {len(YEARS)} years × 9 model variants = " +
          f"{len(GRAND_SLAMS) * len(YEARS) * 9} files")

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