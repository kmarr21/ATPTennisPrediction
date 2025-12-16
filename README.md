# Predicting ATP Match Outcomes: From Traditional Rating Systems to Graph Attention Networks

> **Work in Progress**: This project was started as part of an MS Data Science capstone at Tufts University, but certain adjustments are still being made in this repo. 

## Overview

This project explores ATP tennis match prediction through a progression of modeling approaches, from traditional rating systems to graph-based neural networks. The goal is to investigate whether modern machine learning techniques can capture matchup-specific dynamics that single-number rating systems miss.

### Motivation

Traditional tennis prediction relies on rating systems like Elo or ATP rankings, which summarize a player's strength as a single number. But tennis matchups are often more nuanced: an avid tennis watcher/player will notice style-specific mismatches that may make them think the "non-favorite" will win; and they're often right. This project asks: can we model these stylistic interactions to improve predictions?

## Project Structure
```
├── import_scripts/       # Scripts to populate Neo4j database
├── models_base/          # Traditional rating models (Elo, Glicko-2, rankings)
├── models_NMF/           # Non-negative Matrix Factorization to add style factors to Elo and Glicko-2 models
│   ├── loading/          # Generate and load style factors
│   └── predicting/       # NMF-augmented prediction models
└── models_GAT/           # Graph Attention Network-inspired implementation
```

Each model directory contains its own documentation (`.md` files) with methodology details.

## Database

Match data and derived features are stored in a local **Neo4j** graph database, enabling flexible querying of player relationships, head-to-head records, and temporal match sequences. Import scripts are available to recreate this database if you have the data downloaded in the correct format to your local machine. 

## Data Sources

1. [Tennis Abstract](https://www.tennisabstract.com/) — Historical match data and statistics
2. [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject) (Jeff Sackmann) — Point-by-point charting data
3. [ATP Tour](https://www.atptour.com/en) — Official tour data

## License

This project is for academic purposes. Data sources retain their original licenses. Repository license applies: do not distribute commerically or use without explicit permission.