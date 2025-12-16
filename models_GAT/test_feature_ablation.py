#!/usr/bin/env python3

# feature ablation test for GAT
#   tests whether differential features alone explain model's performance

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

def load_data():
    # load the V2 data
    data = torch.load('gat_cache/gat_torch_data_v2.pt')
    return data

def test_differential_only():
    # can differential features alone get similar accuracy?
    print("="*70)
    print("ABLATION TEST: Differential Features Only")
    print("="*70)
    
    data = load_data()
    
    # differential features are indices 232-269
    DIFF_RANGE = (232, 270)
    
    # extract only differential features
    X_train = data['train']['features'][:, DIFF_RANGE[0]:DIFF_RANGE[1]].numpy()
    y_train = data['train']['labels'].numpy()
    
    X_val = data['val']['features'][:, DIFF_RANGE[0]:DIFF_RANGE[1]].numpy()
    y_val = data['val']['labels'].numpy()
    
    X_test = data['test']['features'][:, DIFF_RANGE[0]:DIFF_RANGE[1]].numpy()
    y_test = data['test']['labels'].numpy()
    
    print(f"\nDifferential features shape: {X_train.shape[1]} features")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # handle NaN
    X_train = np.nan_to_num(X_train, 0)
    X_val = np.nan_to_num(X_val, 0)
    X_test = np.nan_to_num(X_test, 0)
    
    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # simple logreg on diff features only
    print("\nTraining Logistic Regression on DIFF features only...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    # predictions
    train_pred = lr.predict_proba(X_train_scaled)[:, 1]
    val_pred = lr.predict_proba(X_val_scaled)[:, 1]
    test_pred = lr.predict_proba(X_test_scaled)[:, 1]
    
    print("\n" + "-"*50)
    print("RESULTS: Differential Features Only (Logistic Regression)")
    print("-"*50)
    
    for name, y_true, y_prob in [('Train', y_train, train_pred), ('Val', y_val, val_pred), ('Test', y_test, test_pred)]:
        acc = accuracy_score(y_true, (y_prob > 0.5).astype(int))
        auc = roc_auc_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        print(f"{name:5s}: Acc={acc:.4f} ({acc*100:.2f}%)  AUC={auc:.4f}  Brier={brier:.4f}")
    
    # show which diff features have highest coefficients
    print("\n" + "-"*50)
    print("Top Differential Features by Coefficient Magnitude:")
    print("-"*50)
    
    # feature names for diff
    diff_names = [
        'diff_rank', 'diff_elo_overall', 'diff_elo_surface', 'diff_glicko_rating',
        'diff_stats_52w_win_pct', 'diff_stats_52w_serve_points_won_pct',
        'diff_stats_52w_return_points_won_pct', 'diff_stats_52w_total_points_won_pct',
        'diff_stats_4w_bp_converted_pct', 'diff_stats_4w_bp_created_per_return_game',
        'diff_stats_4w_bp_faced_per_game', 'diff_stats_4w_bp_saved_pct',
        'diff_stats_4w_first_serve_pct', 'diff_stats_4w_first_serve_won_pct',
        'diff_stats_4w_return_game_impact', 'diff_stats_4w_return_games_broken_pct',
        'diff_stats_4w_return_points_won_pct', 'diff_stats_4w_second_serve_won_pct',
        'diff_stats_4w_serve_points_won_pct', 'diff_stats_4w_service_game_efficiency',
        'diff_stats_4w_straight_sets_pct', 'diff_stats_4w_total_points_won_pct',
        'diff_stats_4w_win_pct',
        'diff_stats_8w_bp_converted_pct', 'diff_stats_8w_bp_created_per_return_game',
        'diff_stats_8w_bp_faced_per_game', 'diff_stats_8w_bp_saved_pct',
        'diff_stats_8w_first_serve_pct', 'diff_stats_8w_first_serve_won_pct',
        'diff_stats_8w_return_game_impact', 'diff_stats_8w_return_games_broken_pct',
        'diff_stats_8w_return_points_won_pct', 'diff_stats_8w_second_serve_won_pct',
        'diff_stats_8w_serve_points_won_pct', 'diff_stats_8w_service_game_efficiency',
        'diff_stats_8w_straight_sets_pct', 'diff_stats_8w_total_points_won_pct',
        'diff_stats_8w_win_pct'
    ]
    
    coef_importance = list(zip(diff_names, np.abs(lr.coef_[0])))
    coef_importance.sort(key=lambda x: x[1], reverse=True)
    
    for name, coef in coef_importance[:10]:
        print(f"  {name:45s}: {coef:.4f}")
    
    return test_pred, y_test


def test_key_features_only():
    # test: what about just rank + elo + glicko differentials?
    
    print("\n" + "="*70)
    print("ABLATION TEST: Only Rank + ELO + Glicko Differentials")
    print("="*70)
    
    data = load_data()
    
    # just the 4 key rating differentials: indices 232-235
    KEY_DIFF = [232, 233, 234, 235]
    
    X_train = data['train']['features'][:, KEY_DIFF].numpy()
    y_train = data['train']['labels'].numpy()
    
    X_test = data['test']['features'][:, KEY_DIFF].numpy()
    y_test = data['test']['labels'].numpy()
    
    print(f"\nUsing only: diff_rank, diff_elo_overall, diff_elo_surface, diff_glicko_rating")
    
    # handle NaN
    X_train = np.nan_to_num(X_train, 0)
    X_test = np.nan_to_num(X_test, 0)
    
    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # logreg
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    test_pred = lr.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))
    auc = roc_auc_score(y_test, test_pred)
    brier = brier_score_loss(y_test, test_pred)
    
    print(f"\nTest: Acc={acc:.4f} ({acc*100:.2f}%)  AUC={auc:.4f}  Brier={brier:.4f}")
    
    print("\nCoefficients:")
    for name, coef in zip(['diff_rank', 'diff_elo_overall', 'diff_elo_surface', 'diff_glicko_rating'], lr.coef_[0]):
        print(f"  {name}: {coef:.4f}")


def test_without_diff():
    # Test: what happens if I REMOVE differential features?
    
    print("\n" + "="*70)
    print("ABLATION TEST: GAT Model WITHOUT Differential Features")
    print("="*70)
    
    data = load_data()
    
    # all features EXCEPT differentials (0-231 only)
    X_train = data['train']['features'][:, :232].numpy()
    y_train = data['train']['labels'].numpy()
    
    X_test = data['test']['features'][:, :232].numpy()
    y_test = data['test']['labels'].numpy()
    
    print(f"\nUsing P1 + P2 features only: {X_train.shape[1]} features (no diff)")
    
    # handle NaN
    X_train = np.nan_to_num(X_train, 0)
    X_test = np.nan_to_num(X_test, 0)
    
    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # logreg
    print("Training Logistic Regression on P1+P2 features only...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    test_pred = lr.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))
    auc = roc_auc_score(y_test, test_pred)
    brier = brier_score_loss(y_test, test_pred)
    
    print(f"\nTest: Acc={acc:.4f} ({acc*100:.2f}%)  AUC={auc:.4f}  Brier={brier:.4f}")


def compare_to_gat():
    # summary compare --> NEED TO UPDATE THIS SO ACTUAL VALUES GO IN
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print("""
    Model                                    Test Acc    Test AUC
    ─────────────────────────────────────────────────────────────
    Diff features only (LogReg)              ???         ???
    Key ratings diff only (LogReg)           ???         ???
    P1+P2 features, no diff (LogReg)         ???         ???
    """)


if __name__ == "__main__":
    test_differential_only()
    test_key_features_only()
    test_without_diff()
    compare_to_gat()