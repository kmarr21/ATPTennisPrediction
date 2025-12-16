#!/usr/bin/env python3

# Tennis Matchup GAT Training Script V4
"""
Trains the TennisMatchupGAT V4 model with hyperparameter sweep capability

Features:
- Single run mode (default)
- Hyperparameter sweep mode (--sweep)
- Configurable via command line arguments
- output logging and CSV tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# import V4 model
from tennis_gat_model_v4 import TennisMatchupGAT, count_parameters, FeatureIndices

# seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value):
        if self.best_value is None: self.best_value = value
        else:
            if self.mode == 'min': improved = value < self.best_value - self.min_delta
            else: improved = value > self.best_value + self.min_delta
            
            if improved:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        self.counter = 0
        self.best_value = None
        self.early_stop = False


def load_data(cache_dir='gat_cache'):
    cache_path = Path(cache_dir)
    data_file = cache_path / 'gat_torch_data_v2.pt'
    
    if not data_file.exists(): raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading data from {data_file}...")
    data = torch.load(data_file)
    
    print("\nDATA STATISTICS:")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        features = data[split]['features']
        labels = data[split]['labels']
        print(f"{split.upper():5s}: {features.shape[0]:6,} matches x {features.shape[1]} features")
        print(f"       P1 wins: {labels.sum().item():6.0f} ({labels.mean():.1%})")
    
    return data


def create_dataloaders(data, batch_size=64):
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = TensorDataset(data[split]['features'], data[split]['labels'])
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=False)
    
    return dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, (features, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        features = features.to(device)
        labels = labels.to(device)
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps * features.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, accuracy, auc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * features.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, np.clip(all_preds, 1e-7, 1-1e-7))
    brier = brier_score_loss(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc,
        'log_loss': logloss,
        'brier_score': brier}


def generate_test_predictions(model, data, device):
    model.eval()
    idx = FeatureIndices()
    
    test_features = data['test']['features']
    test_labels = data['test']['labels']
    test_metadata = data['test']['metadata']
    
    p1_rank = test_features[:, idx.P1_OFFSET + idx.RANKINGS[0]].numpy()
    p2_rank = test_features[:, idx.P2_OFFSET + idx.RANKINGS[0]].numpy()
    p1_elo = test_features[:, idx.P1_OFFSET + idx.ELO[0]].numpy()
    p2_elo = test_features[:, idx.P2_OFFSET + idx.ELO[0]].numpy()
    
    all_predictions = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(test_features), batch_size):
            batch = test_features[i:i+batch_size].to(device)
            preds = model(batch).cpu().numpy()
            all_predictions.extend(preds)
    
    all_predictions = np.array(all_predictions)
    
    results_df = pd.DataFrame({
        'match_id': test_metadata['match_id'].values,
        'tournament_id': test_metadata['tournament_id'].values,
        'date': test_metadata['date'].values,
        'surface': test_metadata['surface'].values,
        'round': test_metadata['round'].values,
        'player1_id': test_metadata['player1_id'].values,
        'player2_id': test_metadata['player2_id'].values,
        'p1_rank': p1_rank,
        'p2_rank': p2_rank,
        'p1_elo': p1_elo,
        'p2_elo': p2_elo,
        'p1_win_prob': all_predictions,
        'p2_win_prob': 1 - all_predictions,
        'predicted_winner_id': np.where(
            all_predictions > 0.5,
            test_metadata['player1_id'].values,
            test_metadata['player2_id'].values
        ),
        'actual_winner_id': test_metadata['actual_winner_id'].values,
        'actual_loser_id': test_metadata['actual_loser_id'].values,
        'label': test_labels.numpy()
    })
    
    results_df['correct'] = (results_df['predicted_winner_id'] == results_df['actual_winner_id']).astype(int)
    
    return results_df


def calculate_tournament_metrics(results_df):
    tournament_metrics = []
    
    for tournament_id, tournament_df in results_df.groupby('tournament_id'):
        labels = tournament_df['label'].values
        probs = tournament_df['p1_win_prob'].values
        
        metrics = {
            'tournament_id': tournament_id,
            'date': tournament_df['date'].iloc[0],
            'surface': tournament_df['surface'].iloc[0],
            'num_matches': len(tournament_df),
            'accuracy': tournament_df['correct'].mean(),
            'brier_score': brier_score_loss(labels, probs),
            'log_loss': log_loss(labels, np.clip(probs, 1e-7, 1-1e-7)),
            'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5}
        tournament_metrics.append(metrics)
    
    return pd.DataFrame(tournament_metrics)


def calculate_surface_metrics(results_df):
    surface_metrics = []
    
    for surface, surface_df in results_df.groupby('surface'):
        labels = surface_df['label'].values
        probs = surface_df['p1_win_prob'].values
        
        metrics = {
            'surface': surface,
            'num_matches': len(surface_df),
            'accuracy': surface_df['correct'].mean(),
            'brier_score': brier_score_loss(labels, probs),
            'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
        }
        surface_metrics.append(metrics)
    
    return pd.DataFrame(surface_metrics)


def calculate_round_metrics(results_df):
    round_order = ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']
    round_metrics = []
    
    for round_name in round_order:
        if round_name in results_df['round'].values:
            round_df = results_df[results_df['round'] == round_name]
            labels = round_df['label'].values
            probs = round_df['p1_win_prob'].values
            
            metrics = {
                'round': round_name,
                'num_matches': len(round_df),
                'accuracy': round_df['correct'].mean(),
                'brier_score': brier_score_loss(labels, probs),
                'avg_confidence': np.abs(probs - 0.5).mean() + 0.5
            }
            round_metrics.append(metrics)
    
    return pd.DataFrame(round_metrics)


def train_single_config(config, data, dataloaders, device, output_dir, save_model=True, verbose=True):
    #train a single model configuration & return results
    
    # init model
    model = TennisMatchupGAT(
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)
    
    if verbose:
        print(f"\nModel Parameters: {count_parameters(model):,}")
    
    # setup training
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=False)
    early_stopping = EarlyStopping(patience=config['patience'], mode='max')
    
    # training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []}
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(config['num_epochs']):
        train_loss, train_acc, train_auc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device,
            accumulation_steps=config['accumulation_steps'])
        
        val_metrics = evaluate(model, dataloaders['val'], criterion, device)
        scheduler.step(val_metrics['auc'])
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " <- Best!"
        else:
            marker = ""
        
        if verbose:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train[L:{train_loss:.4f} A:{train_acc:.3f} AUC:{train_auc:.3f}] "
                  f"Val[L:{val_metrics['loss']:.4f} A:{val_metrics['accuracy']:.3f} "
                  f"AUC:{val_metrics['auc']:.3f}]{marker}")
        
        if early_stopping(val_metrics['auc']):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(device)
    
    # generate test preds
    results_df = generate_test_predictions(model, data, device)
    
    # calc metrics
    test_accuracy = results_df['correct'].mean()
    test_brier = brier_score_loss(results_df['label'], results_df['p1_win_prob'])
    test_log_loss = log_loss(results_df['label'], np.clip(results_df['p1_win_prob'], 1e-7, 1-1e-7))
    test_auc = roc_auc_score(results_df['label'], results_df['p1_win_prob'])
    
    test_metrics = {
        'accuracy': test_accuracy,
        'auc': test_auc,
        'log_loss': test_log_loss,
        'brier_score': test_brier,
        'best_val_auc': best_val_auc}
    
    # save outputs if requested
    if save_model:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_str = f"h{config['hidden_dim']}_n{config['num_heads']}_d{int(config['dropout']*100)}"
        
        # save predictions
        results_file = output_dir / f'gat_v4_predictions_{config_str}_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        
        # save model
        model_dir = Path('trained_models')
        model_dir.mkdir(exist_ok=True)
        
        model_file = model_dir / f'tennis_gat_v4_{config_str}_{timestamp}.pth'
        torch.save({
            'model_state_dict': best_model_state,
            'config': config,
            'test_metrics': test_metrics,
            'history': history
        }, model_file)
        
        if verbose:
            print(f"\nPredictions saved to: {results_file}")
            print(f"Model saved to: {model_file}")
    
    return test_metrics, results_df, history


def run_hyperparameter_sweep(data, device, output_dir):
    #un hyperparameter sweep over predefined configurations
    
    # define sweep configurations
    sweep_configs = [
        # baseline
        {'hidden_dim': 64, 'num_heads': 2, 'dropout': 0.2, 'learning_rate': 0.001},
        
        # vary hidden_dim
        {'hidden_dim': 128, 'num_heads': 2, 'dropout': 0.2, 'learning_rate': 0.001},
        {'hidden_dim': 96, 'num_heads': 2, 'dropout': 0.2, 'learning_rate': 0.001},
        
        # vary num_heads
        {'hidden_dim': 64, 'num_heads': 4, 'dropout': 0.2, 'learning_rate': 0.001},
        {'hidden_dim': 128, 'num_heads': 4, 'dropout': 0.2, 'learning_rate': 0.001},
        
        # vary dropout
        {'hidden_dim': 64, 'num_heads': 2, 'dropout': 0.3, 'learning_rate': 0.001},
        {'hidden_dim': 64, 'num_heads': 2, 'dropout': 0.15, 'learning_rate': 0.001},
        {'hidden_dim': 128, 'num_heads': 4, 'dropout': 0.3, 'learning_rate': 0.001},
        
        # vary learning rate
        {'hidden_dim': 64, 'num_heads': 2, 'dropout': 0.2, 'learning_rate': 0.0005},
        {'hidden_dim': 128, 'num_heads': 4, 'dropout': 0.2, 'learning_rate': 0.0005},
    ]
    
    # fixed params
    fixed_params = {
        'batch_size': 64,
        'num_epochs': 100,
        'weight_decay': 1e-4,
        'patience': 15,
        'accumulation_steps': 1}
    
    results_list = []
    
    print("\n" + "="*70)
    print("HYPERPARAMETER SWEEP")
    print("="*70)
    print(f"Total configurations: {len(sweep_configs)}")
    
    for i, sweep_config in enumerate(sweep_configs, 1):
        config = {**fixed_params, **sweep_config}
        
        print(f"\n{'='*70}")
        print(f"Configuration {i}/{len(sweep_configs)}")
        print(f"  hidden_dim: {config['hidden_dim']}")
        print(f"  num_heads: {config['num_heads']}")
        print(f"  dropout: {config['dropout']}")
        print(f"  learning_rate: {config['learning_rate']}")
        print("="*70)
        
        # create dataloaders
        dataloaders = create_dataloaders(data, batch_size=config['batch_size'])
        
        # create config-specific output directory
        config_str = f"h{config['hidden_dim']}_n{config['num_heads']}_d{int(config['dropout']*100)}_lr{config['learning_rate']}"
        config_dir = output_dir / 'sweep_configs' / config_str
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # train (save everything for each config)
        test_metrics, results_df, history = train_single_config(
            config, data, dataloaders, device, config_dir,
            save_model=True, verbose=True)
        
        # also save training history as JSON for plotting
        history_file = config_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # record results
        result = {
            'config_id': i,
            'hidden_dim': config['hidden_dim'],
            'num_heads': config['num_heads'],
            'dropout': config['dropout'],
            'learning_rate': config['learning_rate'],
            'test_accuracy': test_metrics['accuracy'],
            'test_auc': test_metrics['auc'],
            'test_brier': test_metrics['brier_score'],
            'test_log_loss': test_metrics['log_loss'],
            'best_val_auc': test_metrics['best_val_auc'],
            'epochs_trained': len(history['train_loss']),
            'final_train_acc': history['train_acc'][-1],
            'final_train_auc': history['train_auc'][-1],
            'num_params': count_parameters(TennisMatchupGAT(
                hidden_dim=config['hidden_dim'],
                num_heads=config['num_heads'],
                dropout=config['dropout']
            )),
            'config_dir': str(config_dir)
        }
        results_list.append(result)
        
        # print intermediate results
        print(f"\nResults: Acc={test_metrics['accuracy']:.4f}, "
              f"AUC={test_metrics['auc']:.4f}, "
              f"Brier={test_metrics['brier_score']:.4f}")
    
    # create summary DF
    sweep_results = pd.DataFrame(results_list)
    
    # save sweep results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_file = output_dir / f'gat_v4_sweep_results_{timestamp}.csv'
    sweep_results.to_csv(sweep_file, index=False)
    
    # print summary
    print("\n" + "="*70)
    print("SWEEP SUMMARY")
    print("="*70)
    print(sweep_results.to_string())
    
    # find best config
    best_idx = sweep_results['test_accuracy'].idxmax()
    best_config = sweep_results.iloc[best_idx]
    
    print(f"\nBest Configuration (by accuracy):")
    print(f"  hidden_dim: {best_config['hidden_dim']}")
    print(f"  num_heads: {best_config['num_heads']}")
    print(f"  dropout: {best_config['dropout']}")
    print(f"  learning_rate: {best_config['learning_rate']}")
    print(f"  Test Accuracy: {best_config['test_accuracy']:.4f}")
    print(f"  Test AUC: {best_config['test_auc']:.4f}")
    print(f"  Test Brier: {best_config['test_brier']:.4f}")
    
    print(f"\nSweep results saved to: {sweep_file}")
    
    return sweep_results


def main():
    parser = argparse.ArgumentParser(description='Train Tennis GAT V4 Model')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TENNIS MATCHUP GAT TRAINING - V4")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # load data
    data = load_data()
    
    # output directory
    output_dir = Path('gat_outputs')
    output_dir.mkdir(exist_ok=True)
    
    if args.sweep:
        # run hyperparameter sweep
        sweep_results = run_hyperparameter_sweep(data, device, output_dir)
    else:
        # single configuration run
        config = {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': 1e-4,
            'patience': args.patience,
            'hidden_dim': args.hidden_dim,
            'num_heads': args.num_heads,
            'dropout': args.dropout,
            'accumulation_steps': 1}
        
        print("\nConfiguration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        dataloaders = create_dataloaders(data, batch_size=config['batch_size'])
        
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        
        test_metrics, results_df, history = train_single_config(
            config, data, dataloaders, device, output_dir,
            save_model=True, verbose=True)
        
        # print final results
        print("\n" + "="*70)
        print("TEST EVALUATION")
        print("="*70)
        
        print(f"\nOverall Test Results:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        print(f"  AUC:       {test_metrics['auc']:.4f}")
        print(f"  Log Loss:  {test_metrics['log_loss']:.4f}")
        print(f"  Brier:     {test_metrics['brier_score']:.4f}")
        
        # surface breakdown
        surface_metrics_df = calculate_surface_metrics(results_df)
        print(f"\nAccuracy by Surface:")
        for _, row in surface_metrics_df.iterrows():
            print(f"  {row['surface']:6s}: {row['accuracy']:.3f} (n={row['num_matches']:4d}, AUC={row['auc']:.3f})")
        
        # round breakdown
        round_metrics_df = calculate_round_metrics(results_df)
        print(f"\nAccuracy by Round:")
        for _, row in round_metrics_df.iterrows():
            print(f"  {row['round']:4s}: {row['accuracy']:.3f} (n={row['num_matches']:3d})")
        
        # tournament metrics
        tournament_metrics_df = calculate_tournament_metrics(results_df)
        print(f"\nTournament-level stats:")
        print(f"  Mean accuracy: {tournament_metrics_df['accuracy'].mean():.4f}")
        print(f"  Mean Brier:    {tournament_metrics_df['brier_score'].mean():.4f}")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()