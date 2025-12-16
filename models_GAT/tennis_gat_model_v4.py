#!/usr/bin/env python3

# Tennis Matchup GAT Model V4
"""
Enhanced model with GRANULAR serve/return matchup modeling and style factor attention

Key Changes from V3:
1. Separate attention for First Serve, Second Serve, and Break Point matchups
2. Style factor attention models how P1's style interacts with P2's style
3. More tennis-realistic feature groupings
4. Residual connections in fusion layers
5. All hyperparameters (hidden_dim, num_heads, dropout) configurable throughout

Matchup Philosophy:
- First Serve: P1's first serve effectiveness vs P2's first return ability
- Second Serve: P1's second serve vs P2's second return (pressure situations)
- Break Points: P1's ability to save BPs vs P2's ability to convert (clutch)
- Style Factors: how playing styles interact (from NMF decomposition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# FEATURE INDICES (from feature_index_dict_v2.py)
# =============================================================================
# Total: 270 features
# P1: indices 0-115 (116 features)
# P2: indices 116-231 (116 features)
# Diff: indices 232-269 (38 features)

class FeatureIndices:
    #centralized feature index management for V2 data with GRANULAR groupings
    
    # P1 indices (P2 = P1 + 116)
    P1_OFFSET = 0
    P2_OFFSET = 116
    
    # basic features
    BASIC = [0, 1, 2]  # is_righthanded, age, height
    
    # 52-week stats (31 features: indices 3-33)
    STATS_52W_ALL = list(range(3, 34))
    
    # =========================================================================
    # GRANULAR SERVE/RETURN GROUPINGS -> 52w
    # =========================================================================
    
    # first serve attack
    STATS_52W_FIRST_SERVE = [
        8, # first_serve_pct (% of first serves in)
        9, # first_serve_won_pct (% points won on first serve)
        6, # ace_pct (free points)
    ]
    
    # first serve defense (returning first serves)
    STATS_52W_FIRST_RETURN = [
        27, # first_return_won_pct
    ]
    
    # second serve attack 
    STATS_52W_SECOND_SERVE = [
        10, # second_serve_won_pct
        7,  # df_pct (risk indicator - negative)
    ]
    
    # second serve defense (returning second serves: pressure)
    STATS_52W_SECOND_RETURN = [
        28, # second_return_won_pct
    ]
    
    # break point saving (clutch serving under pressure)
    STATS_52W_BP_SAVE = [
        12, # bp_saved_pct
        14, # bp_faced_per_game (volume for context)
        26, # service_games_held_pct
        32, # service_game_efficiency
    ]
    
    # break point converting (clutch returning)
    STATS_52W_BP_CONVERT = [
        11, # bp_converted_pct
        13, # bp_created_per_return_game
        25, # return_games_broken_pct
        24, # return_game_impact
    ]
    
    # overall serve stats (for general encoding)
    STATS_52W_SERVE = [
        4, # serve_points_won_pct
        6, # ace_pct
        7, # df_pct
        8,  # first_serve_pct
        9, # first_serve_won_pct
        10, # second_serve_won_pct
        12, # bp_saved_pct
        14, # bp_faced_per_game
        26, # service_games_held_pct
        32, # service_game_efficiency
    ]
    
    # overall return stats
    STATS_52W_RETURN = [
        5, # return_points_won_pct
        11, # bp_converted_pct
        13, # bp_created_per_return_game
        24, # return_game_impact
        25, # return_games_broken_pct
        27, # first_return_won_pct
        28, # second_return_won_pct
    ]
    
    # overall performance 52w stats
    STATS_52W_OVERALL = [
        3, # win_pct
        15, # games_ratio
        16, # total_points_won_pct
        17, # close_match_pct
        18, # deciding_set_pct
        19, # tiebreak_pct
        20, # upset_rate
        21, # upset_avg_magnitude
        22, # efficiency_ratio
        23, # defend_rate
        29, # losses
        30, # matches_played
        31, # wins
        33, # straight_sets_pct
    ]
    
    # surface-specific stats (indices 34-36)
    SURFACE_STATS = [34, 35, 36]  # surface_win_pct, surface_serve_pct, surface_return_pct
    
    # =========================================================================
    # GRANULAR SERVE/RETURN GROUPINGS -> 4w (recent form)
    # =========================================================================
    
    STATS_4W_ALL = list(range(37, 64))
    
    STATS_4W_FIRST_SERVE = [
        45, # first_serve_pct
        46, # first_serve_won_pct
        37, # ace_pct
    ]
    
    STATS_4W_FIRST_RETURN = [
        44, # first_return_won_pct
    ]
    
    STATS_4W_SECOND_SERVE = [
        55, # second_serve_won_pct
        42, # df_pct
    ]
    
    STATS_4W_SECOND_RETURN = [
        54, # second_return_won_pct
    ]
    
    STATS_4W_BP_SAVE = [
        41, # bp_saved_pct
        40, # bp_faced_per_game
        58, # service_games_held_pct
        57, # service_game_efficiency
    ]
    
    STATS_4W_BP_CONVERT = [
        38, # bp_converted_pct
        39, # bp_created_per_return_game
        52, # return_games_broken_pct
        51, # return_game_impact
    ]
    
    STATS_4W_SERVE = [
        56, # serve_points_won_pct
        37, # ace_pct
        42, # df_pct
        45, # first_serve_pct
        46, # first_serve_won_pct
        55, # second_serve_won_pct
        41, # bp_saved_pct
        40, # bp_faced_per_game
        58, # service_games_held_pct
        57, # service_game_efficiency
    ]
    
    STATS_4W_RETURN = [
        53, # return_points_won_pct
        38, # bp_converted_pct
        39, # bp_created_per_return_game
        51, # return_game_impact
        52, # return_games_broken_pct
        44, # first_return_won_pct
        54,   second_return_won_pct
    ]
    
    # =========================================================================
    # GRANULAR SERVE/RETURN GROUPINGS -> 8w
    # =========================================================================
    
    STATS_8W_ALL = list(range(64, 91))
    
    STATS_8W_FIRST_SERVE = [
        72, # first_serve_pct
        73, # first_serve_won_pct
        64, # ace_pct
    ]
    
    STATS_8W_FIRST_RETURN = [
        71, # first_return_won_pct
    ]
    
    STATS_8W_SECOND_SERVE = [
        82, # second_serve_won_pct
        69, # df_pct
    ]
    
    STATS_8W_SECOND_RETURN = [
        81, # second_return_won_pct
    ]
    
    STATS_8W_BP_SAVE = [
        68, # bp_saved_pct
        67, # bp_faced_per_game
        85, # service_games_held_pct
        84, # service_game_efficiency
    ]
    
    STATS_8W_BP_CONVERT = [
        65, # bp_converted_pct
        66, # bp_created_per_return_game
        79, # return_games_broken_pct
        78, # return_game_impact
    ]
    
    STATS_8W_SERVE = [
        83, # serve_points_won_pct
        64, # ace_pct
        69, # df_pct
        72, # first_serve_pct
        73,  # first_serve_won_pct
        82, # second_serve_won_pct
        68,  # bp_saved_pct
        67,  # bp_faced_per_game
        85,  # service_games_held_pct
        84,  # service_game_efficiency
    ]
    
    STATS_8W_RETURN = [
        80,  # return_points_won_pct
        65,  # bp_converted_pct
        66,  # bp_created_per_return_game
        78,  # return_game_impact
        79,  # return_games_broken_pct
        71,  # first_return_won_pct
        81,  # second_return_won_pct
    ]
    
    # form deltas (indices 91-92)
    FORM_DELTAS = [91, 92]  # form_delta_4w, form_delta_8w
    
    # rankings (indices 93-94)
    RANKINGS = [93, 94]  # rank, ranking_points
    
    # relative to average (indices 95-99)
    RELATIVE_AVG = [95, 96, 97, 98, 99]
    
    # ratings (indices 100-105)
    ELO = [100, 101]  # elo_overall, elo_surface
    GLICKO = [102, 103, 104, 105]  # glicko_rating, glicko_rd, glicko_surface, glicko_surface_rd
    RATINGS_ALL = [100, 101, 102, 103, 104, 105]
    
    # style factors (indices 106-115)
    STYLE_FACTORS = list(range(106, 116))  # 5 elo factors + 5 glicko factors (interleaved)
    ELO_FACTORS = [106, 108, 110, 112, 114]  # indices 106, 108, 110, 112, 114
    GLICKO_FACTORS = [107, 109, 111, 113, 115]  # indices 107, 109, 111, 113, 115
    
    # differential features (indices 232-269)
    DIFF_START = 232
    DIFF_RATINGS = [232, 233, 234, 235]  # diff_rank, diff_elo_overall, diff_elo_surface, diff_glicko_rating
    DIFF_52W = [236, 237, 238, 239]  # diff_stats_52w: win_pct, serve, return, total
    DIFF_4W = list(range(240, 255))  # 15 diff features for 4w
    DIFF_8W = list(range(255, 270))  # 15 diff features for 8w
    DIFF_ALL = list(range(232, 270))  # All 38 diff features


# =============================================================================
# GRANULAR MATCHUP ENCODERS
# =============================================================================

class FirstServeEncoder(nn.Module):
    # encodes 1st serve attack and 1st return defense
    #1st serve is about free points (aces) and forcing weak returns
    def __init__(self, hidden_dim=64, dropout=0.2):
        super().__init__()
        
        # rirst serve attack: 3 features x 3 time windows = 9
        self.serve_encoder = nn.Sequential(
            nn.Linear(9, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4))
        
        # first return defense: 1 feature x 3 time windows = 3
        self.return_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 4))
        
    def forward(self, serve_features, return_features):
        return self.serve_encoder(serve_features), self.return_encoder(return_features)


class SecondServeEncoder(nn.Module):
    # encodes second serve attack + second return defense
    def __init__(self, hidden_dim=64, dropout=0.2):
        super().__init__()
        
        # second serve attack: 2 features x 3 time windows = 6
        self.serve_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 4))
        
        # second return defense: 1 feature x 3 time windows = 3
        self.return_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 4))
        
    def forward(self, serve_features, return_features):
        return self.serve_encoder(serve_features), self.return_encoder(return_features)


class BreakPointEncoder(nn.Module):
    # encodes break point situations
    # BP saving vs BP converting 
    def __init__(self, hidden_dim=64, dropout=0.2):
        super().__init__()
        
        # BP save: 4 features x 3 time windows = 12
        self.save_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4))
        
        # BP convert: 4 features x 3 time windows = 12
        self.convert_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4))
        
    def forward(self, save_features, convert_features):
        return self.save_encoder(save_features), self.convert_encoder(convert_features)


# =============================================================================
# ATTENTION MODULES
# =============================================================================

class GranularCrossAttention(nn.Module):
    # cross-attention for specific matchup types
    # learns how one player's skill interacts w/ opponents counter-skills
    def __init__(self, hidden_dim=16, num_heads=2, dropout=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = max(hidden_dim // num_heads, 1)
        self.scale = self.head_dim ** -0.5
        
        self.attack_to_q = nn.Linear(hidden_dim, hidden_dim)
        self.defense_to_k = nn.Linear(hidden_dim, hidden_dim)
        self.defense_to_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, attack_enc, defense_enc):
        batch_size = attack_enc.size(0)
        hidden_dim = attack_enc.size(1)
        
        Q = self.attack_to_q(attack_enc).view(batch_size, self.num_heads, self.head_dim)
        K = self.defense_to_k(defense_enc).view(batch_size, self.num_heads, self.head_dim)
        V = self.defense_to_v(defense_enc).view(batch_size, self.num_heads, self.head_dim)
        
        attn_scores = torch.sum(Q * K, dim=-1, keepdim=True) * self.scale
        attn_weights = torch.sigmoid(attn_scores)
        attn_weights = self.dropout(attn_weights)
        
        context = attn_weights * V
        context = context.view(batch_size, -1)
        
        if context.size(1) < hidden_dim:
            padding = torch.zeros(batch_size, hidden_dim - context.size(1), device=context.device)
            context = torch.cat([context, padding], dim=1)
        
        output = self.output_proj(context)
        output = self.layer_norm(output + attack_enc)
        
        return output


class StyleFactorAttention(nn.Module):
    # models how P1s playing style interacts w/ P2's playing style
    # uses multihead attention to capture diff. aspects of style matchups
    # since Glicko+NMF showed strong results, this gives style factors dedicated attention rather than just bilinear interaction
    def __init__(self, hidden_dim=32, num_heads=2, dropout=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = max(hidden_dim // num_heads, 1)
        self.scale = self.head_dim ** -0.5
        
        # separate encoders for ELO and Glicko style factors (5 each per player)
        self.elo_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        self.glicko_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        # cross-attention: P1's style queries P2's style
        self.p1_to_q = nn.Linear(hidden_dim, hidden_dim)
        self.p2_to_k = nn.Linear(hidden_dim, hidden_dim)
        self.p2_to_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # bilinear interaction (direct factor-to-factor)
        self.bilinear_elo = nn.Bilinear(5, 5, hidden_dim // 2)
        self.bilinear_glicko = nn.Bilinear(5, 5, hidden_dim // 2)
        
        # final fusion
        # attn_output: hidden_dim, bilinear_combined: hidden_dim (hidden_dim//2 * 2)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        
    def forward(self, p1_elo_factors, p1_glicko_factors, p2_elo_factors, p2_glicko_factors):
        batch_size = p1_elo_factors.size(0)
        
        # encode style factors
        p1_elo_enc = self.elo_encoder(p1_elo_factors)
        p1_glicko_enc = self.glicko_encoder(p1_glicko_factors)
        p2_elo_enc = self.elo_encoder(p2_elo_factors)
        p2_glicko_enc = self.glicko_encoder(p2_glicko_factors)
        
        # combine ELO and Glicko encodings
        p1_combined = p1_elo_enc + p1_glicko_enc
        p2_combined = p2_elo_enc + p2_glicko_enc
        
        # cross-attention
        hidden_dim = p1_combined.size(1)
        Q = self.p1_to_q(p1_combined).view(batch_size, self.num_heads, self.head_dim)
        K = self.p2_to_k(p2_combined).view(batch_size, self.num_heads, self.head_dim)
        V = self.p2_to_v(p2_combined).view(batch_size, self.num_heads, self.head_dim)
        
        attn_scores = torch.sum(Q * K, dim=-1, keepdim=True) * self.scale
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        
        context = attn_weights * V
        context = context.view(batch_size, -1)
        
        if context.size(1) < hidden_dim:
            padding = torch.zeros(batch_size, hidden_dim - context.size(1), device=context.device)
            context = torch.cat([context, padding], dim=1)
        
        attn_output = self.output_proj(context)
        attn_output = self.layer_norm(attn_output + p1_combined)
        
        # bilinear interactions
        elo_interaction = self.bilinear_elo(p1_elo_factors, p2_elo_factors)
        glicko_interaction = self.bilinear_glicko(p1_glicko_factors, p2_glicko_factors)
        bilinear_combined = torch.cat([elo_interaction, glicko_interaction], dim=1)
        
        # fuse attention + bilinear
        combined = torch.cat([attn_output, bilinear_combined], dim=1)
        style_repr = self.fusion(combined)
        
        return style_repr


class MatchupFusion(nn.Module):
    # fuses the three granular matchups into overall serve/return advantage
    def __init__(self, hidden_dim=64, dropout=0.2):
        super().__init__()
        
        input_dim = 3 * (hidden_dim // 4) * 2
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim))
        
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
    def forward(self, matchups):
        residual = self.residual_proj(matchups)
        return self.fusion(matchups) + residual


# =============================================================================
# SUPPORTING MODULES
# =============================================================================

class SurfaceConditioner(nn.Module):
    def __init__(self, hidden_dim=64, num_surfaces=3, dropout=0.1):
        super().__init__()
        self.surface_embedding = nn.Embedding(num_surfaces, hidden_dim)
        self.surface_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Sigmoid())
        
    def forward(self, features, surface_idx):
        surface_emb = self.surface_embedding(surface_idx)
        gate = self.surface_gate(surface_emb)
        return features * gate + features


class RatingModule(nn.Module):
    def __init__(self, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim))
        
    def forward(self, p1_ratings, p2_ratings, diff_ratings):
        combined = torch.cat([p1_ratings, p2_ratings, diff_ratings], dim=1)
        return self.encoder(combined)


class FormModule(nn.Module):
    def __init__(self, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim))
        
    def forward(self, p1_form, p2_form):
        combined = torch.cat([p1_form, p2_form], dim=1)
        return self.encoder(combined)


# =============================================================================
# MAIN MODEL
# =============================================================================

class TennisMatchupGAT(nn.Module):
    #Tennis Matchup GAT V4: w/ granular serve/return modeling + style factor attention
    def __init__(self, hidden_dim=64, num_heads=2, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.idx = FeatureIndices()
        
        # granular encoders (Siamese)
        self.first_serve_encoder = FirstServeEncoder(hidden_dim, dropout)
        self.second_serve_encoder = SecondServeEncoder(hidden_dim, dropout)
        self.break_point_encoder = BreakPointEncoder(hidden_dim, dropout)
        
        # cross-attention for matchups
        matchup_dim = hidden_dim // 4
        self.first_serve_attn_p1 = GranularCrossAttention(matchup_dim, num_heads, dropout)
        self.first_serve_attn_p2 = GranularCrossAttention(matchup_dim, num_heads, dropout)
        self.second_serve_attn_p1 = GranularCrossAttention(matchup_dim, num_heads, dropout)
        self.second_serve_attn_p2 = GranularCrossAttention(matchup_dim, num_heads, dropout)
        self.bp_attn_p1 = GranularCrossAttention(matchup_dim, num_heads, dropout)
        self.bp_attn_p2 = GranularCrossAttention(matchup_dim, num_heads, dropout)
        
        # matchup fusion
        self.matchup_fusion = MatchupFusion(hidden_dim, dropout)
        
        # style factor attention
        self.style_attention = StyleFactorAttention(hidden_dim // 2, num_heads, dropout)
        
        # other modules
        self.surface_conditioner = SurfaceConditioner(hidden_dim, num_surfaces=3, dropout=dropout)
        self.rating_module = RatingModule(hidden_dim // 2, dropout)
        self.form_module = FormModule(hidden_dim // 2, dropout)
        
        self.overall_encoder = nn.Sequential(
            nn.Linear(28, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        self.diff_encoder = nn.Sequential(
            nn.Linear(38, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        # fusion
        fusion_input_dim = hidden_dim + (hidden_dim // 2) * 5 + 6
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.fusion_residual = nn.Linear(fusion_input_dim, hidden_dim)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 0}
    
    def _extract_features(self, x, indices, offset=0):
        return x[:, [i + offset for i in indices]]
    
    def _get_first_serve_features(self, x, offset):
        fs_52w = self._extract_features(x, self.idx.STATS_52W_FIRST_SERVE, offset)
        fs_4w = self._extract_features(x, self.idx.STATS_4W_FIRST_SERVE, offset)
        fs_8w = self._extract_features(x, self.idx.STATS_8W_FIRST_SERVE, offset)
        return torch.cat([fs_52w, fs_4w, fs_8w], dim=1)
    
    def _get_first_return_features(self, x, offset):
        fr_52w = self._extract_features(x, self.idx.STATS_52W_FIRST_RETURN, offset)
        fr_4w = self._extract_features(x, self.idx.STATS_4W_FIRST_RETURN, offset)
        fr_8w = self._extract_features(x, self.idx.STATS_8W_FIRST_RETURN, offset)
        return torch.cat([fr_52w, fr_4w, fr_8w], dim=1)
    
    def _get_second_serve_features(self, x, offset):
        ss_52w = self._extract_features(x, self.idx.STATS_52W_SECOND_SERVE, offset)
        ss_4w = self._extract_features(x, self.idx.STATS_4W_SECOND_SERVE, offset)
        ss_8w = self._extract_features(x, self.idx.STATS_8W_SECOND_SERVE, offset)
        return torch.cat([ss_52w, ss_4w, ss_8w], dim=1)
    
    def _get_second_return_features(self, x, offset):
        sr_52w = self._extract_features(x, self.idx.STATS_52W_SECOND_RETURN, offset)
        sr_4w = self._extract_features(x, self.idx.STATS_4W_SECOND_RETURN, offset)
        sr_8w = self._extract_features(x, self.idx.STATS_8W_SECOND_RETURN, offset)
        return torch.cat([sr_52w, sr_4w, sr_8w], dim=1)
    
    def _get_bp_save_features(self, x, offset):
        bp_52w = self._extract_features(x, self.idx.STATS_52W_BP_SAVE, offset)
        bp_4w = self._extract_features(x, self.idx.STATS_4W_BP_SAVE, offset)
        bp_8w = self._extract_features(x, self.idx.STATS_8W_BP_SAVE, offset)
        return torch.cat([bp_52w, bp_4w, bp_8w], dim=1)
    
    def _get_bp_convert_features(self, x, offset):
        bp_52w = self._extract_features(x, self.idx.STATS_52W_BP_CONVERT, offset)
        bp_4w = self._extract_features(x, self.idx.STATS_4W_BP_CONVERT, offset)
        bp_8w = self._extract_features(x, self.idx.STATS_8W_BP_CONVERT, offset)
        return torch.cat([bp_52w, bp_4w, bp_8w], dim=1)
    
    def _get_form_features(self, x, offset):
        form_deltas = self._extract_features(x, self.idx.FORM_DELTAS, offset)
        form_4w = x[:, [offset + 62, offset + 50, offset + 61]]
        form_8w = x[:, [offset + 89, offset + 77, offset + 88]]
        relative = self._extract_features(x, self.idx.RELATIVE_AVG, offset)
        return torch.cat([form_deltas, form_4w, form_8w, relative[:, :2]], dim=1)
    
    def forward(self, x, surface_names=None):
        batch_size = x.size(0)
        
        # extract granular features
        p1_first_serve = self._get_first_serve_features(x, self.idx.P1_OFFSET)
        p1_first_return = self._get_first_return_features(x, self.idx.P1_OFFSET)
        p2_first_serve = self._get_first_serve_features(x, self.idx.P2_OFFSET)
        p2_first_return = self._get_first_return_features(x, self.idx.P2_OFFSET)
        
        p1_second_serve = self._get_second_serve_features(x, self.idx.P1_OFFSET)
        p1_second_return = self._get_second_return_features(x, self.idx.P1_OFFSET)
        p2_second_serve = self._get_second_serve_features(x, self.idx.P2_OFFSET)
        p2_second_return = self._get_second_return_features(x, self.idx.P2_OFFSET)
        
        p1_bp_save = self._get_bp_save_features(x, self.idx.P1_OFFSET)
        p1_bp_convert = self._get_bp_convert_features(x, self.idx.P1_OFFSET)
        p2_bp_save = self._get_bp_save_features(x, self.idx.P2_OFFSET)
        p2_bp_convert = self._get_bp_convert_features(x, self.idx.P2_OFFSET)
        
        # encode (Siamese)
        p1_fs_enc, p1_fr_enc = self.first_serve_encoder(p1_first_serve, p1_first_return)
        p2_fs_enc, p2_fr_enc = self.first_serve_encoder(p2_first_serve, p2_first_return)
        
        p1_ss_enc, p1_sr_enc = self.second_serve_encoder(p1_second_serve, p1_second_return)
        p2_ss_enc, p2_sr_enc = self.second_serve_encoder(p2_second_serve, p2_second_return)
        
        p1_bps_enc, p1_bpc_enc = self.break_point_encoder(p1_bp_save, p1_bp_convert)
        p2_bps_enc, p2_bpc_enc = self.break_point_encoder(p2_bp_save, p2_bp_convert)
        
        # cross-attention matchups
        p1_fs_matchup = self.first_serve_attn_p1(p1_fs_enc, p2_fr_enc)
        p2_fs_matchup = self.first_serve_attn_p2(p2_fs_enc, p1_fr_enc)
        
        p1_ss_matchup = self.second_serve_attn_p1(p1_ss_enc, p2_sr_enc)
        p2_ss_matchup = self.second_serve_attn_p2(p2_ss_enc, p1_sr_enc)
        
        p1_bp_matchup = self.bp_attn_p1(p1_bps_enc, p2_bpc_enc)
        p2_bp_matchup = self.bp_attn_p2(p2_bps_enc, p1_bpc_enc)
        
        # fuse matchups
        all_matchups = torch.cat([
            p1_fs_matchup, p2_fs_matchup,
            p1_ss_matchup, p2_ss_matchup,
            p1_bp_matchup, p2_bp_matchup
        ], dim=1)
        
        matchup_repr = self.matchup_fusion(all_matchups)
        
        # style factor attention
        p1_elo_factors = self._extract_features(x, self.idx.ELO_FACTORS, self.idx.P1_OFFSET)
        p1_glicko_factors = self._extract_features(x, self.idx.GLICKO_FACTORS, self.idx.P1_OFFSET)
        p2_elo_factors = self._extract_features(x, self.idx.ELO_FACTORS, self.idx.P2_OFFSET)
        p2_glicko_factors = self._extract_features(x, self.idx.GLICKO_FACTORS, self.idx.P2_OFFSET)
        
        style_repr = self.style_attention(p1_elo_factors, p1_glicko_factors, p2_elo_factors, p2_glicko_factors)
        
        # other features
        p1_ratings = torch.cat([
            self._extract_features(x, self.idx.RATINGS_ALL, self.idx.P1_OFFSET),
            self._extract_features(x, self.idx.RANKINGS, self.idx.P1_OFFSET)
        ], dim=1)
        p2_ratings = torch.cat([
            self._extract_features(x, self.idx.RATINGS_ALL, self.idx.P2_OFFSET),
            self._extract_features(x, self.idx.RANKINGS, self.idx.P2_OFFSET)
        ], dim=1)
        diff_ratings = x[:, self.idx.DIFF_RATINGS]
        rating_repr = self.rating_module(p1_ratings, p2_ratings, diff_ratings)
        
        p1_form = self._get_form_features(x, self.idx.P1_OFFSET)
        p2_form = self._get_form_features(x, self.idx.P2_OFFSET)
        form_repr = self.form_module(p1_form, p2_form)
        
        p1_overall = self._extract_features(x, self.idx.STATS_52W_OVERALL, self.idx.P1_OFFSET)
        p2_overall = self._extract_features(x, self.idx.STATS_52W_OVERALL, self.idx.P2_OFFSET)
        overall_repr = self.overall_encoder(torch.cat([p1_overall, p2_overall], dim=1))
        
        diff_features = x[:, self.idx.DIFF_ALL]
        diff_repr = self.diff_encoder(diff_features)
        
        p1_surface = self._extract_features(x, self.idx.SURFACE_STATS, self.idx.P1_OFFSET)
        p2_surface = self._extract_features(x, self.idx.SURFACE_STATS, self.idx.P2_OFFSET)
        surface_stats = torch.cat([p1_surface, p2_surface], dim=1)
        
        # fusion with residual
        combined = torch.cat([
            matchup_repr, style_repr, rating_repr, form_repr,
            overall_repr, diff_repr, surface_stats
        ], dim=1)
        
        fused = self.fusion(combined) + self.fusion_residual(combined)
        
        output = self.output(fused).squeeze(-1)
        
        return output
    
    def get_matchup_analysis(self, x):
        with torch.no_grad(): pred = self.forward(x)
        return {'prediction': pred}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("="*70)
    print("Testing TennisMatchupGAT V4 - Granular Matchups + Style Attention")
    print("="*70)
    
    configs = [
        {'hidden_dim': 64, 'num_heads': 2, 'dropout': 0.2},
        {'hidden_dim': 128, 'num_heads': 4, 'dropout': 0.2},
        {'hidden_dim': 64, 'num_heads': 4, 'dropout': 0.3},]
    
    for config in configs:
        print(f"\nConfig: {config}")
        model = TennisMatchupGAT(**config)
        print(f"  Parameters: {count_parameters(model):,}")
        
        batch_size = 32
        x = torch.randn(batch_size, 270)
        output = model(x)
        print(f"  Input: {x.shape}, Output: {output.shape}, Range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)