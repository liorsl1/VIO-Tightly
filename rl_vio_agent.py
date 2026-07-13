"""
RL Agent for adaptive VIO parameter tuning.

Architecture (following UZH "RL Meets Visual Odometry"):
  - Variable Encoder: Perceiver-style cross-attention to project variable-length
    keypoint observations to a fixed-size latent representation.
  - Map Stats MLP: processes fixed-size map statistics.
  - Policy head: multi-discrete action distribution (one categorical per tunable).
  - Value head: scalar state value for PPO critic.
"""

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from rl_vio_env import MAP_STATS_DIM, MAX_KEYPOINTS_FOR_OBS, KEYPOINT_FEAT_DIM


class VariableEncoder(nn.Module):
    """Perceiver-style cross-attention to compress N keypoints → M latent tokens.

    Uses learned query tokens that attend to keypoint features (key/value).
    Output shape: (batch, M * D) flattened.
    """

    def __init__(
        self,
        kp_dim: int = KEYPOINT_FEAT_DIM,
        n_tokens: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        max_keypoints: int = MAX_KEYPOINTS_FOR_OBS,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.max_keypoints = max_keypoints
        self.kp_dim = kp_dim

        # Project keypoint features to d_model
        self.kp_proj = nn.Linear(kp_dim, d_model)

        # Learned query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        # Cross-attention: queries = learned tokens, keys/values = keypoints
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, kp_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kp_flat: (batch, max_keypoints * kp_dim) flattened keypoint features.

        Returns:
            (batch, n_tokens * d_model) flattened latent representation.
        """
        B = kp_flat.shape[0]
        # Reshape to (batch, max_keypoints, kp_dim)
        kp = kp_flat.view(B, self.max_keypoints, self.kp_dim)

        # Create mask: keypoints with all zeros are padding
        mask = (kp.abs().sum(dim=-1) == 0)  # (B, max_keypoints), True = pad

        # Project keypoints
        kv = self.kp_proj(kp)  # (B, N, D)

        # Expand learned queries for batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, M, D)

        # Cross-attention (queries attend to keypoints)
        attn_out, _ = self.cross_attn(
            queries, kv, kv, key_padding_mask=mask
        )
        attn_out = self.norm(attn_out + queries)  # residual + norm

        # Flatten tokens
        return attn_out.reshape(B, self.n_tokens * self.d_model)


class VIOFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for SB3: Variable Encoder + Map Stats MLP.

    Splits the flat observation into:
      - map_stats: first MAP_STATS_DIM elements
      - keypoints: remaining elements (MAX_KEYPOINTS_FOR_OBS * KEYPOINT_FEAT_DIM)

    Processes them separately and concatenates.
    """

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.map_stats_dim = MAP_STATS_DIM
        self.kp_flat_dim = MAX_KEYPOINTS_FOR_OBS * KEYPOINT_FEAT_DIM

        # Variable Encoder for keypoints
        n_tokens = 8
        d_model = 64
        self.variable_encoder = VariableEncoder(
            kp_dim=KEYPOINT_FEAT_DIM,
            n_tokens=n_tokens,
            d_model=d_model,
            max_keypoints=MAX_KEYPOINTS_FOR_OBS,
        )
        ve_out_dim = n_tokens * d_model  # 8 * 64 = 512

        # Map stats MLP
        self.map_mlp = nn.Sequential(
            nn.Linear(MAP_STATS_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Combine
        combined_dim = ve_out_dim + 64  # 512 + 64 = 576
        self.combine = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observation
        map_stats = observations[:, :self.map_stats_dim]
        kp_flat = observations[:, self.map_stats_dim:]

        # Process each branch
        kp_features = self.variable_encoder(kp_flat)
        map_features = self.map_mlp(map_stats)

        # Combine
        combined = torch.cat([kp_features, map_features], dim=-1)
        return self.combine(combined)


def make_vio_policy_kwargs(features_dim: int = 128) -> dict:
    """Build policy_kwargs dict for SB3 PPO with our custom extractor."""
    return {
        "features_extractor_class": VIOFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        "activation_fn": nn.ReLU,
    }
