import torch
import torch.nn as nn


class PlaneHead(nn.Module):
    """
    Plane head. For each token it predicts the plane's center depth (log metric depth when
    depth_mode is "log"), two radii, and a rotation quaternion [w, x, y, z].
    """

    def __init__(self, dec_dim, hidden_dim, depth_mode="linear"):
        super().__init__()
        self.depth_mode = depth_mode

        if depth_mode == "log":
            depth_activation = nn.Identity()  # log depth can be negative
        else:
            depth_activation = nn.ReLU()  # linear depth must be >= 0

        self.mlp_center_depth = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            depth_activation
        )
        self.mlp_radii = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.ReLU()
        )
        self.mlp_rot = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)
        )

        # Meta tensors are initialized by model.init_weights after materialization.
        if not self.mlp_center_depth[0].weight.is_meta:
            self.reset_parameters()

    def reset_parameters(self):
        """Initialize the depth, radii, and rotation heads.

        Initial metric depth is 2 metres in both depth modes; radii have bias 0.1.
        Rotation starts near the identity quaternion [w, x, y, z] = [1, 0, 0, 0].
        """
        for mlp in [self.mlp_center_depth]:
            nn.init.kaiming_normal_(mlp[0].weight, nonlinearity='relu')
            nn.init.zeros_(mlp[0].bias)
            nn.init.kaiming_normal_(mlp[2].weight, nonlinearity='relu')
            nn.init.zeros_(mlp[2].bias)
            nn.init.kaiming_normal_(mlp[4].weight, nonlinearity='relu')
            nn.init.zeros_(mlp[4].bias)
            nn.init.zeros_(mlp[6].weight)

            import math
            if self.depth_mode == "log":
                initial_depth_bias = math.log(2.0)
            else:
                initial_depth_bias = 2.0
            nn.init.constant_(mlp[6].bias, initial_depth_bias)

        for mlp in [self.mlp_radii]:
            nn.init.kaiming_normal_(mlp[0].weight, nonlinearity='relu')
            nn.init.zeros_(mlp[0].bias)
            nn.init.zeros_(mlp[2].weight)
            nn.init.constant_(mlp[2].bias, 0.1)

        nn.init.kaiming_normal_(self.mlp_rot[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.mlp_rot[0].bias)
        nn.init.kaiming_normal_(self.mlp_rot[2].weight, nonlinearity='relu')
        nn.init.zeros_(self.mlp_rot[2].bias)
        nn.init.kaiming_normal_(self.mlp_rot[4].weight, nonlinearity='relu')
        nn.init.zeros_(self.mlp_rot[4].bias)

        nn.init.normal_(self.mlp_rot[-1].weight, mean=0.0, std=0.001)
        # The rotation starts at the identity quaternion.
        self.mlp_rot[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def forward(self, dec_feat):
        """
        Args:
            dec_feat: [B, N, C] - decoder features per token
        """
        # Ensure input dtype matches module dtype (important when module is FP32 but input is BF16)
        module_dtype = self.mlp_center_depth[0].weight.dtype
        if dec_feat.dtype != module_dtype:
            dec_feat = dec_feat.to(module_dtype)

        center = self.mlp_center_depth(dec_feat)
        radii = self.mlp_radii(dec_feat)
        rot = self.mlp_rot(dec_feat)

        result = {
            'pred_center_depth': center,
            'pred_radii': radii,
            'pred_rot': rot,
        }

        return result
