import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, v_dim=768, q_dim=768, out_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(v_dim + q_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, out_dim)
        )

    def forward(self, sv, q_emb):
        """
        sv: [B, T, D]
        q_emb: [B, D]
        """
        # Pool visual tokens
        sv = sv.mean(dim=1)          # [B, D]

        # Concatenate vision + query
        x = torch.cat([sv, q_emb], dim=1)  # [B, 2D]

        # Normalize (important for JEPA-style embedding space)
        x = F.normalize(x, dim=1)

        return self.net(x)