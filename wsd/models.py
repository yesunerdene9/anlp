"""Model architectures for WSD classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """projects sentence embeddings (768-dim) to concept space (500-dim)"""
    # 768 = mmbert
    # 4096 = qwen
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=500, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class ConceptClassifier(nn.Module):
    """WSD classifier with MLP projection and concept embedding matching."""
    # 768 = mmbert
    # 4096 = qwen
    def __init__(self, concept_embeddings, input_dim=768, hidden_dim=1024, 
                 output_dim=500, dropout=0.1, temperature=0.1, concept_mask=None):
        super().__init__()
        self.projector = Projector(input_dim, hidden_dim, output_dim, dropout)
        self.concept_embeddings = nn.Parameter(concept_embeddings)
        self.temperature = temperature
        
    
        self.register_buffer('concept_mask', concept_mask)
    
    def forward(self, x):
        projected = self.projector(x)

        logits = torch.matmul(projected, self.concept_embeddings.t()) / self.temperature
        
        if self.concept_mask is not None:
            logits = logits * self.concept_mask
        
        return logits
