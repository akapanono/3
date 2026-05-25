from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


@dataclass
class CHDSConfig:
    model_name_or_path: str
    num_classes: int
    num_subanchors: int = 3
    anchor_dim: int = 256
    dropout: float = 0.1
    temperature: float = 0.1
    anchor_logit_weight: float = 0.5
    local_files_only: bool = False


class CHDSERCModel(nn.Module):
    def __init__(self, config: CHDSConfig, anchors: torch.Tensor | None = None):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(
            config.model_name_or_path,
            local_files_only=config.local_files_only,
        )
        hidden_dim = self.encoder.config.hidden_size
        self.map_function = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.anchor_dim),
        )
        self.classifier = nn.Linear(config.anchor_dim, config.num_classes)

        if anchors is None:
            anchors = torch.randn(
                config.num_classes,
                config.num_subanchors,
                config.anchor_dim,
            )
            anchors = F.normalize(anchors, dim=-1)
        self.anchors = nn.Parameter(anchors.float(), requires_grad=True)

    def set_anchors(self, anchors: torch.Tensor) -> None:
        expected = (
            self.config.num_classes,
            self.config.num_subanchors,
            self.config.anchor_dim,
        )
        if tuple(anchors.shape) != expected:
            raise ValueError(f"Anchor shape {tuple(anchors.shape)} != expected {expected}")
        with torch.no_grad():
            self.anchors.copy_(F.normalize(anchors.to(self.anchors.device), dim=-1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        h = hidden[batch_idx, mask_pos]

        z = self.map_function(h)
        z = F.normalize(z, dim=-1)
        anchors = F.normalize(self.anchors, dim=-1)

        logits_cls = self.classifier(z)
        scores = torch.einsum("bd,ckd->bck", z, anchors)
        logits_anchor = torch.logsumexp(scores / self.config.temperature, dim=-1)
        logits_anchor = logits_anchor * self.config.temperature
        logits = (
            (1.0 - self.config.anchor_logit_weight) * logits_cls
            + self.config.anchor_logit_weight * logits_anchor
        )

        return {
            "logits": logits,
            "logits_cls": logits_cls,
            "logits_anchor": logits_anchor,
            "z": z,
            "anchors": anchors,
            "scores": scores,
        }

