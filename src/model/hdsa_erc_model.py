from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


@dataclass
class HDSAConfig:
    bert_path: str
    num_classes: int
    num_subanchors: int = 3
    anchor_dim: int = 256
    dropout: float = 0.1
    anchor_temperature: float = 0.1
    anchor_logit_weight: float = 0.3
    domain_anchor_path: str | None = None
    local_files_only: bool = False


class HDSAERCModel(nn.Module):
    def __init__(self, config: HDSAConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(
            config.bert_path,
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
        anchors = self._load_or_init_anchors(config)
        self.register_buffer("domain_anchors", F.normalize(anchors.float(), dim=-1))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mask_pos: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        h = hidden[batch_idx, mask_pos]
        z = F.normalize(self.map_function(h), dim=-1)
        return {"h": h, "z": z}

    def get_anchors(self) -> torch.Tensor:
        return F.normalize(self.domain_anchors, dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mask_pos: torch.Tensor) -> dict[str, torch.Tensor]:
        enc = self.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos)
        z = enc["z"]
        logits_cls = self.classifier(z)
        anchors = self.get_anchors()
        scores = torch.einsum("bd,cmd->bcm", z, anchors)
        logits_anchor = torch.logsumexp(scores / self.config.anchor_temperature, dim=-1)
        logits_anchor = logits_anchor * self.config.anchor_temperature
        logits = (1.0 - self.config.anchor_logit_weight) * logits_cls + self.config.anchor_logit_weight * logits_anchor
        return {
            "logits": logits,
            "logits_cls": logits_cls,
            "logits_anchor": logits_anchor,
            "z": z,
            "anchors": anchors,
            "scores": scores,
        }

    @torch.no_grad()
    def ema_update_anchors(
        self,
        reps: torch.Tensor,
        labels: torch.Tensor,
        soft_targets: torch.Tensor,
        momentum: float = 0.9,
    ) -> None:
        cnum, mnum, _ = self.domain_anchors.shape
        reps = F.normalize(reps.detach(), dim=-1)
        target = soft_targets.detach().reshape(-1, cnum, mnum)
        for cls in range(cnum):
            idx = torch.where(labels == cls)[0]
            if idx.numel() == 0:
                continue
            z_cls = reps[idx]
            gamma_cls = target[idx, cls]
            for sub in range(mnum):
                weight = gamma_cls[:, sub]
                mass = weight.sum()
                if mass.item() <= 1e-6:
                    continue
                mean = F.normalize((weight[:, None] * z_cls).sum(dim=0) / mass, dim=-1)
                old = self.domain_anchors[cls, sub]
                new = momentum * old + (1.0 - momentum) * mean
                self.domain_anchors[cls, sub] = F.normalize(new, dim=-1)

    def _load_or_init_anchors(self, config: HDSAConfig) -> torch.Tensor:
        if config.domain_anchor_path:
            path = Path(config.domain_anchor_path)
            if not path.exists():
                raise FileNotFoundError(f"domain_anchor_path does not exist: {path}")
            obj = torch.load(path, map_location="cpu")
            anchors = obj["anchors"] if isinstance(obj, dict) else obj
            expected = (config.num_classes, config.num_subanchors, config.anchor_dim)
            if tuple(anchors.shape) != expected:
                raise ValueError(f"Anchor shape {tuple(anchors.shape)} != expected {expected}")
            return anchors
        anchors = torch.randn(config.num_classes, config.num_subanchors, config.anchor_dim)
        return F.normalize(anchors, dim=-1)

