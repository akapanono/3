from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anchors.anchor_utils import anchor_similarity_stats, build_label_embeddings
from src.model.hdsa_losses import anchor_domain_loss, anchor_inter_loss, anchor_preserve_loss, anchor_rank_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain HDSA domain anchors on the hypersphere.")
    parser.add_argument("--init_anchor_path", required=True)
    parser.add_argument("--output_anchor_path", required=True)
    parser.add_argument("--anchor_pretrain_epochs", type=int, default=1000)
    parser.add_argument("--anchor_pretrain_lr", type=float, default=0.1)
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--center_weight", type=float, default=0.1)
    parser.add_argument("--div_weight", type=float, default=1.0)
    parser.add_argument("--preserve_weight", type=float, default=0.5)
    parser.add_argument("--rank_weight", type=float, default=1.0)
    parser.add_argument("--same_upper", type=float, default=0.90)
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obj = torch.load(args.init_anchor_path, map_location="cpu")
    init_anchors = obj["anchors"] if isinstance(obj, dict) else obj
    id2label = {int(k): v for k, v in obj.get("id2label", {}).items()} if isinstance(obj, dict) else {}
    if not id2label:
        id2label = {idx: str(idx) for idx in range(init_anchors.size(0))}
    label_embeddings = build_label_embeddings(id2label, init_anchors.size(-1))
    anchors = nn.Parameter(F.normalize(init_anchors.clone().float(), dim=-1))
    optimizer = torch.optim.SGD([anchors], lr=args.anchor_pretrain_lr, momentum=0.9)
    for step in range(1, args.anchor_pretrain_epochs + 1):
        norm_anchors = F.normalize(anchors, dim=-1)
        loss_inter = anchor_inter_loss(norm_anchors)
        loss_domain, loss_center, loss_div = anchor_domain_loss(
            norm_anchors,
            same_upper=args.same_upper,
            center_weight=args.center_weight,
            div_weight=args.div_weight,
        )
        loss_rank = anchor_rank_loss(norm_anchors, label_embeddings)
        loss_preserve = anchor_preserve_loss(norm_anchors, init_anchors)
        loss = (
            loss_inter
            + args.domain_weight * loss_domain
            + args.rank_weight * loss_rank
            + args.preserve_weight * loss_preserve
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            anchors.copy_(F.normalize(anchors, dim=-1))
        if step == 1 or step % args.log_every == 0 or step == args.anchor_pretrain_epochs:
            print(
                f"step={step} loss={loss.item():.6f} inter={loss_inter.item():.6f} "
                f"domain={loss_domain.item():.6f} center={loss_center.item():.6f} "
                f"div={loss_div.item():.6f} rank={loss_rank.item():.6f} "
                f"preserve={loss_preserve.item():.6f}"
            )
    final_stats = anchor_similarity_stats(anchors.detach().cpu())
    out = dict(obj) if isinstance(obj, dict) else {}
    out.update(
        {
            "anchors": F.normalize(anchors.detach().cpu(), dim=-1),
            "init_anchors": F.normalize(init_anchors.detach().cpu(), dim=-1),
            "label_embeddings": label_embeddings.cpu(),
            "args": vars(args),
            "source": "kmeans_init_plus_hyperspherical_pretrain",
            "anchor_stats": final_stats,
        }
    )
    output_path = Path(args.output_anchor_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    print("Pretrained anchor stats:")
    for key, value in final_stats.items():
        print(f"pretrain_{key}={value:.6f}")
    print(f"Saved hyperspherical domain anchors to {output_path}")


if __name__ == "__main__":
    main()
