from __future__ import annotations

import torch


def visual_contrastive_loss(
    image_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    temperature: float,
    top_k: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    positive_logits = image_embeddings @ positive_embeddings.t()
    negative_logits = image_embeddings @ negative_embeddings.t()

    all_logits = torch.cat([positive_logits, negative_logits], dim=1) / temperature
    log_probs = torch.log_softmax(all_logits, dim=1)

    pos_log_probs = log_probs[:, : positive_embeddings.shape[0]]
    neg_log_probs = log_probs[:, positive_embeddings.shape[0] :]

    k = max(1, min(top_k, positive_embeddings.shape[0]))
    topk_positive = torch.topk(pos_log_probs, k=k, dim=1).values

    loss = -topk_positive.mean() + neg_log_probs.mean()
    metrics = {
        "positive_logprob_mean": float(pos_log_probs.mean().detach().cpu()),
        "negative_logprob_mean": float(neg_log_probs.mean().detach().cpu()),
        "topk_positive_logprob_mean": float(topk_positive.mean().detach().cpu()),
    }
    return loss, metrics
