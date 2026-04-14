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


def relative_proxy_loss(
    clean_image_embeddings: torch.Tensor,
    adversarial_image_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    clean_positive_logits = clean_image_embeddings @ positive_embeddings.t()
    clean_negative_logits = clean_image_embeddings @ negative_embeddings.t()
    adversarial_positive_logits = adversarial_image_embeddings @ positive_embeddings.t()
    adversarial_negative_logits = adversarial_image_embeddings @ negative_embeddings.t()

    k = max(1, min(top_k, positive_embeddings.shape[0]))
    clean_positive_topk = torch.topk(clean_positive_logits, k=k, dim=1).values
    adversarial_positive_topk = torch.topk(adversarial_positive_logits, k=k, dim=1).values

    positive_gain = adversarial_positive_topk.mean() - clean_positive_topk.mean()
    negative_shift = adversarial_negative_logits.mean() - clean_negative_logits.mean()
    loss = -positive_gain + negative_shift
    metrics = {
        "relative_positive_gain": float(positive_gain.detach().cpu()),
        "relative_negative_shift": float(negative_shift.detach().cpu()),
    }
    return loss, metrics
