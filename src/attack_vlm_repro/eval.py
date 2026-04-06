from __future__ import annotations

import csv
from pathlib import Path

import torch


def compute_proxy_margin(
    image_embedding: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    top_k: int,
) -> float:
    pos_scores = image_embedding @ positive_embeddings.t()
    neg_scores = image_embedding @ negative_embeddings.t()
    k = max(1, min(top_k, positive_embeddings.shape[0]))
    pos_margin = torch.topk(pos_scores, k=k, dim=1).values.mean()
    neg_margin = neg_scores.mean()
    return float((pos_margin - neg_margin).detach().cpu())


def evaluate_proxy(
    clean_embedding: torch.Tensor,
    adversarial_embedding: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    top_k: int,
    success_margin_threshold: float,
) -> dict:
    clean_margin = compute_proxy_margin(clean_embedding, positive_embeddings, negative_embeddings, top_k)
    adversarial_margin = compute_proxy_margin(adversarial_embedding, positive_embeddings, negative_embeddings, top_k)
    return {
        "clean_margin": clean_margin,
        "adversarial_margin": adversarial_margin,
        "margin_gain": adversarial_margin - clean_margin,
        "proxy_success": adversarial_margin > success_margin_threshold and adversarial_margin > clean_margin,
    }


def summarize_results(results: list[dict]) -> dict:
    num_items = len(results)
    proxy_success_rate = float(sum(1 for item in results if item["proxy_eval"]["proxy_success"]) / max(1, num_items))
    avg_margin_gain = float(sum(item["proxy_eval"]["margin_gain"] for item in results) / max(1, num_items))
    summary = {
        "num_items": num_items,
        "proxy_success_rate": proxy_success_rate,
        "average_margin_gain": avg_margin_gain,
    }
    caption_items = [item for item in results if item.get("caption_eval") is not None]
    if caption_items:
        summary["caption_success_rate"] = float(
            sum(1 for item in caption_items if item["caption_eval"]["caption_success"]) / len(caption_items)
        )
    vqa_items = [item for item in results if item.get("vqa_eval") is not None]
    if vqa_items:
        summary["vqa_success_rate"] = float(
            sum(1 for item in vqa_items if item["vqa_eval"]["vqa_success"]) / len(vqa_items)
        )
    return summary


def write_item_csv(results: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "source_label",
        "target_label",
        "proxy_success",
        "clean_margin",
        "adversarial_margin",
        "margin_gain",
        "caption_success",
        "clean_caption",
        "adversarial_caption",
        "vqa_success",
        "question",
        "clean_answer",
        "adversarial_answer",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            caption_eval = item.get("caption_eval") or {}
            vqa_eval = item.get("vqa_eval") or {}
            writer.writerow(
                {
                    "item_id": item["item_id"],
                    "source_label": item["source_label"],
                    "target_label": item["target_label"],
                    "proxy_success": item["proxy_eval"]["proxy_success"],
                    "clean_margin": item["proxy_eval"]["clean_margin"],
                    "adversarial_margin": item["proxy_eval"]["adversarial_margin"],
                    "margin_gain": item["proxy_eval"]["margin_gain"],
                    "caption_success": caption_eval.get("caption_success"),
                    "clean_caption": caption_eval.get("clean_caption", ""),
                    "adversarial_caption": caption_eval.get("adversarial_caption", ""),
                    "vqa_success": vqa_eval.get("vqa_success"),
                    "question": vqa_eval.get("question", item.get("question", "")),
                    "clean_answer": vqa_eval.get("clean_answer", ""),
                    "adversarial_answer": vqa_eval.get("adversarial_answer", ""),
                }
            )
