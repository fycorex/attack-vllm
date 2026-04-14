from __future__ import annotations

import csv
from pathlib import Path

import torch


VICTIM_SUMMARY_SPECS = [
    ("caption_eval", "caption_success", "caption_success_rate", "caption_eval_count"),
    ("vqa_eval", "vqa_success", "vqa_success_rate", "vqa_eval_count"),
    ("ocr_eval", "ocr_success", "ocr_success_rate", "ocr_eval_count"),
    ("gpt_eval", "gpt_success", "gpt_success_rate", "gpt_eval_count"),
]


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


def summarize_per_surrogate(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for item in results:
        per_surrogate = item.get("proxy_eval", {}).get("per_surrogate", {})
        for surrogate_name, surrogate_metrics in per_surrogate.items():
            grouped.setdefault(surrogate_name, []).append(surrogate_metrics)

    summary = {}
    for surrogate_name, surrogate_results in grouped.items():
        count = len(surrogate_results)
        summary[surrogate_name] = {
            "num_items": count,
            "proxy_success_rate": float(
                sum(1 for result in surrogate_results if result["proxy_success"]) / max(1, count)
            ),
            "average_clean_margin": float(
                sum(result["clean_margin"] for result in surrogate_results) / max(1, count)
            ),
            "average_adversarial_margin": float(
                sum(result["adversarial_margin"] for result in surrogate_results) / max(1, count)
            ),
            "average_margin_gain": float(
                sum(result["margin_gain"] for result in surrogate_results) / max(1, count)
            ),
        }
    return summary


def summarize_results(results: list[dict]) -> dict:
    num_items = len(results)
    proxy_success_rate = float(sum(1 for item in results if item["proxy_eval"]["proxy_success"]) / max(1, num_items))
    avg_margin_gain = float(sum(item["proxy_eval"]["margin_gain"] for item in results) / max(1, num_items))
    summary = {
        "num_items": num_items,
        "proxy_success_rate": proxy_success_rate,
        "average_margin_gain": avg_margin_gain,
        "per_surrogate_proxy_summary": summarize_per_surrogate(results),
    }
    for eval_key, success_key, rate_key, count_key in VICTIM_SUMMARY_SPECS:
        eval_items = [item for item in results if item.get(eval_key) is not None]
        summary[count_key] = len(eval_items)
        if eval_items:
            summary[rate_key] = float(sum(1 for item in eval_items if item[eval_key][success_key]) / len(eval_items))
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
        "source_answer_text",
        "target_answer_text",
        "clean_answer",
        "adversarial_answer",
        "ocr_success",
        "clean_text",
        "adversarial_text",
        "gpt_success",
        "gpt_model_name",
        "gpt_task_type",
        "gpt_api_mode",
        "gpt_prompt_mode",
        "gpt_success_mode",
        "gpt_prompt",
        "gpt_clean_output",
        "gpt_adversarial_output",
        "gpt_clean_judge_label",
        "gpt_adversarial_judge_label",
        "gpt_clean_judge_pass",
        "gpt_adversarial_judge_pass",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            proxy = item["proxy_eval"]
            caption = item.get("caption_eval") or {}
            vqa_eval = item.get("vqa_eval") or {}
            ocr_eval = item.get("ocr_eval") or {}
            gpt_eval = item.get("gpt_eval") or {}
            writer.writerow(
                {
                    "item_id": item["item_id"],
                    "source_label": item["source_label"],
                    "target_label": item["target_label"],
                    "proxy_success": proxy["proxy_success"],
                    "clean_margin": proxy["clean_margin"],
                    "adversarial_margin": proxy["adversarial_margin"],
                    "margin_gain": proxy["margin_gain"],
                    "caption_success": caption.get("caption_success"),
                    "clean_caption": caption.get("clean_caption", ""),
                    "adversarial_caption": caption.get("adversarial_caption", ""),
                    "vqa_success": vqa_eval.get("vqa_success"),
                    "question": vqa_eval.get("question", item.get("question", "")),
                    "source_answer_text": gpt_eval.get("source_answer_text", item.get("source_answer_text", "")),
                    "target_answer_text": gpt_eval.get("target_answer_text", item.get("target_answer_text", "")),
                    "clean_answer": vqa_eval.get("clean_answer", ""),
                    "adversarial_answer": vqa_eval.get("adversarial_answer", ""),
                    "ocr_success": ocr_eval.get("ocr_success"),
                    "clean_text": ocr_eval.get("clean_text", ""),
                    "adversarial_text": ocr_eval.get("adversarial_text", ""),
                    "gpt_success": gpt_eval.get("gpt_success"),
                    "gpt_model_name": gpt_eval.get("model_name", ""),
                    "gpt_task_type": gpt_eval.get("task_type", ""),
                    "gpt_api_mode": gpt_eval.get("api_mode", ""),
                    "gpt_prompt_mode": gpt_eval.get("prompt_mode", ""),
                    "gpt_success_mode": gpt_eval.get("success_mode", ""),
                    "gpt_prompt": gpt_eval.get("question", gpt_eval.get("prompt", "")),
                    "gpt_clean_output": gpt_eval.get("clean_output", ""),
                    "gpt_adversarial_output": gpt_eval.get("adversarial_output", ""),
                    "gpt_clean_judge_label": (
                        gpt_eval.get("clean_caption_judgment", {}).get("judge_label")
                        or gpt_eval.get("clean_vqa_judgment", {}).get("judge_label")
                    ),
                    "gpt_adversarial_judge_label": (
                        gpt_eval.get("adversarial_caption_judgment", {}).get("judge_label")
                        or gpt_eval.get("adversarial_vqa_judgment", {}).get("judge_label")
                    ),
                    "gpt_clean_judge_pass": (
                        gpt_eval.get("clean_caption_judgment", {}).get("matches_source")
                        if gpt_eval.get("task_type") == "caption"
                        else gpt_eval.get("clean_vqa_judgment", {}).get("meets_ground_truth")
                    ),
                    "gpt_adversarial_judge_pass": (
                        gpt_eval.get("adversarial_caption_judgment", {}).get("matches_target")
                        if gpt_eval.get("task_type") == "caption"
                        else gpt_eval.get("adversarial_vqa_judgment", {}).get("meets_ground_truth")
                    ),
                }
            )
