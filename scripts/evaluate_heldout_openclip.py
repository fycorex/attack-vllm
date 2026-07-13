#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import SurrogateConfig
from data import load_image_tensor, load_manifest
from eval import evaluate_proxy
from surrogates import create_surrogate, unload_surrogate


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay generated pairs on disjoint held-out OpenCLIP models.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--attack-output", required=True)
    parser.add_argument("--models", required=True, help="JSON list of SurrogateConfig objects")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="models/open_clip")
    args = parser.parse_args()
    specs = [SurrogateConfig(**x) for x in json.loads(args.models)]
    manifest = load_manifest(args.manifest); attack_output = Path(args.attack_output)
    item_rows = []
    for spec in specs:
        model = create_surrogate(spec, args.device if torch.cuda.is_available() else "cpu", cache_dir=args.cache_dir)
        try:
            for item in manifest.items:
                item_dir = attack_output / item.item_id
                if not (item_dir / "adversarial.png").exists(): continue
                size = spec.input_size
                clean = load_image_tensor(item_dir / "clean.png", size).unsqueeze(0).to(args.device)
                adv = load_image_tensor(item_dir / "adversarial.png", size).unsqueeze(0).to(args.device)
                positives = torch.stack([load_image_tensor(p, size) for p in item.positive_image_paths]).to(args.device)
                negatives = torch.stack([load_image_tensor(p, size) for p in item.negative_image_paths]).to(args.device)
                with torch.no_grad():
                    result = evaluate_proxy(model.encode_image(clean), model.encode_image(adv), model.encode_image(positives), model.encode_image(negatives), top_k=min(4, len(item.positive_image_paths)), success_margin_threshold=0.0)
                item_rows.append({"item_id": item.item_id, "model": model.name, **result})
        finally: unload_surrogate(model)
    models = sorted({r["model"] for r in item_rows})
    per_model = {name: {"items": sum(r["model"] == name for r in item_rows),
                        "asr": sum(bool(r["proxy_success"]) for r in item_rows if r["model"] == name) / max(1, sum(r["model"] == name for r in item_rows)),
                        "mean_margin_gain": sum(float(r["margin_gain"]) for r in item_rows if r["model"] == name) / max(1, sum(r["model"] == name for r in item_rows))} for name in models}
    summary = {"paired_items": len({r["item_id"] for r in item_rows}), "heldout_models": per_model,
               "heldout_macro_asr": sum(v["asr"] for v in per_model.values()) / max(1, len(per_model)),
               "heldout_macro_margin_gain": sum(v["mean_margin_gain"] for v in per_model.values()) / max(1, len(per_model)), "items": item_rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=2))


if __name__ == "__main__": main()
