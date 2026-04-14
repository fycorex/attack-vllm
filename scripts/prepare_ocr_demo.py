#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont


TEXT_SPECS = [
    ("item_00", "plane", "ship"),
    ("item_01", "truck", "boat"),
    ("item_02", "car", "truck"),
    ("item_03", "ship", "plane"),
]

TEXT_KEYWORDS = {
    "plane": ["plane", "airplane", "aircraft", "jet"],
    "ship": ["ship", "boat", "vessel"],
    "boat": ["boat", "ship", "vessel"],
    "car": ["car", "automobile", "vehicle"],
    "truck": ["truck", "lorry"],
}

BACKGROUND_PALETTE = [
    (245, 248, 252),
    (248, 245, 240),
    (240, 246, 243),
    (250, 250, 250),
]

TEXT_PALETTE = [
    (20, 20, 20),
    (10, 50, 90),
    (80, 30, 30),
    (30, 80, 40),
]


def resolve_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_text_image(text: str, variant_seed: int, size: int = 224) -> Image.Image:
    rng = random.Random(variant_seed)
    canvas_size = size * 2
    background = BACKGROUND_PALETTE[variant_seed % len(BACKGROUND_PALETTE)]
    image = Image.new("RGB", (canvas_size, canvas_size), background)
    draw = ImageDraw.Draw(image)
    font = resolve_font(rng.randint(88, 112))
    text_color = TEXT_PALETTE[variant_seed % len(TEXT_PALETTE)]
    text_value = text.upper()
    bbox = draw.textbbox((0, 0), text_value, font=font, stroke_width=2)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (canvas_size - text_width) // 2 + rng.randint(-24, 24)
    y = (canvas_size - text_height) // 2 + rng.randint(-18, 18)
    draw.text((x, y), text_value, fill=text_color, font=font, stroke_width=2, stroke_fill=(255, 255, 255))

    if rng.random() < 0.5:
        underline_y = y + text_height + rng.randint(4, 14)
        draw.line((x, underline_y, x + text_width, underline_y), fill=text_color, width=rng.randint(2, 5))

    image = image.rotate(rng.uniform(-8.0, 8.0), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=background)
    crop_margin = rng.randint(16, 40)
    image = image.crop((crop_margin, crop_margin, canvas_size - crop_margin, canvas_size - crop_margin))
    return image.resize((size, size), resample=Image.Resampling.LANCZOS)


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a synthetic OCR-word demo manifest.")
    parser.add_argument("--output_dir", default="data/ocr_word_attack_demo")
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--num_examples_per_class", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for item_index, (item_id, source_text, target_text) in enumerate(TEXT_SPECS[: args.num_items]):
        source_image = render_text_image(source_text, variant_seed=item_index)
        image_path = images_dir / f"{item_id}.png"
        save_image(source_image, image_path)

        positive_paths = []
        negative_paths = []
        for example_idx in range(args.num_examples_per_class):
            positive_image = render_text_image(target_text, variant_seed=1000 + item_index * 100 + example_idx)
            negative_image = render_text_image(source_text, variant_seed=2000 + item_index * 100 + example_idx)
            positive_path = examples_dir / target_text / f"positive_{item_id}_{example_idx:02d}.png"
            negative_path = examples_dir / source_text / f"negative_{item_id}_{example_idx:02d}.png"
            save_image(positive_image, positive_path)
            save_image(negative_image, negative_path)
            positive_paths.append(str(positive_path.resolve()))
            negative_paths.append(str(negative_path.resolve()))

        items.append(
            {
                "id": item_id,
                "image_path": str(image_path.resolve()),
                "source_label": source_text,
                "target_label": target_text,
                "source_keywords": TEXT_KEYWORDS[source_text],
                "target_keywords": TEXT_KEYWORDS[target_text],
                "source_text_keywords": TEXT_KEYWORDS[source_text],
                "target_text_keywords": TEXT_KEYWORDS[target_text],
                "positive_image_paths": positive_paths,
                "negative_image_paths": negative_paths,
            }
        )

    manifest = {
        "dataset_name": "ocr-word-attack-demo",
        "metadata": {
            "note": "Synthetic word-image OCR demo for local attack validation. This is an OCR smoke-test dataset, not the paper's receipt benchmark.",
            "num_items": min(args.num_items, len(TEXT_SPECS)),
            "num_examples_per_class": args.num_examples_per_class,
        },
        "items": items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
