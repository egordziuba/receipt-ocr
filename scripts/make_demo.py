"""Render a fictional receipt and illustrate the unchanged accurate OCR pipeline.

Run from any directory: python scripts/make_demo.py
Requires the normal project dependencies, Tesseract 5, downloaded models and a
Cyrillic-capable font (DejaVu Sans on Linux; Arial on Windows/macOS).
Only generated pixels are passed to OCR; no photographic fixtures are read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr import recognize, save_json
from autotess import page_regions


RENDERED_LINES = (
    "КРАМНИЦЯ ХМАРОЛИС",
    "FICTIONAL DEMO SHOP",
    "СИНТЕТИЧНИЙ ЧЕК",
    "Хмарний чай 42.50",
    "Moon biscuits 31.20",
    "Зоряний мед 26.30",
    "TOTAL / РАЗОМ 100.00",
    "DEMO ONLY",
    "НЕ ДЛЯ ОПЛАТИ",
)
PNG_LIMIT = 400_000  # Decimal kB, stricter than 400 KiB.


def load_font(size, override=None):
    candidates = [override] if override else [
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    raise RuntimeError("Install DejaVu Sans or pass --font with a Cyrillic font path")


def render_receipt(font):
    paper = Image.new("RGBA", (660, 840), (250, 248, 240, 255))
    draw = ImageDraw.Draw(paper)
    for index, line in enumerate(RENDERED_LINES):
        if draw.textlength(line, font=font) > paper.width - 80:
            raise ValueError("The selected font is too wide for the receipt")
        draw.text((40, 65 + index * 78), line, font=font, fill=(25, 28, 32))
    paper = paper.rotate(4, resample=Image.Resampling.BICUBIC, expand=True)
    photo = Image.new("RGBA", (840, 1040), (41, 53, 65, 255))
    position = ((photo.width - paper.width) // 2, (photo.height - paper.height) // 2)
    shadow = Image.new("RGBA", photo.size)
    shadow.paste((0, 0, 0, 100), (position[0] + 10, position[1] + 14), paper.getchannel("A"))
    photo = Image.alpha_composite(photo, shadow.filter(ImageFilter.GaussianBlur(10)))
    photo.alpha_composite(paper, position)
    return photo.convert("RGB")


def confidence_color(confidence):
    if confidence >= 85:
        return "#15803d"
    if confidence >= 60:
        return "#c27000"
    return "#dc2626"


def draw_detection(photo, result, legend_font):
    overlay = photo.copy()
    draw = ImageDraw.Draw(overlay)
    # These are the detector's crop transforms, before orientation/deskew.
    # Invert them to draw detected sheet edges in original photo coordinates.
    for region in page_regions(np.asarray(photo)):
        height, width = region["image"].shape[:2]
        corners = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
        points = cv2.perspectiveTransform(corners, np.linalg.inv(region["matrix"]))[0]
        points = [tuple(map(float, point)) for point in points]
        draw.line(points + [points[0]], fill="#38bdf8", width=4)
    for word in result["words"]:
        draw.rectangle(word["box"], outline=confidence_color(word["confidence"]), width=2)
    draw.text((22, 18), "Sheet outline", font=legend_font, fill="#38bdf8")
    for x, label, color in [
        (210, "Confidence >=85", "#4ade80"),
        (440, "60-85", "#fbbf24"),
        (570, "<60", "#f87171"),
    ]:
        draw.text((x, 18), label, font=legend_font, fill=color)
    return overlay


def save_png(image, path):
    image.save(path, optimize=True)
    size = path.stat().st_size
    if size > PNG_LIMIT:
        raise RuntimeError(f"{path.name}: {size} bytes exceeds {PNG_LIMIT}")
    return size


def compare_text(expected, actual):
    def normalize(text):
        return "".join(unicodedata.normalize("NFKC", text).casefold().split())

    reference, recognized = normalize(expected), normalize(actual)
    previous = list(range(len(recognized) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(recognized, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    errors = previous[-1]
    return {
        "scope": "This synthetic example only; not the evaluation set",
        "normalization": "Unicode NFKC, casefold, remove whitespace; no alphabet or digit corrections",
        "reference_characters": len(reference),
        "edit_errors": errors,
        "character_accuracy_pct": round(max(0, 1 - errors / len(reference)) * 100, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, help="Cyrillic-capable TrueType font")
    args = parser.parse_args()
    font = load_font(32, args.font)
    photo = render_receipt(font)
    images = ROOT / "docs" / "img"
    images.mkdir(parents=True, exist_ok=True)
    input_path = images / "demo_input.png"
    input_size = save_png(photo, input_path)

    # Same entry point used by ocr.py --mode accurate. No expected text is supplied.
    result = recognize(input_path, ROOT / "models", mode="accurate")
    if not result["words"]:
        raise RuntimeError("OCR produced no words")
    boxes_size = save_png(draw_detection(photo, result, load_font(18, args.font)),
                          images / "demo_boxes.png")
    rendered = "\n".join(RENDERED_LINES)
    trimmed = {
        "synthetic": True,
        "mode": result["mode"],
        "engine": result["engine"],
        "input_sha256": result["input_sha256"],
        "input_size": result["input_size"],
        "font": Path(font.path).name,
        "rendered_text": rendered,
        "text": result["text"],
        "comparison": compare_text(rendered, result["text"]),
        "word_count": len(result["words"]),
        "word_sample_note": "First five final words; boxes are [x1, y1, x2, y2] in original image pixels; confidence is Tesseract's score, not probability",
        "words": [{key: word[key] for key in ("text", "box", "confidence")}
                  for word in result["words"][:5]],
        "model_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted((ROOT / "models").glob("*.traineddata"))},
    }
    save_json(ROOT / "docs" / "demo_output.json", trimmed)
    print(f"demo_input.png: {input_size} bytes; demo_boxes.png: {boxes_size} bytes")
    print("Rendered text:\n" + rendered)
    print("Actual OCR:\n" + result["text"])
    print(json.dumps(trimmed["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
