"""Download the Tesseract tessdata_best models listed in model_manifest.json and verify their SHA-256."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    target = ROOT / "models"
    target.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        path = target / f"{item['language']}.traineddata"
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]:
            print(f"ok       {path.name}")
            continue
        print(f"download {path.name} ({item['bytes'] / 1e6:.1f} MB)")
        data = urllib.request.urlopen(item["url"], timeout=120).read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["sha256"]:
            print(f"checksum mismatch for {path.name}: {digest}", file=sys.stderr)
            return 1
        path.write_bytes(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
