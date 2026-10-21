# Receipt OCR — offline Tesseract pipeline for photographed business documents

Local OCR for phone photos of printed receipts, fiscal day reports, card-terminal slips, supplier invoices and sales screens in Ukrainian, Russian and English. Built for a small retail business that wanted to stop retyping daily paperwork; the photos and reference data of that business are **not** part of this repository.

Project documentation is in Russian ([README.ru.md](README.ru.md)).

## How it works

- Finds light paper sheets by geometry (no predefined layout), so several receipts in one photo are processed separately
- Picks the rotation out of four by OCR quality, corrects small skew and maps word boxes back to the original image
- Several image variants per page: untouched grey, illumination normalisation, upscaling, grid-line removal, binarisation (Sauvola optional and experimental)
- Tesseract `tessdata_best` models, page segmentation modes 6 and 11, fusion of words already recognised in different passes — no characters are "completed" from expected answers
- Automatic re-scan of detected text groups on an enlarged mosaic canvas
- Every pass, coordinate and alternative is kept; duplicates are detected by SHA-256 of the file content
- No network calls, no cloud OCR, no LLM, no accounting-equation "corrections"

## Honest evaluation

A separate evaluator (`evaluate.py`) compares results with an independent, region-level gold annotation using Levenshtein character accuracy and exact money-amount matches, with strict provenance checks (image hashes, single algorithm version, complete run).

On the private set of 20 unique photos (176 legible control regions, 1,940 characters) the best profile reached **90.4 % character accuracy** and **58 / 77 money amounts exactly right**. The 95 % target was **not** reached, so the tool is not cleared for automatic bookkeeping. Handwriting and faint dot-matrix print remain the hardest cases.

## Usage

```bash
# Tesseract 5 must be installed (or set TESSERACT_CMD)
pip install -r requirements.txt
python scripts/download_models.py          # ukr / rus / eng, verified by SHA-256
python ocr.py --input path/to/photos --output results --mode accurate
```

Output per image: `.ocr.json` (words, boxes, confidences, all passes), `.ocr.txt` and a `manifest.json`. On Windows, drag photos onto `RUN_OCR.cmd` after `SETUP.cmd`.

## Tests

`pip install -r requirements-test.txt && python -m pytest -q` — 31 tests that render synthetic text and run the real OCR (Tesseract 5 and the downloaded models are required; Windows font paths are mapped automatically on Linux/macOS). Two tests that use real photos are skipped unless `OCR_FIXTURE_DIR` points to them. Last run: 29 passed, 2 skipped.

Models: Tesseract `tessdata_best`, Apache-2.0 ([LICENSE_MODELS.txt](LICENSE_MODELS.txt)).
