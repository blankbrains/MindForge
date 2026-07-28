# Document Parsing Operations

## Persisted assets

MindForge stores source files, extracted image crops, rendered OCR-page previews,
and table structure under `PARSER_ASSET_STORAGE_DIR`. Asset records belong to
the document catalog and are deleted during rollback or document deletion.
The asset API only serves rendered visual assets; retained raw source files are
kept for reprocessing and are not exposed by that route.

Native and OCR tables retain Markdown for retrieval plus HTML and normalized
cell JSON for reconstruction. HTML and cell maps live in the document-asset
record rather than every Qdrant or BM25 payload. Native text blocks exclude
recognized table regions, preventing duplicate table indexing.

## Complex documents

Parsing records block-level reading order, page coordinates, OCR confidence, and
route flags. `formula_candidate`, `embedded_visual`, and
`low_ocr_confidence` indicate that a page should be visually reviewed; they do
not claim that formulas, charts, or handwriting have been fully understood.

Cancellation is checked between native-text batches, pages, OCR operations, and
table recognition boundaries. Index jobs expose parser progress, elapsed time,
estimated remaining time, OCR counts, and page-level metrics.

## Optional visual retrieval

Visual retrieval is off by default. Enable it only when `VISUAL_ENABLED=true`
and `VISUAL_MODEL` plus `VISUAL_API_KEY` identify a compatible
OpenAI-compatible vision endpoint. The selected image assets are sent to that
endpoint, their factual captions are stored, and the captions are indexed with
the existing text embedding pipeline.

When visual retrieval is disabled or incomplete, the application makes no
external visual request and creates no visual chunks. This is intentional:
persisted images alone are not represented as semantic vision vectors.

## Versioning and benchmarks

`PARSER_PIPELINE_VERSION`, `PARSER_OCR_MODEL_VERSION`,
`PARSER_TABLE_MODEL_VERSION`, parser settings, and non-secret visual settings
are included in the index signature. A change requires reindexing.

The benchmark manifest is `benchmarks/parser/manifest.json`. Keep private
corpus PDFs under `benchmarks/parser/corpus/` and results under
`benchmarks/parser/results/`; both paths are ignored by Git. Run:

```bash
python scripts/benchmark_parser.py --corpus <private-corpus> --output <result.json>
```
