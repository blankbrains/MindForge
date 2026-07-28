# Parser Benchmark Corpus

`manifest.json` is safe to commit. It defines the private corpus contract but
does not include source PDFs, generated captions, assets, or benchmark reports.

Run the benchmark only with a local or server-side private corpus:

```bash
python scripts/benchmark_parser.py \
  --corpus /secure/path/to/parser-corpus \
  --output data/benchmarks/parser-report.json
```

Each manifest case can require native/OCR pages, tables, element kinds, visual
routing reasons, and a maximum elapsed time. Treat model, pipeline, or layout
changes as incomplete until this corpus and the automated regression tests pass.
