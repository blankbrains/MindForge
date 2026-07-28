"""Run a private, manifest-driven parser benchmark corpus."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from mindforge.ingestion.parsers import DocumentParser, DocumentParserError


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("Unsupported benchmark manifest schema.")
    if not isinstance(payload.get("cases"), list):
        raise ValueError("Benchmark manifest must contain a cases list.")
    return payload


def _validate_case(parsed: Any, expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    metadata = dict(parsed.metadata or {})
    if int(metadata.get("ocr_pages") or 0) < int(
        expected.get("minimum_ocr_pages") or 0
    ):
        failures.append("OCR page count below expectation")
    if int(metadata.get("native_text_pages") or 0) < int(
        expected.get("minimum_native_text_pages") or 0
    ):
        failures.append("Native text page count below expectation")
    if int(metadata.get("table_count") or 0) < int(
        expected.get("minimum_table_count") or 0
    ):
        failures.append("Table count below expectation")
    element_types = {element.kind for element in parsed.elements}
    for element_type in expected.get("required_element_types") or []:
        if element_type not in element_types:
            failures.append(f"Missing required element type: {element_type}")
    actual_reasons = {
        reason
        for page in metadata.get("page_metrics") or []
        for reason in page.get("routing_reasons") or []
    }
    for reason in expected.get("required_routing_reasons") or []:
        if reason not in actual_reasons:
            failures.append(f"Missing routing reason: {reason}")
    return failures


def run_case(parser: DocumentParser, source: Path, expected: dict[str, Any]) -> dict:
    started = time.perf_counter()
    try:
        parsed = parser.parse(source)
    except DocumentParserError as exc:
        return {
            "status": "error",
            "error": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    failures = _validate_case(parsed, expected)
    elapsed = time.perf_counter() - started
    maximum_seconds = expected.get("maximum_elapsed_seconds")
    if maximum_seconds is not None and elapsed > float(maximum_seconds):
        failures.append(
            f"Elapsed time {elapsed:.3f}s exceeded budget {maximum_seconds}s"
        )
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "elapsed_seconds": round(elapsed, 3),
        "doc_id": parsed.doc_id,
        "metadata": parsed.metadata,
        "element_counts": {
            kind: sum(element.kind == kind for element in parsed.elements)
            for kind in ("text", "table", "image")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/parser/manifest.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        raise ValueError(f"Corpus directory does not exist: {corpus}")

    document_parser = DocumentParser()
    results: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_version": document_parser._parser_config.pipeline_version,
        "parser_config": document_parser._parser_config.model_dump(
            mode="json"
        ),
        "generated_at_epoch": round(time.time(), 3),
        "cases": {},
    }
    passed = True
    for case in manifest["cases"]:
        case_id = str(case["id"])
        source = corpus / str(case["source"])
        if not source.is_file():
            results["cases"][case_id] = {
                "status": "error",
                "error": f"Missing corpus file: {source.name}",
            }
            passed = False
            continue
        result = run_case(
            document_parser,
            source,
            dict(case.get("expect") or {}),
        )
        results["cases"][case_id] = result
        passed = passed and result["status"] == "passed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
