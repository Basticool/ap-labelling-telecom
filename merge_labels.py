#!/usr/bin/env python3
"""
Merge per-annotator flat-schema label files (JSONL) into a single canonical file.

Use this when each annotator ran the app locally and exported their own labels.
Each input file must be a JSONL produced by the app's Export page.

Usage:
    python merge_labels.py annotator1.jsonl annotator2.jsonl -o merged.jsonl
    python merge_labels.py outputs/*.jsonl -o merged.jsonl --overwrite
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-annotator JSONL label files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL files")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(
            f"Error: '{args.output}' already exists. Use --overwrite to replace it.",
            file=sys.stderr,
        )
        sys.exit(1)

    records: list[dict] = []
    for path in args.inputs:
        if not path.exists():
            print(f"Warning: '{path}' not found, skipping.", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    records.sort(key=lambda r: (
        r.get("annotator_id", ""),
        r.get("sample_id", ""),
        r.get("target_ap", ""),
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Merged {len(records)} records from {len(args.inputs)} file(s) → {args.output}")


if __name__ == "__main__":
    main()
