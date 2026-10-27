#!/usr/bin/env python3
"""Profile every JSON field and nested record in a TCGA-OV case export."""
import argparse
import json
from collections import Counter
from pathlib import Path


def kind(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def walk(value, path, stats, parent_count):
    """Collect path coverage/types; array indices are collapsed to []."""
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        field_path = f"{path}.{key}" if path else key
        entry = stats.setdefault(field_path, {
            "records_with_field": 0, "types": Counter(), "examples": Counter(),
            "array_item_types": Counter(), "array_items": 0,
        })
        entry["records_with_field"] += 1
        entry["types"][kind(item)] += 1
        if item is not None and not isinstance(item, (dict, list)):
            entry["examples"][str(item)[:100]] += 1
        elif isinstance(item, dict):
            walk(item, field_path, stats, parent_count)
        elif isinstance(item, list):
            for child in item:
                entry["array_items"] += 1
                entry["array_item_types"][kind(child)] += 1
                if isinstance(child, dict):
                    walk(child, field_path + "[]", stats, parent_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="docs/data/dataset_field_audit.json")
    args = parser.parse_args()
    cases = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not all(isinstance(c, dict) for c in cases):
        raise ValueError("Expected a JSON array of case objects")
    stats = {}
    for case in cases:
        walk(case, "", stats, len(cases))

    fields = {}
    for path, entry in sorted(stats.items()):
        examples = sorted(entry["examples"].items(), key=lambda x: (-x[1], x[0]))
        fields[path] = {
            "records_with_field": entry["records_with_field"],
            "case_coverage_percent": round(100 * entry["records_with_field"] / len(cases), 2)
                if "[]" not in path else None,
            "types": dict(sorted(entry["types"].items())),
            "distinct_scalar_values_observed": len(entry["examples"]),
            "common_scalar_examples": [{"value": v, "count": n} for v, n in examples[:20]],
            "array_item_count": entry["array_items"],
            "array_item_types": dict(sorted(entry["array_item_types"].items())),
        }
    report = {
        "input_file": Path(args.input).name,
        "case_count": len(cases),
        "field_path_count_including_nested": len(fields),
        "notes": [
            "Coverage is the number of records at the immediate parent level containing the key; for paths under arrays, this is record-level coverage, not patient-level coverage.",
            "Scalar examples are descriptive and may be truncated; source values are unchanged.",
            "This mechanical audit does not establish clinical meaning, timing origin, availability at a decision time, or causal validity.",
        ],
        "fields": fields,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audited {len(cases)} cases and {len(fields)} nested field paths; wrote {output.resolve()}")


if __name__ == "__main__":
    main()
