#!/usr/bin/env python3
"""claim_validator.py — generic cross-checker for prose claims vs. structured data.

Generalized from a project-specific script. Instead of hardcoding field names,
JSON paths, and phrases from one experiment, all of that now lives in a rules
file you write per-project. The engine is domain-agnostic: point it at any
JSON results file, any text/markdown report, and a rules file describing what
"consistent" means for your data, and it will tell you where the prose and
the numbers disagree.

Usage:
    python claim_validator.py --data results.json --report report.md --rules rules.json
    python claim_validator.py --data results.json --report report.md --rules rules.json --out validation.json

Exit code is 0 if every check passes, 1 otherwise (useful in CI pipelines).

Rules file format (JSON), a list of check objects. Every check has "type"
and "id"; other fields depend on the type. Supported types:

  "compare"
      Compares two numeric values pulled from the data by dot-path and,
      if the comparison holds, forbids (or requires) certain phrases in
      the report text.
      {
        "id": "no_false_improvement_claim",
        "type": "compare",
        "left": "results.model_b.accuracy",
        "op": "<",                 # one of < <= > >= == !=
        "right": "results.baseline.accuracy",
        "if_true": {
          "forbid_patterns": ["b\\s+outperformed\\s+baseline"]
        }
      }

  "bool_equals"
      Asserts a boolean field in the data equals an expected value.
      {
        "id": "frozen_weights_required",
        "type": "bool_equals",
        "path": "training.transplanted_unit_frozen",
        "expected": true
      }

  "keys_exist"
      Asserts a list of dot-paths all exist in the data.
      {
        "id": "controls_present",
        "type": "keys_exist",
        "paths": ["controls.random_component", "controls.target_native"]
      }

  "threshold_classification"
      If an observed value is below a threshold, a classification field
      must not equal a forbidden label, and the report text must not
      claim a specific phrase.
      {
        "id": "no_overclaiming_agreement",
        "type": "threshold_classification",
        "observed": "summary.observed_agreement_pct",
        "threshold": "summary.predeclared_threshold_pct",
        "classification_path": "summary.classification",
        "forbidden_label": "HIGH_AGREEMENT",
        "forbidden_phrase": "achieved high agreement"
      }

  "forbidden_patterns"
      Unconditionally forbids a list of regex patterns in the report
      text (e.g. overreaching universal claims).
      {
        "id": "no_universal_overreach",
        "type": "forbidden_patterns",
        "patterns": ["proves .* can never", "fundamentally impossible"]
      }

  "unique_values"
      Asserts that a list of values (pulled from a list-of-dicts by
      dot-path) are all distinct.
      {
        "id": "distinct_seeds",
        "type": "unique_values",
        "list_path": "runs",
        "field": "seed_hash"
      }

This file has no dependency on any specific project's schema — it only
knows how to walk dotted paths through JSON and evaluate the check types
above. Add new check types by extending CHECK_HANDLERS.
"""
from __future__ import annotations

import argparse
import json
import operator
import re
import sys
from typing import Any, Dict, List

OPS = {
    "<": operator.lt, "<=": operator.le,
    ">": operator.gt, ">=": operator.ge,
    "==": operator.eq, "!=": operator.ne,
}


def get_path(data: Any, path: str) -> Any:
    """Walk a dotted path (with optional [index]) through nested dict/list data.

    "a.b.c" -> data["a"]["b"]["c"]
    "runs[2].seed" -> data["runs"][2]["seed"]
    Raises KeyError/IndexError with the failing path if not found.
    """
    cur = data
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            idx = int(part[1:-1])
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"path '{path}' not found (missing key '{part}')")
            cur = cur[part]
    return cur


def path_exists(data: Any, path: str) -> bool:
    try:
        get_path(data, path)
        return True
    except (KeyError, IndexError, TypeError):
        return False


def check_compare(data: Dict, text: str, rule: Dict, errors: List[str]) -> None:
    left = get_path(data, rule["left"])
    right = get_path(data, rule["right"])
    op = OPS[rule["op"]]
    if op(left, right):
        branch = rule.get("if_true", {})
        for pat in branch.get("forbid_patterns", []):
            if re.search(pat, text, re.IGNORECASE):
                errors.append(
                    f"[{rule['id']}] {rule['left']}={left} {rule['op']} {rule['right']}={right} holds, "
                    f"but report matches forbidden pattern '{pat}'"
                )
        for pat in branch.get("require_patterns", []):
            if not re.search(pat, text, re.IGNORECASE):
                errors.append(
                    f"[{rule['id']}] {rule['left']}={left} {rule['op']} {rule['right']}={right} holds, "
                    f"but report is missing required pattern '{pat}'"
                )


def check_bool_equals(data: Dict, text: str, rule: Dict, errors: List[str]) -> None:
    val = get_path(data, rule["path"])
    if bool(val) != bool(rule["expected"]):
        errors.append(f"[{rule['id']}] {rule['path']}={val}, expected {rule['expected']}")


def check_keys_exist(data: Dict, text: str, rule: Dict, errors: List[str]) -> None:
    for p in rule["paths"]:
        if not path_exists(data, p):
            errors.append(f"[{rule['id']}] required path '{p}' is missing from data")


def check_threshold_classification(data: Dict, text: str, rule: Dict, errors: List[str]) -> None:
    observed = get_path(data, rule["observed"])
    threshold = get_path(data, rule["threshold"])
    if observed < threshold:
        cls = get_path(data, rule["classification_path"]) if path_exists(data, rule["classification_path"]) else None
        if cls == rule.get("forbidden_label"):
            errors.append(
                f"[{rule['id']}] observed {observed} < threshold {threshold}, "
                f"but classification is '{cls}'"
            )
        phrase = rule.get("forbidden_phrase")
        if phrase and phrase.lower() in text.lower():
            errors.append(
                f"[{rule['id']}] observed {observed} < threshold {threshold}, "
                f"but report text contains '{phrase}'"
            )


def check_forbidden_patterns(data: Dict, text: str, rule: Dict, errors: List[str]) -> None:
    for pat in rule["patterns"]:
        if re.search(pat, text, re.IGNORECASE):
            errors.append(f"[{rule['id']}] report matches forbidden pattern '{pat}'")


def check_unique_values(data: Dict, text: str, rule: Dict, errors: List[str]) -> None:
    items = get_path(data, rule["list_path"])
    values = [it[rule["field"]] for it in items]
    if len(set(values)) != len(values):
        errors.append(f"[{rule['id']}] values at '{rule['list_path']}[].{rule['field']}' are not all unique: {values}")


CHECK_HANDLERS = {
    "compare": check_compare,
    "bool_equals": check_bool_equals,
    "keys_exist": check_keys_exist,
    "threshold_classification": check_threshold_classification,
    "forbidden_patterns": check_forbidden_patterns,
    "unique_values": check_unique_values,
}


def run(data_path: str, report_path: str, rules_path: str) -> Dict:
    with open(data_path) as f:
        data = json.load(f)
    with open(report_path) as f:
        text = f.read()
    with open(rules_path) as f:
        rules = json.load(f)

    errors: List[str] = []
    passed: List[str] = []
    skipped: List[str] = []

    for rule in rules:
        handler = CHECK_HANDLERS.get(rule["type"])
        if handler is None:
            skipped.append(f"{rule.get('id', '?')}: unknown check type '{rule['type']}'")
            continue
        before = len(errors)
        try:
            handler(data, text, rule, errors)
        except (KeyError, IndexError, TypeError) as e:
            errors.append(f"[{rule.get('id', '?')}] check raised an error (likely a bad path): {e}")
        if len(errors) == before:
            passed.append(rule.get("id", rule["type"]))

    result = {
        "all_checks_passed": len(errors) == 0,
        "total_checks": len(rules),
        "passed_checks": passed,
        "skipped_checks": skipped,
        "error_count": len(errors),
        "errors": errors,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a text report's claims against structured data.")
    ap.add_argument("--data", required=True, help="Path to the JSON results/data file")
    ap.add_argument("--report", required=True, help="Path to the markdown/text report file")
    ap.add_argument("--rules", required=True, help="Path to the JSON rules file describing checks")
    ap.add_argument("--out", default=None, help="Where to write the validation result JSON (optional)")
    args = ap.parse_args()

    result = run(args.data, args.report, args.rules)

    print("=" * 70)
    print("CLAIM VALIDATOR")
    print("=" * 70)
    print(f"Checks run:    {result['total_checks']}")
    print(f"Checks passed: {len(result['passed_checks'])}")
    print(f"Errors:        {result['error_count']}")
    if result["skipped_checks"]:
        print(f"Skipped (unknown type): {result['skipped_checks']}")
    if result["errors"]:
        print("\nVIOLATIONS:")
        for e in result["errors"]:
            print(f"  - {e}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote full result to {args.out}")

    print("=" * 70)
    if result["all_checks_passed"]:
        print("ALL CHECKS PASSED.")
    else:
        print("VALIDATION FAILED.", file=sys.stderr)
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
