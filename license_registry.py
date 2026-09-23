#!/usr/bin/env python3
"""license_registry.py — license compatibility registry for third-party components.

Generalized from a project-specific script. The license definitions used to
be hardcoded in the .py file (so updating them meant editing code); they now
live in an external, editable JSON file that ships with a starter set of
common open/open-weight licenses and can be extended without touching this
script at all.

What this tool actually does:
  - Looks up a named license in your registry file and reports its terms
    (commercial use, redistribution, attribution, conditions).
  - Given a list of components and the license each one uses, tells you
    whether the *combination* is distributable, and collects every
    attribution notice and derivative condition you'd need to satisfy.

What it does NOT do (be accurate about this when describing it to anyone):
  - It does not inspect model weights, hashes, or any binary artifact.
  - It does not verify that a component actually uses the license you say
    it does -- it's a bookkeeping and compatibility-logic tool, not a
    forensic scanner. If you need weight-provenance verification, that is
    a separate (much harder) problem this script does not solve.

Usage:
    # Look up one license
    python license_registry.py info --db licenses.json --license Apache-2.0

    # Check whether a set of components can be distributed together
    python license_registry.py check --db licenses.json --components components.json

    # Add or update a license entry in the registry file
    python license_registry.py add --db licenses.json --name "MPL-2.0" \\
        --type permissive --commercial --redistribute --attribution "LICENSE" \\
        --notes "Weak copyleft, file-level"

components.json format: a list of {"name": ..., "license": ...} objects, e.g.
[
  {"name": "component-a", "license": "Apache-2.0"},
  {"name": "component-b", "license": "MIT"}
]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

DEFAULT_LICENSES: Dict[str, dict] = {
    "Apache-2.0": {"type": "permissive", "commercial": True, "redistribute": True,
                   "attribution": "LICENSE + NOTICE", "derivative_conditions": [],
                   "notes": "Widely used permissive license with an explicit patent grant."},
    "MIT": {"type": "permissive", "commercial": True, "redistribute": True,
            "attribution": "LICENSE", "derivative_conditions": [],
            "notes": "Simple permissive license, no patent clause."},
    "AGPL-3.0": {"type": "copyleft", "commercial": True, "redistribute": True,
                 "attribution": "LICENSE", "derivative_conditions": ["network use triggers source disclosure"],
                 "notes": "Strong copyleft; SaaS use counts as distribution."},
    "GPL-3.0": {"type": "copyleft", "commercial": True, "redistribute": True,
                "attribution": "LICENSE", "derivative_conditions": ["derivative works must be GPL-3.0"],
                "notes": "Strong copyleft for distributed binaries."},
    "proprietary-closed": {"type": "closed", "commercial": False, "redistribute": False,
                            "attribution": "n/a", "derivative_conditions": ["no redistribution rights"],
                            "notes": "No license grant exists; treat as excluded until one is obtained."},
}


def load_db(path: str) -> Dict[str, dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return dict(DEFAULT_LICENSES)


def save_db(path: str, db: Dict[str, dict]) -> None:
    with open(path, "w") as f:
        json.dump(db, f, indent=2, sort_keys=True)


def check_component(db: Dict[str, dict], name: str, license_name: str) -> dict:
    info = db.get(license_name)
    if info is None:
        return {"component": name, "license": license_name, "known": False,
                "distributable": False, "action": "ADD LICENSE INFO BEFORE USE"}
    return {
        "component": name, "license": license_name, "known": True,
        "type": info["type"], "commercial_use_ok": info["commercial"],
        "redistributable": info["redistribute"],
        "attribution_required": info["attribution"],
        "conditions": info["derivative_conditions"],
        "distributable": info["redistribute"] and info["type"] != "closed",
        "notes": info.get("notes", ""),
    }


def check_configuration(db: Dict[str, dict], components: List[dict]) -> dict:
    checks = [check_component(db, c["name"], c.get("license", "unknown")) for c in components]
    distributable = all(c["distributable"] for c in checks)
    conditions = []
    for c in checks:
        conditions += [f"{c['component']}: {cond}" for cond in c.get("conditions", [])]
    attributions = [f"{c['component']} ({c.get('attribution_required', '?')})" for c in checks if c["known"]]
    unknown = [c["component"] for c in checks if not c["known"]]
    return {
        "components": [c["component"] for c in checks],
        "overall": "DISTRIBUTABLE" if distributable and not unknown else "NOT DISTRIBUTABLE",
        "unknown_licenses": unknown,
        "required_attributions": attributions,
        "derivative_conditions": conditions,
        "detail": checks,
    }


def cmd_info(args: argparse.Namespace) -> int:
    db = load_db(args.db)
    info = db.get(args.license)
    if info is None:
        print(f"'{args.license}' is not in the registry ({args.db}).")
        return 1
    print(json.dumps({args.license: info}, indent=2))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    db = load_db(args.db)
    with open(args.components) as f:
        components = json.load(f)
    result = check_configuration(db, components)
    print(json.dumps(result, indent=2))
    return 0 if result["overall"] == "DISTRIBUTABLE" else 1


def cmd_add(args: argparse.Namespace) -> int:
    db = load_db(args.db)
    db[args.name] = {
        "type": args.type,
        "commercial": args.commercial,
        "redistribute": args.redistribute,
        "attribution": args.attribution,
        "derivative_conditions": args.condition or [],
        "notes": args.notes or "",
    }
    save_db(args.db, db)
    print(f"Saved '{args.name}' to {args.db}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Track and check third-party license compatibility.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="Show terms for one license")
    p_info.add_argument("--db", required=True)
    p_info.add_argument("--license", required=True)
    p_info.set_defaults(func=cmd_info)

    p_check = sub.add_parser("check", help="Check a set of components for combined distributability")
    p_check.add_argument("--db", required=True)
    p_check.add_argument("--components", required=True)
    p_check.set_defaults(func=cmd_check)

    p_add = sub.add_parser("add", help="Add or update a license entry")
    p_add.add_argument("--db", required=True)
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--type", required=True, choices=["permissive", "copyleft", "community", "closed"])
    p_add.add_argument("--commercial", action="store_true")
    p_add.add_argument("--redistribute", action="store_true")
    p_add.add_argument("--attribution", required=True)
    p_add.add_argument("--condition", action="append", help="Repeatable: a derivative condition")
    p_add.add_argument("--notes", default="")
    p_add.set_defaults(func=cmd_add)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
