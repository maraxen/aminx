"""U5 checker — vendor/reference feature-surface superset.

Public API:
    extract_argparse_flags(source_path: Path) -> list[dict]
    check_u5(target_dataclasses: Iterable[type], vendor_flags: list[dict],
             alias_map: dict[str, dict]) -> dict
    run_admissibility(config: dict) -> dict

Answers a different question from U1. U1 asks "is this declared field reachable from the
target codebase's own consumers" (an internal reachability question). U5 asks "does the
target codebase's declared surface even contain a concept that a reference/vendor
implementation exposes" (a cross-codebase coverage question). A field can pass U5 (the
concept exists) and still fail U1 (it's declared but never read) -- the two are independent
axes and should not be conflated into one verdict.

`extract_argparse_flags` is AST-only because vendor CLIs are typically argparse-based, not
reflective dataclasses -- there is no `fields()` to call. The target side uses
`dataclasses.fields()` reflection (like U1), since that is more robust than re-deriving
AnnAssign structure via AST for a class that is already safely importable.

`alias_map` is supplied by the caller's config, never hardcoded here: `{vendor_flag_name:
{"target_fields": [...], "category": str, "notes": str}}`. An empty `target_fields` list
means the caller asserts no target equivalent exists -- U5 will report ABSENT for that flag
rather than silently skipping it, so a deliberately-absent mapping still counts toward
coverage instead of disappearing from the matrix.

Portable: no target-project or vendor-project names appear in this file.
"""
from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


def extract_argparse_flags(source_path: Path) -> list[dict[str, Any]]:
    """AST-walk a Python source file for every `<obj>.add_argument(...)` call.

    Returns one dict per flag: {flag, type, default, help, source_line}. Missing/unparseable
    keyword values degrade to None rather than raising -- a vendor CLI is read-only input,
    never assumed to parse cleanly in every corner.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    out: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        first_value = node.args[0].value
        if not isinstance(first_value, str):
            continue
        flag_name = first_value.lstrip("-")
        type_txt = default_txt = help_txt = None
        for kw in node.keywords:
            if kw.arg == "type" and isinstance(kw.value, ast.Name):
                type_txt = kw.value.id
            elif kw.arg == "default":
                try:
                    default_txt = ast.unparse(kw.value)
                except Exception:
                    default_txt = "<unparseable>"
            elif kw.arg == "help" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                help_txt = kw.value.value
        out.append({
            "flag": flag_name, "type": type_txt, "default": default_txt,
            "help": help_txt, "source_line": node.lineno,
        })
    return out


def check_u5(target_dataclasses: Iterable[type], vendor_flags: list[dict[str, Any]],
             alias_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Run U5 superset coverage: which alias_map-listed vendor flags have a target field."""
    target_field_names: set[str] = set()
    for cls in target_dataclasses:
        assert is_dataclass(cls), f"{cls} must be a dataclass"
        target_field_names.update(f.name for f in fields(cls))

    vendor_by_flag = {vf["flag"]: vf for vf in vendor_flags}
    verdicts: list[dict[str, Any]] = []
    for flag_name, mapping in alias_map.items():
        expected_fields = mapping.get("target_fields", [])
        present_fields = [f for f in expected_fields if f in target_field_names]
        vendor_entry = vendor_by_flag.get(flag_name)
        verdicts.append({
            "vendor_flag": flag_name,
            "vendor_flag_found_in_source": vendor_entry is not None,
            "vendor_help": vendor_entry.get("help") if vendor_entry else None,
            "category": mapping.get("category"),
            "expected_target_fields": expected_fields,
            "present_target_fields": present_fields,
            "verdict": (
                "U5-HYPOTHESIS-PASS" if expected_fields and present_fields
                else "U5-HYPOTHESIS-FAIL"
            ),
            "notes": mapping.get("notes"),
        })
    return {
        "alias_map_entries_checked": len(alias_map),
        "vendor_flags_extracted_total": len(vendor_flags),
        "target_fields_extracted_total": len(target_field_names),
        "verdicts": verdicts,
        "present_count": sum(1 for v in verdicts if v["verdict"] == "U5-HYPOTHESIS-PASS"),
        "absent_count": sum(1 for v in verdicts if v["verdict"] == "U5-HYPOTHESIS-FAIL"),
    }


def run_admissibility(
    target_dataclasses: Iterable[type], vendor_flags: list[dict[str, Any]],
    known_positive_flag: str, known_positive_target_field: str,
    injection_flag: str = "_audit_synth_vendor_flag",
) -> dict[str, Any]:
    """Admissibility harness for U5.

    - known_positive_flag / known_positive_target_field: a real vendor flag you expect the
      checker to report PASS for, mapped to a real field on one of `target_dataclasses`.
    - injection_flag: a synthetic vendor flag name, mapped to a synthetic target field that
      provably does not exist on any of `target_dataclasses` -- the checker must report FAIL.
    """
    target_dataclasses = list(target_dataclasses)
    kp_alias_map = {
        known_positive_flag: {"target_fields": [known_positive_target_field], "category": "kp"},
    }
    kp_result = check_u5(target_dataclasses, vendor_flags, kp_alias_map)
    kp_verdict = kp_result["verdicts"][0]

    kn_alias_map = {
        injection_flag: {"target_fields": ["_audit_synth_target_field_that_does_not_exist"], "category": "kn"},
    }
    kn_result = check_u5(target_dataclasses, vendor_flags, kn_alias_map)
    kn_verdict = kn_result["verdicts"][0]

    return {
        "checker": "U5 vendor/reference feature-surface superset (static, symbolic)",
        "known_positive": {
            "vendor_flag": known_positive_flag,
            "target_field": known_positive_target_field,
            "expected": "U5-HYPOTHESIS-PASS",
            "actual_verdict": kp_verdict["verdict"],
        },
        "known_negative": {
            "vendor_flag": injection_flag,
            "injection_diff": (
                "Synthetic alias_map entry naming a target field that provably does not "
                "exist on any target dataclass; no source modified."
            ),
            "expected": "U5-HYPOTHESIS-FAIL",
            "actual_verdict": kn_verdict["verdict"],
        },
        "admitted": (
            kp_verdict["verdict"] == "U5-HYPOTHESIS-PASS"
            and kn_verdict["verdict"] == "U5-HYPOTHESIS-FAIL"
        ),
        "notes": (
            "Structural surface-existence HYPOTHESIS only -- this checker never claims a "
            "present field is WIRED into the target's own consumers. That is U1's question, "
            "scoped to the target codebase alone; do not conflate the two axes in one verdict."
        ),
    }
