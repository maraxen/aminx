"""HLO loop-body structural cost instrumentation (T7, P07 diagnostic; xtrax #1983 tie-in).

From `compiled.as_text()`, find every `while` instruction's body/condition computations
and compute the metric `body_operand_elements` (r1 M14 rename -- this is NOT an
output-element count): Sigma over the body's instructions of a per-instruction weight:

- `fusion`: counted ONCE, at the call site, as Sigma operand elements + output elements.
  The walker does NOT descend into the fused computation (`calls=` of a fusion), so
  fusion internals are never double-counted;
- unfused `reduce`, `reduce-window`, `dot`, `convolution`, `gather`, `sort`: Sigma operand
  elements + output elements (a reduction to a scalar costs its operand, not 1);
- `dynamic-update-slice` / `scatter`: update operand (+ scatter indices) elements, not
  the full output; `dynamic-slice`: output elements;
- any other unfused op: output elements;
- `parameter`, `constant`, `tuple`, `get-tuple-element` and `bitcast` are excluded;
  `copy` is counted (a carried-buffer copy is exactly what #1983 wants to find).

Traversal: descend into `call` (`to_apply=`/`calls=` of a `call` op) and into
`conditional` branches (`branch_computations={...}`, max over branches). A nested
`while` has its body counted once and is flagged `nested_while` (trip count unknown).
The scalar reducer computations named by `to_apply=` on reduce/scatter/sort/all-reduce
are NOT descended.

Scaling alpha = log2(m(2L) / m(L)) for m = body_operand_elements, guarded (None if
either side is <= 0). Reuses `xtrax.profiling.trace.scope_map_from_hlo_text` for
named-scope attribution where op_name-carried labels exist (best-effort, non-gating).

Real compiled HLO (confirmed empirically against the installed jax/jaxlib before writing
this parser, per BATHOS.md "verify the measurement pipeline on synthetic ground truth
first"):
- computations are printed at the top level, never lexically nested; each opens with a
  header line ending in `{` (`NAME (params) -> RETTYPE {`, optionally `ENTRY `-prefixed)
  and closes with a bare `}` line;
- a `while` instruction is `%name = TYPE while(%operand), condition=%C, body=%B, ...`
  (condition/body order is NOT guaranteed -- resolved independently via search, not
  positional parsing);
- a `conditional` instruction carries `branch_computations={%A, %B, ...}`, not a
  positional branch-computation list;
- XLA frequently wraps a whole `while` (and its surrounding tuple-packing) in a
  top-level `call(...), to_apply=%some_computation` (`xla_cpu_small_call` heuristic) --
  `while` instructions are therefore searched across EVERY parsed computation, not just
  ENTRY.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

try:
  from xtrax.profiling.trace import scope_map_from_hlo_text
except ImportError:  # pragma: no cover -- best-effort enrichment only, never gating
  scope_map_from_hlo_text = None  # type: ignore[assignment]

# Ops excluded from the weight entirely (never appear in the top-10 list either).
EXCLUDED_OPS = frozenset({"parameter", "constant", "tuple", "get-tuple-element", "bitcast"})

# Ops whose weight is Sigma(operand elements) + Sigma(output elements), when UNFUSED
# (i.e. this literal op name appears directly in a body, not wrapped in a `fusion`).
_OPERAND_PLUS_OUTPUT_OPS = frozenset(
  {"reduce", "reduce-window", "dot", "convolution", "gather", "sort"}
)

# Ops that descend into another computation instead of being weighed directly.
_DESCEND_OPS = frozenset({"call", "conditional", "while"})


@dataclass
class OpWeight:
  """Per-instruction weight for the body_operand_elements metric."""

  op_name: str
  operand_elements: int
  output_elements: int
  name: str = ""
  scope: str | None = None

  @property
  def total_weight(self) -> int:
    """Total weight: operand + output elements."""
    return self.operand_elements + self.output_elements

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "op": self.op_name,
      "operand_elements": self.operand_elements,
      "output_elements": self.output_elements,
      "total_weight": self.total_weight,
      "scope": self.scope,
    }


@dataclass
class WhileLoopAnalysis:
  """Analysis of a single while loop's body."""

  name: str
  body_operand_elements: int
  has_nested_while: bool = False
  top_10_instructions: list[dict[str, Any]] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "body_operand_elements": self.body_operand_elements,
      "has_nested_while": self.has_nested_while,
      "top_10_instructions": self.top_10_instructions,
    }


# ---------------------------------------------------------------------------------------
# Shape parsing
# ---------------------------------------------------------------------------------------

_DIMS_RE = re.compile(r"\[(.*?)\]")


def _parse_shape(shape_str: str) -> tuple[int, ...]:
  """Parse an array shape's bracket group, e.g. 'f32[16,256]{1,0}' -> (16, 256).

  Scalars ('f32[]') and non-array types return (). Not valid on tuple types (callers
  route those through `_count_elements`'s tuple branch instead).
  """
  match = _DIMS_RE.search(shape_str)
  if not match:
    return ()
  dims_str = match.group(1)
  if not dims_str:
    return ()
  return tuple(int(d) for d in dims_str.split(","))


def _split_top_level(s: str, sep: str = ",") -> list[str]:
  """Split `s` on `sep` at bracket/paren/brace depth 0 only."""
  parts: list[str] = []
  depth = 0
  cur: list[str] = []
  for ch in s:
    if ch in "([{":
      depth += 1
    elif ch in ")]}":
      depth -= 1
    if ch == sep and depth == 0:
      parts.append("".join(cur))
      cur = []
    else:
      cur.append(ch)
  parts.append("".join(cur))
  return parts


def _count_elements(shape_str: str) -> int:
  """Count total elements in a shape string. Scalars return 1; tuples sum recursively."""
  s = shape_str.strip()
  if s.startswith("("):
    inner = s[1:-1] if s.endswith(")") else s[1:]
    inner = inner.strip()
    if not inner:
      return 0
    return sum(_count_elements(p) for p in _split_top_level(inner) if p.strip())
  dims = _parse_shape(s)
  if not dims:
    return 1
  result = 1
  for d in dims:
    result *= d
  return result


# ---------------------------------------------------------------------------------------
# Computation / instruction parsing
# ---------------------------------------------------------------------------------------


@dataclass
class Instruction:
  name: str
  op: str
  shape: str
  args: list[str]
  to_apply: str | None = None
  calls: str | None = None
  branches: list[str] = field(default_factory=list)
  condition: str | None = None
  body: str | None = None


_LEAD_RE = re.compile(r"^\s*(?:ROOT\s+)?%([\w.\-]+)\s*=\s*(.*)$")
_HEADER_RE = re.compile(r"^(?:ENTRY\s+)?%([\w.\-]+)\s*\(")
_OPNAME_RE = re.compile(r"^([A-Za-z][\w-]*)\(")
_TO_APPLY_RE = re.compile(r"\bto_apply=%([\w.\-]+)")
_CALLS_RE = re.compile(r"\bcalls=%([\w.\-]+)")
_BRANCHES_RE = re.compile(r"\bbranch_computations=\{([^}]*)\}")
_CONDITION_RE = re.compile(r"\bcondition=%([\w.\-]+)")
_BODY_RE = re.compile(r"\bbody=%([\w.\-]+)")


def _split_computations(hlo_text: str) -> dict[str, list[str]]:
  """Split HLO text into {computation_name: [body_lines]}.

  Computations are printed at the top level in the installs this was resolved against
  (never lexically nested): a header line ends with `{`, the body is every line until a
  bare `}` line. See module docstring.
  """
  comps: dict[str, list[str]] = {}
  name: str | None = None
  buf: list[str] = []
  for line in hlo_text.splitlines():
    stripped = line.strip()
    if name is None:
      if stripped.endswith("{") and "->" in stripped and _HEADER_RE.match(stripped):
        name = _HEADER_RE.match(stripped).group(1)  # type: ignore[union-attr]
        buf = []
      continue
    if stripped == "}":
      comps[name] = buf
      name = None
      buf = []
      continue
    buf.append(line)
  return comps


def _split_type_and_rest(remainder: str) -> tuple[str, str] | None:
  """Split `TYPE OPNAME(ARGS)...` at the depth-0 space preceding `OPNAME(`."""
  depth = 0
  for i, ch in enumerate(remainder):
    if ch in "([{":
      depth += 1
    elif ch in ")]}":
      depth -= 1
    elif ch == " " and depth == 0:
      rest = remainder[i + 1 :]
      if _OPNAME_RE.match(rest):
        return remainder[:i], rest
  return None


def _parse_instruction_line(line: str) -> Instruction | None:
  lead = _LEAD_RE.match(line)
  if not lead:
    return None
  name, remainder = lead.group(1), lead.group(2)
  split = _split_type_and_rest(remainder)
  if split is None:
    return None
  shape, rest = split
  op_match = _OPNAME_RE.match(rest)
  if not op_match:
    return None
  op = op_match.group(1)
  after_paren = rest[op_match.end() :]
  depth = 1
  close_idx = None
  for i, ch in enumerate(after_paren):
    if ch == "(":
      depth += 1
    elif ch == ")":
      depth -= 1
      if depth == 0:
        close_idx = i
        break
  if close_idx is None:
    return None
  args_str = after_paren[:close_idx]
  tail = after_paren[close_idx + 1 :]
  args = [a.strip().lstrip("%") for a in _split_top_level(args_str) if a.strip()]

  to_apply_m = _TO_APPLY_RE.search(tail)
  calls_m = _CALLS_RE.search(tail)
  branches_m = _BRANCHES_RE.search(tail)
  condition_m = _CONDITION_RE.search(tail)
  body_m = _BODY_RE.search(tail)
  branches = (
    [b.strip().lstrip("%") for b in branches_m.group(1).split(",") if b.strip()]
    if branches_m
    else []
  )

  return Instruction(
    name=name,
    op=op,
    shape=shape.strip(),
    args=args,
    to_apply=to_apply_m.group(1) if to_apply_m else None,
    calls=calls_m.group(1) if calls_m else None,
    branches=branches,
    condition=condition_m.group(1) if condition_m else None,
    body=body_m.group(1) if body_m else None,
  )


def _parse_instructions(lines: list[str]) -> dict[str, Instruction]:
  out: dict[str, Instruction] = {}
  for line in lines:
    instr = _parse_instruction_line(line)
    if instr is not None:
      out[instr.name] = instr
  return out


# ---------------------------------------------------------------------------------------
# Weighing
# ---------------------------------------------------------------------------------------


def _leaf_weight(op: str, instr: Instruction, symtab: dict[str, str]) -> tuple[int, int]:
  """(operand_elements, output_elements) for a non-descending instruction."""

  def elems(idx: int) -> int:
    if idx >= len(instr.args):
      return 0
    shape = symtab.get(instr.args[idx])
    return _count_elements(shape) if shape else 0

  output_elements = _count_elements(instr.shape)

  if op == "fusion":
    operand_elements = sum(elems(i) for i in range(len(instr.args)))
    return operand_elements, output_elements
  if op in _OPERAND_PLUS_OUTPUT_OPS:
    operand_elements = sum(elems(i) for i in range(len(instr.args)))
    return operand_elements, output_elements
  if op == "dynamic-update-slice":
    # operand(0)=array being updated, operand(1)=update, operand(2+)=start indices
    # (scalars). Weight is the update operand only -- never the full output.
    return elems(1), 0
  if op == "scatter":
    # Canonical XLA scatter operand order: operand, scatter_indices, updates.
    return elems(1) + elems(2), 0
  if op == "dynamic-slice":
    return 0, output_elements
  # Any other unfused op (add, multiply, select, compare, copy, broadcast, convert, ...):
  # output elements only. `copy` falls through here deliberately (spec: "counted").
  return 0, output_elements


class HLOLoopAnalyzer:
  """Parses HLO text and analyzes while-loop body costs."""

  def __init__(self, hlo_text: str, known_scope_labels: frozenset[str] = frozenset()):
    self.hlo_text = hlo_text
    self.while_loops: list[WhileLoopAnalysis] = []
    self._computations = _split_computations(hlo_text)
    self._parsed: dict[str, dict[str, Instruction]] = {
      name: _parse_instructions(lines) for name, lines in self._computations.items()
    }
    self._symtabs: dict[str, dict[str, str]] = {
      name: {i.name: i.shape for i in instrs.values()} for name, instrs in self._parsed.items()
    }
    self._memo: dict[str, tuple[int, bool, list[OpWeight]]] = {}
    self._scope_map: dict[str, str | None] = {}
    if known_scope_labels and scope_map_from_hlo_text is not None:
      try:
        self._scope_map = scope_map_from_hlo_text(hlo_text, known_scope_labels)
      except Exception:  # noqa: BLE001 -- best-effort enrichment only
        self._scope_map = {}

  def _computation_weight(self, comp_name: str) -> tuple[int, bool, list[OpWeight]]:
    if comp_name in self._memo:
      return self._memo[comp_name]
    instrs = self._parsed.get(comp_name)
    if instrs is None:
      logger.warning("Could not resolve computation %r", comp_name)
      return 0, False, []
    self._memo[comp_name] = (0, False, [])  # cycle guard
    symtab = self._symtabs.get(comp_name, {})
    total = 0
    nested_while = False
    entries: list[OpWeight] = []

    for instr in instrs.values():
      op = instr.op
      if op in EXCLUDED_OPS:
        continue

      if op == "call" and instr.to_apply:
        w, nw, sub = self._computation_weight(instr.to_apply)
        total += w
        nested_while = nested_while or nw
        entries.extend(sub)
        continue

      if op == "conditional" and instr.branches:
        branch_results = [self._computation_weight(b) for b in instr.branches]
        best = max(branch_results, key=lambda r: r[0], default=(0, False, []))
        total += best[0]
        nested_while = nested_while or any(r[1] for r in branch_results)
        entries.extend(best[2])
        continue

      if op == "while" and instr.body:
        w, _nw, sub = self._computation_weight(instr.body)
        total += w
        nested_while = True
        entries.extend(sub)
        continue

      operand_elements, output_elements = _leaf_weight(op, instr, symtab)
      weight = operand_elements + output_elements
      total += weight
      entries.append(
        OpWeight(
          op_name=op,
          operand_elements=operand_elements,
          output_elements=output_elements,
          name=instr.name,
          scope=self._scope_map.get(instr.name),
        )
      )

    self._memo[comp_name] = (total, nested_while, entries)
    return self._memo[comp_name]

  def analyze(self) -> list[WhileLoopAnalysis]:
    """Find every `while` instruction (in every parsed computation) and weigh its body."""
    self.while_loops = []
    for instrs in self._parsed.values():
      for instr in instrs.values():
        if instr.op != "while" or not instr.body:
          continue
        weight, nested_while, entries = self._computation_weight(instr.body)
        top_10 = sorted(entries, key=lambda e: e.total_weight, reverse=True)[:10]
        self.while_loops.append(
          WhileLoopAnalysis(
            name=instr.name,
            body_operand_elements=weight,
            has_nested_while=nested_while,
            top_10_instructions=[e.to_dict() for e in top_10],
          )
        )
    return self.while_loops


def compute_scaling_factor(m_2l: int | float | None, m_l: int | float | None) -> float | None:
  """Compute alpha = log2(m_2l / m_l), guarded against a null/non-positive input."""
  if m_l is None or m_2l is None or m_l <= 0 or m_2l <= 0:
    return None
  return math.log2(m_2l / m_l)


def analyze_hlo_file(
  hlo_text: str, known_scope_labels: frozenset[str] = frozenset()
) -> dict[str, Any]:
  """Analyze HLO text and return structured results."""
  analyzer = HLOLoopAnalyzer(hlo_text, known_scope_labels=known_scope_labels)
  loops = analyzer.analyze()
  return {
    "num_while_loops": len(loops),
    "loops": [loop.to_dict() for loop in loops],
  }


def primary_while_loop(loops: list[WhileLoopAnalysis]) -> WhileLoopAnalysis | None:
  """Pick the "primary" while loop out of possibly several found (max by weight)."""
  if not loops:
    return None
  return max(loops, key=lambda loop: loop.body_operand_elements)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Analyze HLO loop-body costs")
  parser.add_argument("--hlo-text", type=str, help="HLO text to analyze")
  parser.add_argument("--hlo-file", type=Path, help="File containing HLO text")

  args = parser.parse_args()

  if args.hlo_file:
    hlo_text_arg = args.hlo_file.read_text()
  elif args.hlo_text:
    hlo_text_arg = args.hlo_text
  else:
    raise ValueError("Either --hlo-text or --hlo-file required")

  result = analyze_hlo_file(hlo_text_arg)
  print(json.dumps(result, indent=2))
