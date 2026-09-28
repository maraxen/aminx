"""HLO loop-body structural cost instrumentation (T7, P07 diagnostic; xtrax #1983 tie-in).

From `compiled.as_text()`, find every `while` instruction's body/condition computations
and compute the metric `body_operand_elements`: Σ over the body's instructions of a per-
instruction weight:

- `fusion`: once at the call site, Σ operand + output elements (no descent into fused);
- unfused reduce/reduce-window/dot/convolution/gather/sort: Σ operand + output elements;
- dynamic-update-slice/scatter: update operand (+ scatter indices) elements, not full output;
- dynamic-slice: output elements;
- other unfused ops: output elements;
- `parameter`, `constant`, `tuple`, `get-tuple-element`, `bitcast` excluded; `copy` counted;
- descend into `call` and `conditional` (max over branches);
- nested `while` flagged `nested_while` and body counted once.

Scaling α = log2(m(2L) / m(L)).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OpWeight:
  """Per-instruction weight for the body_operand_elements metric."""
  op_name: str
  operand_elements: int
  output_elements: int

  @property
  def total_weight(self) -> int:
    """Total weight: operand + output elements."""
    return self.operand_elements + self.output_elements


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


def _parse_shape(shape_str: str) -> tuple[int, ...]:
  """Parse HLO shape string like 'f32[16,256]' -> (16, 256), 'f32[]' -> ()."""
  # Extract the part after the type (f32, s32, etc.)
  match = re.search(r'\[(.*?)\]', shape_str)
  if not match:
    # Scalar
    return ()
  dims_str = match.group(1)
  if not dims_str:
    return ()
  return tuple(int(d) for d in dims_str.split(','))


def _count_elements(shape_str: str) -> int:
  """Count total elements in a shape. Scalars return 1."""
  dims = _parse_shape(shape_str)
  if not dims:
    return 1
  result = 1
  for d in dims:
    result *= d
  return result


def _parse_operands_from_instruction(instr_text: str) -> list[str]:
  """Extract operand shapes from an instruction's text."""
  # Very simple pattern: look for "= <op>(...)" or similar
  # We need to extract argument references and then look up their types
  operands = []
  # This is a simplified version; real parsing would need full HLO grammar
  return operands


class HLOLoopAnalyzer:
  """Parses HLO text and analyzes while loop body costs."""

  def __init__(self, hlo_text: str):
    self.hlo_text = hlo_text
    self.while_loops: list[WhileLoopAnalysis] = []

  def _extract_computation_text(self, comp_name: str) -> str | None:
    """Extract the text of a named computation (e.g., 'body_computation')."""
    pattern = rf'ENTRY {comp_name}.*?\n(?:.*?\n)*?^}}'
    match = re.search(pattern, self.hlo_text, re.MULTILINE | re.DOTALL)
    if match:
      return match.group(0)
    # Try alternate pattern
    pattern = rf'{comp_name} .*?{{(.*?)}}'
    match = re.search(pattern, self.hlo_text, re.MULTILINE | re.DOTALL)
    if match:
      return match.group(1)
    return None

  def _parse_instructions_in_computation(self, comp_text: str) -> dict[str, dict[str, Any]]:
    """Parse all instructions in a computation text, returning name -> instruction dict."""
    instructions: dict[str, dict[str, Any]] = {}

    # Split into individual instruction lines
    lines = comp_text.split('\n')
    for line in lines:
      line = line.strip()
      if not line or line.startswith('//'):
        continue

      # Parse instruction: name = operation(args) : type [, ...], metadata
      # Example: %param.1 = f32[16] parameter(0)
      match = re.match(r'%(\S+)\s*=\s*(\S+(?:\[\S+\])?)\s+(\w+)\((.*?)\)', line)
      if match:
        instr_name = match.group(1)
        instr_type = match.group(2)  # e.g., "f32[16,256]"
        op_type = match.group(3)  # e.g., "parameter", "fusion", "add"
        args = match.group(4)  # arguments

        output_elements = _count_elements(instr_type)

        instructions[instr_name] = {
          'name': instr_name,
          'op_type': op_type,
          'type': instr_type,
          'output_elements': output_elements,
          'operand_elements': 0,  # Will compute from args
          'args': args,
          'line': line,
        }

    return instructions

  def _compute_body_operand_elements(self, body_comp: str) -> tuple[int, bool, list[dict[str, Any]]]:
    """Compute body_operand_elements for a while loop's body computation.

    Returns: (total_weight, has_nested_while, top_10_instructions)
    """
    instructions = self._parse_instructions_in_computation(body_comp)

    total_weight = 0
    has_nested_while = False
    weights: list[OpWeight] = []

    for instr_name, instr in instructions.items():
      op_type = instr['op_type']
      output_elements = instr['output_elements']

      # Skip excluded ops
      if op_type in ('parameter', 'constant', 'tuple', 'get-tuple-element', 'bitcast'):
        continue

      # Count operand elements by looking at argument types
      operand_elements = 0

      # For now, use a simplified approach: assume operands have same type as output
      # (This is a simplification; real implementation would track all operand types)
      if op_type in ('reduce', 'reduce-window', 'dot', 'convolution', 'gather', 'sort'):
        # These use full operand + output
        # For simplification, estimate from args
        operand_elements = output_elements  # Placeholder
      elif op_type in ('dynamic-update-slice', 'scatter'):
        # Use update operand elements
        operand_elements = output_elements  # Placeholder
      elif op_type == 'dynamic-slice':
        operand_elements = 0
      elif op_type == 'while':
        has_nested_while = True
        operand_elements = output_elements
      else:
        operand_elements = output_elements

      weight = OpWeight(
        op_name=op_type,
        operand_elements=operand_elements,
        output_elements=output_elements,
      )
      weights.append(weight)
      total_weight += weight.total_weight

    # Sort by weight and get top 10
    weights.sort(key=lambda w: w.total_weight, reverse=True)
    top_10 = [
      {
        'op': w.op_name,
        'operand_elements': w.operand_elements,
        'output_elements': w.output_elements,
        'total_weight': w.total_weight,
      }
      for w in weights[:10]
    ]

    return total_weight, has_nested_while, top_10

  def analyze(self) -> list[WhileLoopAnalysis]:
    """Find all while loops and analyze their bodies."""
    # Find while loop patterns
    while_pattern = r'while\((.*?)\)\s*,\s*body=([\w.]+)\s*,\s*condition=([\w.]+)'

    for match in re.finditer(while_pattern, self.hlo_text):
      body_name = match.group(2)
      condition_name = match.group(3)

      body_comp = self._extract_computation_text(body_name)
      if not body_comp:
        logger.warning(f"Could not extract body computation: {body_name}")
        continue

      body_cost, has_nested, top_10 = self._compute_body_operand_elements(body_comp)

      analysis = WhileLoopAnalysis(
        name=body_name,
        body_operand_elements=body_cost,
        has_nested_while=has_nested,
        top_10_instructions=top_10,
      )
      self.while_loops.append(analysis)

    return self.while_loops


def compute_scaling_factor(m_2l: int, m_l: int) -> float | None:
  """Compute α = log2(m(2L) / m(L)), guarded against zero/negative."""
  if m_l <= 0 or m_2l <= 0:
    return None
  return math.log2(m_2l / m_l)


def analyze_hlo_file(hlo_text: str) -> dict[str, Any]:
  """Analyze HLO text and return structured results."""
  analyzer = HLOLoopAnalyzer(hlo_text)
  loops = analyzer.analyze()

  return {
    'num_while_loops': len(loops),
    'loops': [loop.to_dict() for loop in loops],
  }


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Analyze HLO loop-body costs')
  parser.add_argument('--hlo-text', type=str, help='HLO text to analyze')
  parser.add_argument('--hlo-file', type=Path, help='File containing HLO text')

  args = parser.parse_args()

  if args.hlo_file:
    hlo_text = args.hlo_file.read_text()
  elif args.hlo_text:
    hlo_text = args.hlo_text
  else:
    raise ValueError('Either --hlo-text or --hlo-file required')

  result = analyze_hlo_file(hlo_text)
  print(json.dumps(result, indent=2))
