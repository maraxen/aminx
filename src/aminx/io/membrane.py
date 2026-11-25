"""Host-side membrane residue list parser and physics_features constructor (Clause 2).

Implements the reference MembraneMPNN label construction protocol (clause
`membrane_label_construction`, run.py:252-277, model_utils.py:1572-1576):
- Parses buried and interface residue identifiers (e.g. "A1 A2 A5" or {"A1", "A2"}).
- Assigns 3-class membrane topology labels:
    0: neither / non-membrane
    1: interface (lipid head-group interface)
    2: buried (lipid core hydrophobic pore)
  Preserves the upstream quirk: residues with both buried and interface flags set
  evaluate to 0 (2 * 1 * (1 - 1) + 1 * 1 * (1 - 1) == 0).
- Supports global membrane topology label broadcast (--global_transmembrane_label in {0, 1, 2}).
- Produces (L, 3) one-hot float32 physics_features for consumption by Aminx membrane models.

Library-only: these helpers are not wired into any run path (no runner or spec calls them).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import jax
import jax.numpy as jnp


def parse_residue_spec(
  spec: str | Collection[str] | Collection[int] | None,
) -> set[str]:
  """Parse a residue specification into a set of normalized string identifiers.

  Accepts:
    - None or empty string -> empty set
    - Space-separated or comma-separated string (e.g. "A1 A2 A5" or "A1,A2,A5")
    - Iterable of strings (e.g. ["A1", "A2", "A5"] or {"A1", "A2"})
    - Iterable of ints (e.g. [0, 1, 4] -> {"0", "1", "4"})

  Args:
    spec: Residue specification.

  Returns:
    Set of string residue identifiers.
  """
  if spec is None:
    return set()

  if isinstance(spec, str):
    cleaned = spec.replace(",", " ").strip()
    if not cleaned:
      return set()
    return set(cleaned.split())

  return {str(item).strip() for item in spec if str(item).strip()}


def construct_membrane_labels(
  encoded_residues: Sequence[str] | Sequence[int] | int,
  buried_residues: str | Collection[str] | Collection[int] | None = None,
  interface_residues: str | Collection[str] | Collection[int] | None = None,
  global_label: int | None = None,
) -> jax.Array:
  """Construct per-residue membrane topology labels in {0, 1, 2}.

  Formula:
      labels = 2 * buried * (1 - interface) + 1 * interface * (1 - buried)
  When both buried and interface are 1, label resolves to 0.

  Args:
    encoded_residues: Sequence of residue identifiers (e.g. ["A1", "A2", ...])
      or total sequence length L as an integer.
    buried_residues: Residues in the buried membrane region (label 2).
    interface_residues: Residues in the interface membrane region (label 1).
    global_label: If set, broadcasts a constant label in {0, 1, 2} to all residues.

  Returns:
    (L,) int32 array with per-residue membrane labels in {0, 1, 2}.

  Raises:
    ValueError: If global_label is not in {0, 1, 2}.
  """
  if isinstance(encoded_residues, int):
    l_len = encoded_residues
    res_list = [str(i) for i in range(l_len)]
  else:
    res_list = [str(item) for item in encoded_residues]
    l_len = len(res_list)

  if global_label is not None:
    if global_label not in (0, 1, 2):
      msg = f"global_label must be in {{0, 1, 2}}, got {global_label!r}"
      raise ValueError(msg)
    return jnp.full((l_len,), global_label, dtype=jnp.int32)

  buried_set = parse_residue_spec(buried_residues)
  interface_set = parse_residue_spec(interface_residues)

  buried_flags = [1 if r in buried_set else 0 for r in res_list]
  interface_flags = [1 if r in interface_set else 0 for r in res_list]

  buried_arr = jnp.array(buried_flags, dtype=jnp.int32)
  interface_arr = jnp.array(interface_flags, dtype=jnp.int32)

  return 2 * buried_arr * (1 - interface_arr) + 1 * interface_arr * (1 - buried_arr)


def construct_membrane_physics_features(
  encoded_residues: Sequence[str] | Sequence[int] | int,
  buried_residues: str | Collection[str] | Collection[int] | None = None,
  interface_residues: str | Collection[str] | Collection[int] | None = None,
  global_label: int | None = None,
) -> jax.Array:
  """Construct (L, 3) one-hot physics_features array for MembraneMPNN.

  Matches reference node_embedding inputs (model_utils.py:1572-1576):
      C_1hot = F.one_hot(membrane_per_residue_labels, 3).float()

  Args:
    encoded_residues: Sequence of residue identifiers or sequence length.
    buried_residues: Residues in the buried membrane region.
    interface_residues: Residues in the interface membrane region.
    global_label: Constant global membrane label in {0, 1, 2}.

  Returns:
    (L, 3) float32 one-hot physics features array.
  """
  labels = construct_membrane_labels(
    encoded_residues=encoded_residues,
    buried_residues=buried_residues,
    interface_residues=interface_residues,
    global_label=global_label,
  )
  return jax.nn.one_hot(labels, 3, dtype=jnp.float32)
