"""Fixed-shape functions of the pH block descent, for export to ONNX and a JS driver loop (ADR decision 11).

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §54. The browser does not run the whole descent as one
graph: control flow in a graph forces device copies on WebGPU and the Potts export abandoned that shape. JS owns the
sweep, the convergence test and the injected uniforms, and calls these loop-free functions:

- ``make_visit`` -> ``visit``: one block visit, i.e. the block objective (:func:`ph_descent.block_objective`) and the
  commit of its minimiser (``temperature <= 0``) or of an inverse-CDF draw (``temperature > 0``). Returns the digits.
- ``zblock`` -> :func:`ph_descent.block_zstats`: one block's z-scale statistics; ``pool`` combines them.
- ``field_at`` -> :func:`ph_potentials.candidate_energies_at`: conditional energies at a chunk of positions. The
  full-chain ``candidate_energies`` is NOT exported: it sums with a repeated-index scatter-add, which ONNX Runtime does
  not make thread-safe (see ``ph_potentials.block_stability_potentials``).

Every function takes arrays only (no Python scalars that vary at run time) and returns tuples of arrays, with no
boolean outputs. The block size ``B`` and pin count ``P`` are shapes, so a graph is exported per ``(bucket, B, P)``;
the repetitive-window weight and radius are static (a config constant), as upstream's are per criteria.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from aminx.families.protonpotts_mpnn import ph_descent, ph_potentials

if TYPE_CHECKING:
  from collections.abc import Callable

  from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig


def make_visit(config: PHDesignConfig) -> Callable[..., tuple[Array]]:
  """The ``visit`` function for one config (its repetitive-window constants are closed over)."""

  def visit(
    table: Float[Array, "L K V V"],
    e_idx: Int[Array, "L K"],
    seq: Int[Array, " L"],
    block: Int[Array, " B"],
    block_valid: Bool[Array, " B"],
    pin_positions: Int[Array, " P"],
    pin_prot: Int[Array, " P"],
    pin_dep: Int[Array, "P D"],
    pin_dep_valid: Bool[Array, "P D"],
    valid_tokens: Bool[Array, " V"],
    rep_mask: Float[Array, " V"],
    residue_number: Int[Array, " L"],
    wh: Float[Array, " 1"],
    wsel: Float[Array, " 1"],
    uniform: Float[Array, " 1"],
    temperature: Float[Array, " 1"],
  ) -> tuple[Array]:
    n_block = block.shape[0]
    vocab = table.shape[-1]
    flat = ph_descent.block_objective(
      table, e_idx, seq, block, block_valid, pin_positions, pin_prot, pin_dep, pin_dep_valid,
      valid_tokens, rep_mask, residue_number, config=config, wh=wh[0], wsel=wsel[0],
    )
    # select_joint at temperature > 0, with the temperature a runtime input: the minimiser otherwise.
    shifted = flat - jnp.min(flat)
    probs = jnp.exp(-shifted / jnp.maximum(temperature[0], jnp.asarray(1e-30, dtype=flat.dtype)))
    cdf = jnp.cumsum(probs)
    target = uniform[0] * cdf[-1]
    drawn = jnp.minimum(jnp.sum(cdf <= target).astype(jnp.int32), jnp.int32(flat.shape[0] - 1))
    choice = jnp.where(temperature[0] > 0, drawn, jnp.argmin(flat).astype(jnp.int32))
    return (ph_descent._digits(choice, n_block, vocab),)  # noqa: SLF001

  return visit


def zblock(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  block: Int[Array, " B"],
  block_valid: Bool[Array, " B"],
  pin_positions: Int[Array, " P"],
  pin_prot: Int[Array, " P"],
  pin_dep: Int[Array, "P D"],
  pin_dep_valid: Bool[Array, "P D"],
  valid_tokens: Bool[Array, " V"],
) -> tuple[Array, Array, Array, Array]:
  """``(var_h, has_h, var_s, has_s)`` of one block, each shape ``(1,)``; the flags are int32. ``seq`` carries the pins."""
  var_h, has_h, var_s, has_s = ph_descent.block_zstats(
    table, e_idx, seq, block, block_valid, pin_positions, pin_prot, pin_dep, pin_dep_valid, valid_tokens,
  )
  return var_h[None], has_h.astype(jnp.int32)[None], var_s[None], has_s.astype(jnp.int32)[None]


def pool(
  var_h: Float[Array, " N"],
  has_h: Int[Array, " N"],
  var_s: Float[Array, " N"],
  has_s: Int[Array, " N"],
  w_h: Float[Array, " 1"],
  w_s: Float[Array, " 1"],
) -> tuple[Array, Array, Array]:
  """``(zscales (3,), wh (1,), wsel (1,))`` from the per-block statistics, padded with ``has == 0``.

  ``w_h`` and ``w_s`` are ``(1 - lambda)`` and ``lambda``, rounded to the run dtype on the host exactly as
  ``block_descent`` forms them (a Python float against a float32 array), so ``wh = w_h / sdH`` is the same operation.
  """
  dtype = var_h.dtype
  sd_h = ph_descent._pool(var_h, has_h > 0, dtype)  # noqa: SLF001
  sd_s = ph_descent._pool(var_s, has_s > 0, dtype)  # noqa: SLF001
  zscales = jnp.stack([sd_h, sd_s, jnp.asarray(1.0, dtype=dtype)])
  return zscales, (w_h[0] / sd_h)[None], (w_s[0] / sd_s)[None]


def field_at(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  positions: Int[Array, " N"],
) -> tuple[Float[Array, "N V"]]:
  """Conditional energies at ``positions`` (see :func:`ph_potentials.candidate_energies_at`)."""
  return (ph_potentials.candidate_energies_at(table, e_idx, seq, positions),)
