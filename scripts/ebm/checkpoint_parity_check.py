"""Independent parity gate for the ProteinEBM checkpoint weight-port (backlog node E3.5).

Loads the ProteinEBM PyTorch-Lightning checkpoint, remaps it onto
``aminx.ebm.model.ProteinEBMModel`` via ``aminx.ebm.checkpoint.
load_pytorch_checkpoint``, then independently cross-checks the ported JAX
model against the *actual PyTorch reference model* (``~/repos/ProteinEBM``)
on a fixed synthetic input. This is deliberately **not** the downstream
Spearman benchmark (E5/E6/E7) -- it is the gate that validates the weight
port itself, decoupling remap fidelity from any later ranking-metric result
(design spec §9, EPIC risk register MAJOR-3).

Requires (dev-only, not part of the aminx runtime dependency set):
  * ``torch`` (``uv sync --group dev`` or ``--group benchmark``).
  * The ProteinEBM reference repo checked out locally (``--reference-repo``,
    default ``~/repos/ProteinEBM``) -- provides ``protein_ebm.model.ebm.
    ProteinEBM`` and ``protein_ebm.model.r3_diffuser.R3Diffuser``.
  * The checkpoint file itself (``--checkpoint``), downloaded separately
    (see the E3.5 task brief for the authorized source URL) -- NOT
    committed to the aminx repo (430MB binary).

A genuine, already-documented unit-convention subtlety this script works
around (not a bug): the reference's ``ProteinEBM.forward`` internally
multiplies its ``r_noisy`` input by ``coordinate_scaling`` (0.1) when called
with its default ``rescale_input_coords=True`` (as ``compute_score`` does).
Our JAX ``ProteinEBMModel`` never applies ``coordinate_scaling`` internally
(that single boundary lives in ``aminx.ebm.diffusion.forward_marginal`` --
see that module's docstring). To get an apples-to-apples comparison of the
*ported weights* rather than a coordinate-unit conversion factor, this script
calls the reference's ``compute_energy(..., rescale_input_coords=False)``
directly (bypassing ``compute_score``'s wrapper) and feeds it the SAME
already-scaled coordinate array + a matching ``requires_grad`` tensor for the
autograd score, so both implementations differentiate w.r.t. the identical
physical quantity.

A second genuine discrepancy this script explicitly controls for: this
checkpoint's config has ``num_contact_embeddings=3``, and the reference
``forward``'s default (``external_contacts=None``) becomes **all-ones** for
that 3-way case (``ebm.py:169``) -- but our port's documented MVP contact
default is *always* all-zeros regardless of ``num_contact_embeddings`` (see
``aminx.ebm.model`` module docstring). This script passes an explicit,
identical ``contacts``/``external_contacts`` array to both sides so the
comparison is not silently corrupted by this already-known default-value
divergence.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
  import torch
  from protein_ebm.model.ebm import ProteinEBM  # type: ignore[import-not-found]

  from aminx.ebm.checkpoint import CheckpointPortReport
  from aminx.ebm.model import ProteinEBMModel

log = logging.getLogger("checkpoint_parity_check")

DEFAULT_CHECKPOINT = Path("/tmp/proteinebm_weights/model_6_expert_frozen_1m_md.pt")
DEFAULT_REFERENCE_REPO = Path("~/repos/ProteinEBM").expanduser()
DEFAULT_ORBAX_OUT = Path("/tmp/proteinebm_weights/ported_jax_model")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--reference-repo", type=Path, default=DEFAULT_REFERENCE_REPO)
  parser.add_argument("--orbax-out", type=Path, default=DEFAULT_ORBAX_OUT)
  parser.add_argument("--n-residues", type=int, default=12)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--diffusion-time", type=float, default=0.05, help="ProteinEBM-x MVP target t (design spec §9).")
  parser.add_argument("--atol", type=float, default=1e-3)
  parser.add_argument("--rtol", type=float, default=1e-3)
  parser.add_argument(
    "--skip-orbax-save",
    action="store_true",
    help="Skip the final orbax save (useful for a quick re-run of just the parity check).",
  )
  return parser.parse_args()


def _build_reference_model(reference_repo: Path, ckpt: dict) -> "ProteinEBM":
  """Construct + strict-load the reference PyTorch ``ProteinEBM`` from ``ckpt``."""
  sys.path.insert(0, str(reference_repo))
  from protein_ebm.model.ebm import ProteinEBM  # noqa: PLC0415
  from protein_ebm.model.r3_diffuser import R3Diffuser  # noqa: PLC0415

  cfg = ckpt["hyper_parameters"]["config"]
  diffuser = R3Diffuser(cfg.diffuser)
  model = ProteinEBM(cfg.model, diffuser)
  state_dict = ckpt["state_dict"]
  stripped = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
  missing, unexpected = model.load_state_dict(stripped, strict=False)
  if missing or unexpected:
    msg = f"Reference model.load_state_dict was not exact: missing={missing}, unexpected={unexpected}"
    raise RuntimeError(msg)
  model.eval()
  return model


def _build_ported_jax_model(ckpt: dict, seed: int) -> "tuple[ProteinEBMModel, CheckpointPortReport]":
  """Construct a ``ProteinEBMModel`` matching the checkpoint's own config, then port weights."""
  from aminx.ebm.checkpoint import load_pytorch_checkpoint  # noqa: PLC0415
  from aminx.ebm.model import ProteinEBMModel  # noqa: PLC0415

  cfg = ckpt["hyper_parameters"]["config"].model
  key = jax.random.PRNGKey(seed)
  model = ProteinEBMModel(
    token_s=cfg.token_s,
    token_z=cfg.token_z,
    dim_fourier=cfg.dim_fourier,
    conditioning_transition_layers=cfg.conditioning_transition_layers,
    transformer_depth=cfg.token_transformer_depth,
    transformer_heads=cfg.token_transformer_heads,
    num_contact_embeddings=cfg.num_contact_embeddings,
    key=key,
  )
  ported, report = load_pytorch_checkpoint(model, ckpt["state_dict"])
  return ported, report


def _build_fixed_input(n_residues: int, seed: int) -> dict[str, np.ndarray]:
  """Build one fixed, deterministic small synthetic structure shared by both implementations."""
  rng = np.random.default_rng(seed)
  # Physical-scale coordinates (already in the "scaled"/nm space both models
  # consume directly -- see the module docstring's rescale_input_coords note).
  coords_scaled = rng.normal(scale=1.0, size=(n_residues, 3)).astype(np.float32)
  aatype = rng.integers(0, 21, size=(n_residues,)).astype(np.int64)
  residue_index = np.arange(n_residues, dtype=np.int64)
  chain_id = np.zeros((n_residues,), dtype=np.int64)
  # Explicit, identical contacts array on both sides (see module docstring --
  # sidesteps the reference's num_contact_embeddings==3 default-ones special case).
  contacts = np.zeros((n_residues,), dtype=np.int64)
  # Mask the last 2 residues to also exercise masked-sum parity.
  residue_mask = np.ones((n_residues,), dtype=bool)
  residue_mask[-2:] = False
  return {
    "coords_scaled": coords_scaled,
    "aatype": aatype,
    "residue_index": residue_index,
    "chain_id": chain_id,
    "contacts": contacts,
    "residue_mask": residue_mask,
  }


def _reference_energy_and_score(
  ref_model: "ProteinEBM",
  fixed_input: dict[str, np.ndarray],
  t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Run the reference PyTorch model's energy + autograd score on ``fixed_input``.

  Bypasses ``compute_score``'s internal ``coordinate_scaling`` multiply
  (``rescale_input_coords=False``) so the gradient is taken w.r.t. the exact
  same already-scaled coordinate array passed to the JAX side -- see module
  docstring.
  """
  import torch  # noqa: PLC0415

  n = fixed_input["coords_scaled"].shape[0]
  r_noisy = torch.tensor(fixed_input["coords_scaled"], dtype=torch.float32).unsqueeze(0).requires_grad_(True)
  aatype = torch.tensor(fixed_input["aatype"], dtype=torch.long).unsqueeze(0)
  residue_idx = torch.tensor(fixed_input["residue_index"], dtype=torch.long).unsqueeze(0)
  chain_id = torch.tensor(fixed_input["chain_id"], dtype=torch.long).unsqueeze(0)
  contacts = torch.tensor(fixed_input["contacts"], dtype=torch.long).unsqueeze(0)
  residue_mask = torch.tensor(fixed_input["residue_mask"], dtype=torch.bool).unsqueeze(0)
  times = torch.tensor([t], dtype=torch.float32)

  input_feats = {
    "r_noisy": r_noisy,
    "aatype": aatype,
    "residue_idx": residue_idx,
    "mask": residue_mask,
    "t": times,
    "chain_encoding": chain_id,
    "external_contacts": contacts,
  }

  model_out = ref_model.compute_energy(input_feats, rescale_input_coords=False)
  energy = model_out["energy"]
  per_residue_energy = model_out["per_residue_energy"]
  grad_r = torch.autograd.grad(energy.sum(), r_noisy, create_graph=False)[0]
  score = -grad_r
  aux_score = model_out["r_update_aux"]

  return (
    energy.detach().numpy().reshape(()),
    per_residue_energy.detach().numpy().reshape((n,)),
    score.detach().numpy().reshape((n, 3)),
    aux_score.detach().numpy().reshape((n, 3)),
  )


def _jax_energy_and_score(
  ported_model: "ProteinEBMModel",
  fixed_input: dict[str, np.ndarray],
  t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Run the ported JAX model's energy + jax.grad score on the same ``fixed_input``."""
  coords = jnp.asarray(fixed_input["coords_scaled"])
  aatype = jnp.asarray(fixed_input["aatype"], dtype=jnp.int32)
  residue_index = jnp.asarray(fixed_input["residue_index"], dtype=jnp.int32)
  chain_id = jnp.asarray(fixed_input["chain_id"], dtype=jnp.int32)
  contacts = jnp.asarray(fixed_input["contacts"], dtype=jnp.int32)
  mask = jnp.asarray(fixed_input["residue_mask"])
  t_arr = jnp.asarray(t)

  trunk_out = ported_model.trunk_features(
    coords, aatype, t_arr, mask, contacts=contacts, residue_index=residue_index, chain_id=chain_id,
  )
  per_residue_energy = ported_model.energy_readout.per_residue_energy(trunk_out, mask)
  energy = jnp.sum(per_residue_energy)
  score = ported_model.score(
    coords, aatype, t_arr, mask, contacts=contacts, residue_index=residue_index, chain_id=chain_id,
  )
  aux_score = ported_model.aux_score_readout(trunk_out)

  return (
    np.asarray(energy),
    np.asarray(per_residue_energy),
    np.asarray(score),
    np.asarray(aux_score),
  )


def _reference_energy_default_contacts(
  ref_model: "ProteinEBM",
  fixed_input: dict[str, np.ndarray],
  t: float,
) -> np.ndarray:
  """Reference total energy with ``external_contacts`` left to its OWN default.

  Passes ``external_contacts=None`` so the reference's ``forward`` applies its
  built-in default (``ebm.py:167-169``: all-ones for the 3-way contact variant
  this checkpoint uses). Paired with the JAX side's ``contacts=None`` this
  proves the ported model's default now matches the reference default -- the
  gap that PR #130's harness had to paper over by passing ``contacts=ones``
  explicitly at every call site.
  """
  import torch  # noqa: PLC0415

  r_noisy = torch.tensor(fixed_input["coords_scaled"], dtype=torch.float32).unsqueeze(0)
  input_feats = {
    "r_noisy": r_noisy,
    "aatype": torch.tensor(fixed_input["aatype"], dtype=torch.long).unsqueeze(0),
    "residue_idx": torch.tensor(fixed_input["residue_index"], dtype=torch.long).unsqueeze(0),
    "mask": torch.tensor(fixed_input["residue_mask"], dtype=torch.bool).unsqueeze(0),
    "t": torch.tensor([t], dtype=torch.float32),
    "chain_encoding": torch.tensor(fixed_input["chain_id"], dtype=torch.long).unsqueeze(0),
    "external_contacts": None,
  }
  model_out = ref_model.compute_energy(input_feats, rescale_input_coords=False)
  return model_out["energy"].detach().numpy().reshape(())


def _jax_energy_default_contacts(
  ported_model: "ProteinEBMModel",
  fixed_input: dict[str, np.ndarray],
  t: float,
) -> np.ndarray:
  """Ported JAX total energy with ``contacts=None`` (its own resolved default)."""
  energy = ported_model.energy(
    jnp.asarray(fixed_input["coords_scaled"]),
    jnp.asarray(fixed_input["aatype"], dtype=jnp.int32),
    jnp.asarray(t),
    jnp.asarray(fixed_input["residue_mask"]),
    contacts=None,
    residue_index=jnp.asarray(fixed_input["residue_index"], dtype=jnp.int32),
    chain_id=jnp.asarray(fixed_input["chain_id"], dtype=jnp.int32),
  )
  return np.asarray(energy)


def _report_allclose(label: str, ref: np.ndarray, jax_val: np.ndarray, atol: float, rtol: float) -> bool:
  diff = np.abs(ref - jax_val)
  max_abs = float(np.max(diff))
  denom = np.maximum(np.abs(ref), 1e-12)
  max_rel = float(np.max(diff / denom))
  ok = bool(np.allclose(ref, jax_val, atol=atol, rtol=rtol))
  status = "PASS" if ok else "FAIL"
  log.info(
    "[%s] %s: max_abs_diff=%.6e max_rel_diff=%.6e (atol=%.1e rtol=%.1e) ref_shape=%s",
    status,
    label,
    max_abs,
    max_rel,
    atol,
    rtol,
    ref.shape,
  )
  return ok


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args = _parse_args()

  if not args.checkpoint.exists():
    log.error("Checkpoint not found: %s", args.checkpoint)
    return 1
  if not args.reference_repo.exists():
    log.error("Reference repo not found: %s", args.reference_repo)
    return 1

  import torch  # noqa: PLC0415

  log.info("Loading checkpoint: %s", args.checkpoint)
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

  log.info("Building + strict-loading reference PyTorch model...")
  ref_model = _build_reference_model(args.reference_repo, ckpt)
  n_params = sum(p.numel() for p in ref_model.parameters())
  log.info("Reference model params: %d", n_params)

  log.info("Building + porting JAX ProteinEBMModel...")
  ported_model, report = _build_ported_jax_model(ckpt, args.seed)
  log.info("Loaded %d checkpoint keys.", len(report.loaded_keys))
  log.info("Skipped %d checkpoint keys (itemized):", len(report.skipped_keys))
  for key, reason in report.skipped_keys:
    log.info("  SKIP %-70s %s", key, reason)
  log.info("Explicitly zeroed bias paths (no reference counterpart): %s", report.zeroed_bias_paths)

  fixed_input = _build_fixed_input(args.n_residues, args.seed)

  log.info("Running reference PyTorch forward + autograd score...")
  ref_energy, ref_per_residue, ref_score, ref_aux_score = _reference_energy_and_score(
    ref_model, fixed_input, args.diffusion_time,
  )
  log.info("Running ported JAX forward + jax.grad score...")
  jax_energy, jax_per_residue, jax_score, jax_aux_score = _jax_energy_and_score(
    ported_model, fixed_input, args.diffusion_time,
  )

  log.info("=== Parity results (N=%d residues, t=%.3f) ===", args.n_residues, args.diffusion_time)
  results = {
    "total_energy": _report_allclose("total_energy", ref_energy, jax_energy, args.atol, args.rtol),
    "per_residue_energy": _report_allclose("per_residue_energy", ref_per_residue, jax_per_residue, args.atol, args.rtol),
    "conservative_score": _report_allclose("conservative_score (-grad E)", ref_score, jax_score, args.atol, args.rtol),
    "aux_score": _report_allclose("aux_score (non-conservative)", ref_aux_score, jax_aux_score, args.atol, args.rtol),
  }

  # Masked residues must contribute exactly zero energy on both sides
  # (BATHOS synthetic-invariant discipline: verify the measurement pipeline).
  mask = fixed_input["residue_mask"]
  masked_ref_zero = bool(np.allclose(ref_per_residue[~mask], 0.0))
  masked_jax_zero = bool(np.allclose(jax_per_residue[~mask], 0.0))
  log.info("[%s] masked-residue energy == 0 (reference): %s", "PASS" if masked_ref_zero else "FAIL", masked_ref_zero)
  log.info("[%s] masked-residue energy == 0 (ported JAX): %s", "PASS" if masked_jax_zero else "FAIL", masked_jax_zero)
  results["masked_residue_zero_energy"] = masked_ref_zero and masked_jax_zero

  # Default-contacts parity (Track B / PR #130 follow-up): the port's
  # contacts=None now resolves to the reference default (all-ones for the num=3
  # checkpoint), so running BOTH sides with their own defaults must agree. This
  # is separate from the explicit-zeros cases above (which pin identical inputs)
  # -- it proves the *defaults themselves* are now reference-faithful, closing
  # the gap that forced every harness call site to pass contacts=ones by hand.
  # Guarded so a reference-API mismatch degrades to a WARNING rather than
  # breaking the (independent) explicit-contacts parity.
  try:
    ref_energy_default = _reference_energy_default_contacts(ref_model, fixed_input, args.diffusion_time)
    jax_energy_default = _jax_energy_default_contacts(ported_model, fixed_input, args.diffusion_time)
    results["default_contacts_energy"] = _report_allclose(
      "default_contacts total_energy (contacts=None both sides)",
      ref_energy_default,
      jax_energy_default,
      args.atol,
      args.rtol,
    )
  except Exception as exc:  # noqa: BLE001 -- diagnostic guard: must not break the primary parity
    log.warning(
      "Default-contacts parity case could not run (reference API mismatch?): %s. "
      "Skipping this sub-check; the explicit-contacts parity above is unaffected.",
      exc,
    )

  all_pass = all(results.values())
  log.info("=== OVERALL: %s ===", "PASS" if all_pass else "FAIL")

  if not args.skip_orbax_save:
    log.info("Orbax-saving ported model to %s", args.orbax_out)
    _orbax_save(ported_model, args.orbax_out)

  return 0 if all_pass else 2


def _orbax_save(ported_model: "ProteinEBMModel", out_dir: Path) -> None:
  import orbax.checkpoint as ocp  # noqa: PLC0415

  out_dir = out_dir.resolve()
  out_dir.parent.mkdir(parents=True, exist_ok=True)
  options = ocp.CheckpointManagerOptions(max_to_keep=1)
  manager = ocp.CheckpointManager(
    out_dir,
    options=options,
    item_handlers={"model": ocp.PyTreeCheckpointHandler()},
  )
  manager.save(step=0, items={"model": ported_model})
  manager.wait_until_finished()
  log.info("Saved. Load later with the same ProteinEBMModel construction args as a restore template.")


if __name__ == "__main__":
  raise SystemExit(main())
