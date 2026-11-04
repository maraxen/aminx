"""Bathos parity vehicle and negative controls for Caliby against sealed upstream oracle dumps.

The upstream side is a sealed ``.npz`` written by ``dump_caliby_oracle.py`` from the real
ProteinDesignLab/caliby code and the real released checkpoints (see that file for how it is made).
This vehicle imports no torch and no upstream code. It loads the SAME checkpoint into aminx with
``CalibyDriver.load`` semantics (``_load_caliby_torch``), runs aminx on the same structure
(``tests/data/1ubq.pdb``) and diffs every stage against the dump:

  s0  aminx ``_prepare_structure`` vs the upstream featurized batch
  s1  neighbor graph (exact integer ``E_idx``), edge features, gap embedding, ``W_e``
  s2  every encoder layer          s3  every decoder layer
  s4  raw 32-state Potts fields and dense couplings
  s5  21-state aminx-alphabet fields/couplings, local conditional energies ``U_i``, energies of the
      wild type and of seeded alternative sequences
  s6  Gaussian-conformer ensemble with the upstream noise injected as conformer coordinates
  s7  driver end to end (``score:energy`` from the PDB) vs upstream ``score_samples`` U
  pk  (packer checkpoint) MPNN trunk and the first two sidechain-denoiser calls under the seeded prior

Network stages (s1 onward) are fed the upstream's own featurized coordinates, so a parsing
difference (s0) cannot hide or fake network parity; s7 then exercises the parser plus the network.
Mutants are aminx-side code perturbations applied by monkeypatching, and every one is diffed
against the same sealed dump. Nothing here compares aminx with aminx.

Sampler: the upstream Potts sampler (DLMC/500 sweeps/LCP, debt #3121) and the upstream Euler
sidechain sampler with self-conditioning (TBD-CALIBY) are NOT compared; aminx does not implement
those algorithms. That absence is recorded in the output, not hidden.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO_ROOT / "src", Path(__file__).resolve().parent):
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from artifact_key import stable_artifact_key  # noqa: E402

import aminx.families.caliby.driver as drv  # noqa: E402
import aminx.families.caliby.features as feat  # noqa: E402
import aminx.families.caliby.model as mdl  # noqa: E402
import aminx.families.caliby.potts as pts  # noqa: E402
import aminx.families.caliby.sidechain_diffusion as scn  # noqa: E402
from aminx.run.options import CalibyOptions  # noqa: E402

log = logging.getLogger("caliby_parity")

BANDS = {"float64": 1e-7, "float32": 1e-4}
PDB = _REPO_ROOT / "tests" / "data" / "1ubq.pdb"
# (checkpoint, dtype) arms. The packer is float32 only: its timestep embedder is an inherent float32
# computation upstream (the oracle refuses to build a float64 packer dump).
POTTS_ARMS = (
  ("caliby", "float64"),
  ("caliby", "float32"),
  ("soluble_caliby_v1", "float64"),
  ("soluble_caliby_v1", "float32"),
  ("caliby_distill", "float64"),
  ("caliby_distill", "float32"),
)
PACKER_ARMS = (("caliby_packer_010", "float32"),)

# Mutants: aminx-side perturbations. ``stage`` names the stage family each is expected to trip.
MUTANTS = (
  "rbf_range_shrunk",
  "virtual_cb_sign_flipped",
  "neighbors_farthest",
  "neighbor_gather_off_by_one",
  "gap_embedding_row_swapped",
  "masked_atoms_not_ca_filled",
  "potts_unsymmetrized",
  "alphabet_unpermuted",
  "energy_pair_term_doubled",
  "ensemble_sum_not_mean",
)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def maxabs(a: Any, b: Any) -> float:  # noqa: ANN401
  """Max |a-b| in float64; ``inf`` on a shape mismatch, ``nan`` if either side is non-finite."""
  x = np.asarray(a, dtype=np.float64)
  y = np.asarray(b, dtype=np.float64)
  if x.shape != y.shape:
    return math.inf
  if x.size == 0:
    return 0.0
  if not (np.isfinite(x).all() and np.isfinite(y).all()):
    return math.nan
  return float(np.max(np.abs(x - y)))


def mismatches(a: Any, b: Any) -> float:  # noqa: ANN401
  """Count of differing entries of two integer arrays (``inf`` on shape mismatch)."""
  x, y = np.asarray(a), np.asarray(b)
  if x.shape != y.shape:
    return math.inf
  return float(np.sum(x != y))


# ---------------------------------------------------------------------------------------------
# mutants (monkeypatches). Each returns a list of (owner, attribute, replacement).
# ---------------------------------------------------------------------------------------------
def _mutation_patches(name: str) -> list[tuple[Any, str, Any]]:  # noqa: C901, PLR0911
  orig_rbf = feat.compute_rbf_grid
  orig_knn = feat.compute_caliby_knn
  orig_cb = feat.make_virtual_cb
  orig_cat = mdl._cat_neighbors_nodes  # noqa: SLF001
  orig_sym = pts._symmetrize_J  # noqa: SLF001
  orig_slice = pts.slice_potts_params
  orig_energy = pts.compute_caliby_potts_energy
  orig_agg = pts.aggregate_ensemble_potts_params
  orig_tf_call = feat.CalibyTokenFeatures.__call__

  if name == "rbf_range_shrunk":
    return [(feat, "compute_rbf_grid", lambda dist, **kw: orig_rbf(dist, **{**kw, "d_max": 20.0}))]
  if name == "virtual_cb_sign_flipped":
    return [(feat, "make_virtual_cb", lambda n, ca, c: -orig_cb(n, ca, c))]
  if name == "neighbors_farthest":

    def knn(ca_xyz, ca_mask, top_k, eps=1e-6):  # noqa: ANN001, ANN202
      d, idx = orig_knn(ca_xyz, ca_mask, ca_xyz.shape[0], eps)
      return d[:, -top_k:], idx[:, -top_k:]

    return [(feat, "compute_caliby_knn", knn)]
  if name == "neighbor_gather_off_by_one":
    return [(mdl, "_cat_neighbors_nodes", lambda h_nodes, h_neighbors, e_idx: orig_cat(h_nodes, h_neighbors, jnp.roll(e_idx, 1, axis=1)))]
  if name == "masked_atoms_not_ca_filled":

    def tf_call(self, atom23_xyz, atom23_mask, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
      # Treat every atom slot as present: masked slots keep their raw (zeroed) coordinates.
      return orig_tf_call(self, atom23_xyz, jnp.ones_like(atom23_mask), *a, **k)

    return [(feat.CalibyTokenFeatures, "__call__", tf_call)]
  if name == "potts_unsymmetrized":
    return [(pts, "_symmetrize_J", lambda j_knn, e_idx, valid_edge, num_tokens, *, symmetric=True: orig_sym(j_knn, e_idx, valid_edge, num_tokens, symmetric=False))]
  if name == "alphabet_unpermuted":
    unperm = lambda h, j, *, to_aminx_alphabet=False: orig_slice(h, j, to_aminx_alphabet=False)  # noqa: E731
    return [(pts, "slice_potts_params", unperm), (drv, "slice_potts_params", unperm)]
  if name == "energy_pair_term_doubled":

    def energy(h_21, j_21, sequence, mask):  # noqa: ANN001, ANN202
      site = orig_energy(h_21, jnp.zeros_like(j_21), sequence, mask)
      return site + 2.0 * (orig_energy(h_21, j_21, sequence, mask) - site)

    return [(pts, "compute_caliby_potts_energy", energy), (drv, "compute_caliby_potts_energy", energy)]
  if name == "ensemble_sum_not_mean":

    def agg(h_stack, j_stack, **kw):  # noqa: ANN001, ANN003, ANN202
      h, j = orig_agg(h_stack, j_stack, **kw)
      return h * h_stack.shape[0], j * h_stack.shape[0]

    return [(pts, "aggregate_ensemble_potts_params", agg), (drv, "aggregate_ensemble_potts_params", agg)]
  return []


@contextlib.contextmanager
def apply_mutant(name: str | None) -> Iterator[None]:
  patches = _mutation_patches(name) if name else []
  saved = [(owner, attr, getattr(owner, attr)) for owner, attr, _ in patches]
  try:
    for owner, attr, repl in patches:
      setattr(owner, attr, repl)
    yield
  finally:
    for owner, attr, orig in saved:
      setattr(owner, attr, orig)


def _swap_gap_row(model: Any) -> Any:  # noqa: ANN401
  """Mutant ``gap_embedding_row_swapped``: exchange the gap (31) and 0 rows of ``W_s``."""
  w = model.mpnn.W_s.weight
  return eqx.tree_at(lambda m: m.mpnn.W_s.weight, model, w.at[jnp.array([0, 31])].set(w[jnp.array([31, 0])]))


# ---------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------
def _cast(model: Any, dtype: str) -> Any:  # noqa: ANN401
  jd = jnp.float64 if dtype == "float64" else jnp.float32
  return jax.tree_util.tree_map(lambda x: x.astype(jd) if eqx.is_inexact_array(x) else x, model)


def _inputs_from_dump(d: dict[str, np.ndarray], dtype: str) -> dict[str, jax.Array]:
  """Network inputs rebuilt from the upstream's own featurized batch (bb-conditioned atoms only)."""
  jd = jnp.float64 if dtype == "float64" else jnp.float32
  n = d["feat/atom14_coords"].shape[0]
  xyz = np.zeros((n, 23, 3))
  msk = np.zeros((n, 23))
  xyz[:, :14] = d["feat/atom14_coords"]
  msk[:, :14] = d["feat/atom14_cond_mask"].astype(np.float64)
  return {
    "atom23_xyz": jnp.asarray(xyz, jd),
    "atom23_mask": jnp.asarray(msk, jd),
    "token_mask": jnp.asarray(d["feat/token_pad_mask"], jd),
    "residue_idx": jnp.asarray(d["feat/residue_index"], jnp.int32),
    "token_idx": jnp.arange(n, dtype=jnp.int32),
    "asym_id": jnp.asarray(d["feat/asym_id"], jnp.int32),
    "entity_id": jnp.asarray(d["feat/asym_id"], jnp.int32),
    "sym_id": jnp.zeros((n,), jnp.int32),
  }


def trace_mpnn(mpnn: Any, inp: dict[str, jax.Array], aatype: jax.Array | None) -> dict[str, Any]:  # noqa: ANN401
  """Replay ``CalibyMPNN.__call__`` layer by layer using its own layers; pinned to the real call."""
  dtype = mpnn.W_e.weight.dtype
  mask_v = inp["token_mask"].astype(dtype)
  v_init, e_init, e_idx = mpnn.features(
    atom23_xyz=inp["atom23_xyz"], atom23_mask=inp["atom23_mask"], residue_idx=inp["residue_idx"],
    token_idx=inp["token_idx"], asym_id=inp["asym_id"], entity_id=inp["entity_id"], sym_id=inp["sym_id"],
  )  # fmt: skip
  tr: dict[str, Any] = {"e_idx": e_idx, "e_init": e_init}
  h_v = v_init
  h_e = mdl._linear(mpnn.W_e, e_init)  # noqa: SLF001
  tr["W_e"] = h_e
  default_tok = mpnn.W_s.num_embeddings - 1 if mpnn.return_decoder_edges else 0
  seq_tokens = jnp.full((h_v.shape[0],), default_tok, dtype=jnp.int32) if aatype is None else aatype
  h_s = mpnn.W_s.weight[jnp.clip(seq_tokens, 0, mpnn.W_s.num_embeddings - 1)]
  tr["h_S"] = h_s
  if mpnn.update_decoder_edges:
    h_v = h_v + h_s
  mask_attend = jnp.take_along_axis(mask_v[:, None] * mask_v[None, :], e_idx, axis=1)
  for i, layer in enumerate(mpnn.encoder_layers):
    h_v, h_e = layer(h_v, h_e, e_idx, mask_v, mask_attend)
    tr[f"enc{i}"] = (h_v, h_e)
  h_es = mdl._cat_neighbors_nodes(h_s, h_e, e_idx)  # noqa: SLF001
  h_esv = mdl._cat_neighbors_nodes(h_v, h_es, e_idx)  # noqa: SLF001
  for i, layer in enumerate(mpnn.decoder_layers):
    h_v, h_esv = layer.forward_with_edges(h_v, h_esv, e_idx, mask_v, mask_attend)
    tr[f"dec{i}"] = (h_v, h_esv)
  tr["mask_attend"] = mask_attend
  return tr


def _dense_from_sparse(j: np.ndarray, edge_idx: np.ndarray, mask_ij: np.ndarray) -> np.ndarray:
  n, k = edge_idx.shape
  dense = np.zeros((n, n, *j.shape[-2:]), dtype=np.float64)
  for i in range(n):
    for kk in range(k):
      dense[i, edge_idx[i, kk]] += j[i, kk] * mask_ij[i, kk]
  return dense


def _offdiag(x: np.ndarray) -> np.ndarray:
  n = x.shape[0]
  keep = ~np.eye(n, dtype=bool)
  return x[keep]


_RED_OF_AMINX = [pts.CALIBY21_ALPHABET.index(ch) if ch != "X" else 21 for ch in pts.AMINX21_ALPHABET]


def potts_stages(model: Any, d: dict[str, np.ndarray], dtype: str, out: dict[str, float]) -> None:  # noqa: ANN401, C901, PLR0915
  inp = _inputs_from_dump(d, dtype)
  jd = jnp.float64 if dtype == "float64" else jnp.float32
  tr = trace_mpnn(model.mpnn, inp, aatype=None)
  # The replay must equal the real call, or the per-layer stages would grade a copy.
  real = model.mpnn(**inp)
  out["s0_replay_vs_call_h_v"] = maxabs(tr[f"dec{len(model.mpnn.decoder_layers) - 1}"][0], real[0])
  out["s0_replay_vs_call_h_esv"] = maxabs(tr[f"dec{len(model.mpnn.decoder_layers) - 1}"][1], real[1])

  out["s1_e_idx_mismatches"] = mismatches(tr["e_idx"], d["tf/E_idx"])
  out["s1_edge_init"] = maxabs(tr["e_init"], d["tf/E"])
  out["s1_h_s_gap_embed"] = maxabs(tr["h_S"], d["ws/h_S"])
  out["s1_W_e"] = maxabs(tr["W_e"], d["we/h_E"])
  for i in range(len(model.mpnn.encoder_layers)):
    out[f"s2_enc{i}_h_v"] = maxabs(tr[f"enc{i}"][0], d[f"enc{i}/h_V"])
    out[f"s2_enc{i}_h_e"] = maxabs(tr[f"enc{i}"][1], d[f"enc{i}/h_E"])
  for i in range(len(model.mpnn.decoder_layers)):
    out[f"s3_dec{i}_h_v"] = maxabs(tr[f"dec{i}"][0], d[f"dec{i}/h_V"])
    out[f"s3_dec{i}_h_esv"] = maxabs(tr[f"dec{i}"][1], d[f"dec{i}/h_E"])

  pout = model(**inp)
  up_dense_h = d["potts/h_full"]
  up_dense_j = _dense_from_sparse(d["potts/J_full"], d["potts/edge_idx"], d["potts/mask_ij"])
  out["s4_potts_h"] = maxabs(pout.h_32, up_dense_h)
  out["s4_potts_J_offdiag"] = maxabs(_offdiag(np.asarray(pout.j_32)), _offdiag(up_dense_j))

  h21, j21 = pts.slice_potts_params(pout.h_32, pout.j_32, to_aminx_alphabet=True)
  red_h = d["potts/h_red"]
  red_j = _dense_from_sparse(d["potts/J_red"], d["potts/edge_idx"], d["potts/mask_ij"])
  exp_h = red_h[:, _RED_OF_AMINX]
  exp_j = red_j[:, :, _RED_OF_AMINX][:, :, :, _RED_OF_AMINX]
  out["s5_h21_aminx_alphabet"] = maxabs(h21, exp_h)
  out["s5_J21_offdiag"] = maxabs(_offdiag(np.asarray(j21)), _offdiag(exp_j))

  mask = inp["token_mask"].astype(jd)
  s_wt_cal = d["feat/restype_idx"]
  s_wt_amx = np.array([pts.AMINX21_ALPHABET.index(pts.CALIBY21_ALPHABET[s]) for s in s_wt_cal])
  wt_oh = jax.nn.one_hot(jnp.asarray(s_wt_amx), 21, dtype=jd)
  n = h21.shape[0]
  off = (1.0 - jnp.eye(n, dtype=jd))[:, :, None, None]
  cond_field = h21 + jnp.einsum("ijab,jb->ia", j21 * off, wt_oh)
  out["s5_Ui_conditional_field"] = maxabs(cond_field, d["score/U_i"][:, _RED_OF_AMINX])
  u_wt = pts.compute_caliby_potts_energy(h21, j21, jnp.asarray(s_wt_amx), mask)
  out["s5_U_wt"] = maxabs(u_wt, d["score/U"])
  alt_amx = np.array([[pts.AMINX21_ALPHABET.index(pts.CALIBY21_ALPHABET[s]) for s in row] for row in d["score/alt_S"]])
  u_alt = pts.compute_caliby_potts_energy(h21, j21, jnp.asarray(alt_amx), mask)
  out["s5_U_alt"] = maxabs(u_alt, d["score/alt_U"])

  # s6 ensemble: the upstream-injected conformer coordinates are fed to aminx conformer by conformer.
  hs, js = [], []
  for c in range(d["ens/atom14_coords"].shape[0]):
    dc = {"feat/atom14_coords": d["ens/atom14_coords"][c], "feat/atom14_cond_mask": d["ens/atom14_cond_mask"][c],
          "feat/token_pad_mask": d["feat/token_pad_mask"], "feat/residue_index": d["feat/residue_index"], "feat/asym_id": d["feat/asym_id"]}  # fmt: skip
    oc = model(**_inputs_from_dump(dc, dtype))
    hh, jj = pts.slice_potts_params(oc.h_32, oc.j_32, to_aminx_alphabet=True)
    hs.append(hh)
    js.append(jj)
  h_ens, j_ens = pts.aggregate_ensemble_potts_params(jnp.stack(hs), jnp.stack(js))
  ens_red_j = d["ens/J"]
  exp_hens = d["ens/h"][:, _RED_OF_AMINX]
  exp_jens = ens_red_j[:, :, _RED_OF_AMINX][:, :, :, _RED_OF_AMINX]
  out["s6_ens_h"] = maxabs(h_ens, exp_hens)
  out["s6_ens_J_offdiag"] = maxabs(_offdiag(np.asarray(j_ens)), _offdiag(exp_jens))
  out["s6_ens_U_wt"] = maxabs(pts.compute_caliby_potts_energy(h_ens, j_ens, jnp.asarray(s_wt_amx), mask), d["ens/U"])


def prep_stages(d: dict[str, np.ndarray], out: dict[str, float]) -> None:
  prep = drv._prepare_structure(PDB, SimpleNamespace(sequences_to_score=None))  # noqa: SLF001
  # Upstream centers the structure on the mean of all resolved atoms (CenterRandomAugmentation,
  # sd_featurizer_transforms.py:171-196); the model is translation invariant, so grade the
  # coordinates up to one common translation. Upstream's centered coordinates are float32 values.
  delta = np.asarray(prep.atom23_xyz[:, :4], dtype=np.float64) - d["feat/atom14_coords"][:, :4]
  out["s0_prep_bb_xyz_up_to_translation"] = maxabs(delta - delta.reshape(-1, 3).mean(axis=0), 0.0 * delta)
  out["s0_prep_bb_mask"] = maxabs(prep.atom23_mask[:, :4], d["feat/atom14_cond_mask"][:, :4].astype(np.float64))
  out["s0_prep_token_mask"] = maxabs(prep.token_mask, d["feat/token_pad_mask"])
  out["s0_prep_seq_mismatches"] = mismatches(prep.aatype_caliby, d["feat/restype_idx"])
  # residue numbering: aminx counts from 1, upstream from 0; only differences enter the model.
  out["s0_prep_residue_offsets"] = maxabs(np.diff(prep.residue_idx), np.diff(d["feat/residue_index"]))
  out["s0_prep_asym_mismatches"] = mismatches(prep.asym_id, d["feat/asym_id"])


def e2e_stage(model: Any, ckpt_path: Path, d: dict[str, np.ndarray], out: dict[str, float]) -> None:  # noqa: ANN401
  spec = SimpleNamespace(
    inputs=[PDB], model_local_path=str(ckpt_path), checkpoint_id=None, caliby=CalibyOptions(),
    sequences_to_score=None, random_seed=0, num_samples=1, temperature=0.01,
  )  # fmt: skip
  driver = drv.CalibyDriver()
  batch = next(iter(driver.batches(spec)))
  stages = driver.stages(spec, "score:energy", model)
  res = stages(batch, chunk_start=0, chunk_count=1, scalar_dropout=False, dropout_key=jax.random.PRNGKey(0))
  out["s7_e2e_driver_U_wt"] = maxabs(np.asarray(res["energy"]).reshape(-1)[:1], d["score/U"].reshape(-1))


def packer_stages(model: Any, d: dict[str, np.ndarray], out: dict[str, float]) -> None:  # noqa: ANN401
  inp = _inputs_from_dump(d, "float32")
  aat = jnp.asarray(d["pk/call1_aatype"], jnp.int32)
  tr = trace_mpnn(model.mpnn, inp, aatype=aat)
  last = len(model.mpnn.decoder_layers) - 1
  out["pk_trunk_h_v"] = maxabs(tr[f"dec{last}"][0], d[f"dec{last}/h_V"])
  out["pk_e_idx_mismatches"] = mismatches(tr["e_idx"], d["tf/E_idx"])
  s_min, s_max, rho, sd = float(d["pk/s_min"]), float(d["pk/s_max"]), float(d["pk/rho"]), float(d["pk/sigma_data"])
  n = d["pk/call1_x_scn"].shape[0]
  ones = jnp.ones((n, 33), jnp.float32)

  def sigma_of(t: float) -> float:
    return ((s_max ** (1 / rho)) + t * ((s_min ** (1 / rho)) - (s_max ** (1 / rho)))) ** rho * sd

  out["pk_prior_vs_call1_input"] = maxabs(d["pk/prior_x0"], d["pk/call1_x_scn"])
  h_v = jnp.asarray(d["pk/call1_h_V"], jnp.float32)
  sig1 = jnp.full((n,), sigma_of(float(d["pk/call1_t"])), jnp.float32)
  y1 = model.denoise_from_trunk(jnp.asarray(d["pk/call1_x_scn"], jnp.float32), sig1, h_v, ones, x_cond=None, aatype=aat)
  out["pk_call1_denoise"] = maxabs(y1, d["pk/call1_x_out"])
  sig2 = jnp.full((n,), sigma_of(float(d["pk/call2_t"])), jnp.float32)
  y2 = model.denoise_from_trunk(jnp.asarray(d["pk/call2_x_scn"], jnp.float32), sig2, h_v, ones, x_cond=jnp.asarray(d["pk/call2_x_self_cond"], jnp.float32), aatype=aat)
  out["pk_call2_denoise_selfcond"] = maxabs(y2, d["pk/call2_x_out"])


def stage_band(name: str, dtype: str) -> float:
  """Per-stage tolerance. Counts (mismatches) must be exactly zero; floats use the dtype band."""
  if name.endswith("_mismatches"):
    return 0.0
  if name.startswith("s7_"):
    return BANDS["float32"]  # the driver path casts its inputs to float32 (driver.py _potts_21_in_aminx_order)
  if name.startswith("s0_prep_bb_xyz"):
    return 1e-4  # float32-origin centered coordinates; independent of the model dtype
  return BANDS[dtype]


# ---------------------------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------------------------
def load_dump(oracle_dir: Path, ckpt: str, dtype: str, expected_sha: dict[str, str]) -> dict[str, np.ndarray]:
  stem = f"{ckpt}_1ubq_{dtype}"
  npz = oracle_dir / f"{stem}.npz"
  man = json.loads((oracle_dir / f"{stem}.manifest.json").read_text(encoding="utf-8"))
  sha = sha256_file(npz)
  if sha != man["dump_sha256"] or sha != expected_sha.get(stem, sha):
    msg = f"{stem}: dump sha256 {sha} != manifest {man['dump_sha256']} / pinned {expected_sha.get(stem)}"
    raise RuntimeError(msg)
  with np.load(npz) as z:
    return {k: z[k] for k in z.files}


def run_arm(ckpt: str, dtype: str, weights_dir: Path, oracle_dir: Path, mutant: str | None, expected_sha: dict[str, str]) -> dict[str, float]:
  d = load_dump(oracle_dir, ckpt, dtype, expected_sha)
  path = weights_dir / f"{ckpt}.ckpt"
  model = _cast(drv._load_caliby_torch(path), dtype)  # noqa: SLF001
  if mutant == "gap_embedding_row_swapped":
    model = _swap_gap_row(model)
  out: dict[str, float] = {}
  with apply_mutant(mutant):
    if "packer" in ckpt:
      packer_stages(model, d, out)
    else:
      prep_stages(d, out)
      potts_stages(model, d, dtype, out)
      e2e_stage(model, path, d, out)
  return out


def summarize(arm: dict[str, float], dtype: str) -> dict[str, Any]:
  fails = {k: v for k, v in arm.items() if not (v <= stage_band(k, dtype))}
  worst = max((v for k, v in arm.items() if not k.endswith("_mismatches") and math.isfinite(v)), default=0.0)
  return {"n_stages": len(arm), "n_out_of_band": len(fails), "out_of_band": sorted(fails), "max_stage_diff": worst}


def main() -> None:
  parser = argparse.ArgumentParser(description="Caliby parity vehicle (sealed upstream oracle dumps).")
  parser.add_argument("--oracle-dir", type=Path, required=True)
  parser.add_argument("--weights-dir", type=Path, default=Path.home() / "repos" / "caliby_weights" / "caliby")
  parser.add_argument("--pinned-dumps", type=Path, default=Path(__file__).with_name("caliby_oracle_dumps.sha256.json"))
  parser.add_argument("--mutants", type=str, default=",".join(MUTANTS))
  parser.add_argument("--arms", type=str, default="", help="Comma list ckpt/dtype to restrict clean arms (dev only).")
  parser.add_argument("--out", type=Path, default=Path("outputs/parity/caliby_parity.json"))
  parser.add_argument("--controls-out", type=Path, default=None)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")

  pinned = json.loads(args.pinned_dumps.read_text(encoding="utf-8"))
  mutant_list = [m for m in args.mutants.split(",") if m]
  clean: dict[str, Any] = {}
  weights: dict[str, str] = {}
  arms = POTTS_ARMS + PACKER_ARMS
  if args.arms:
    wanted = set(args.arms.split(","))
    arms = tuple(a for a in arms if f"{a[0]}/{a[1]}" in wanted)
  for ckpt, dtype in arms:
    key = f"{ckpt}/{dtype}"
    clean[key] = run_arm(ckpt, dtype, args.weights_dir, args.oracle_dir, None, pinned)
    weights[stable_artifact_key(args.weights_dir / f"{ckpt}.ckpt")] = sha256_file(args.weights_dir / f"{ckpt}.ckpt")
    log.info("clean %s: %s", key, summarize(clean[key], dtype))

  # Mutants run on the two arms that exercise the most code: caliby (v0 features) and
  # soluble_caliby_v1 (v1 features), float64; the packer mutant is not applicable to Potts mutants.
  mutants: dict[str, Any] = {}
  for m in mutant_list:
    per_arm = {}
    for ckpt, dtype in (("caliby", "float64"), ("soluble_caliby_v1", "float64")):
      try:
        arm = run_arm(ckpt, dtype, args.weights_dir, args.oracle_dir, m, pinned)
        per_arm[f"{ckpt}/{dtype}"] = summarize(arm, dtype)
      except Exception as exc:  # noqa: BLE001
        per_arm[f"{ckpt}/{dtype}"] = {"error": f"{type(exc).__name__}: {exc}"}
    tripped = any(("error" not in v and v["n_out_of_band"] > 0) for v in per_arm.values())
    mutants[m] = {"status": "failed" if tripped else "survived", "arms": per_arm}
    log.info("mutant %s: %s", m, mutants[m]["status"])

  clean_summary = {k: summarize(v, k.split("/")[1]) for k, v in clean.items()}
  clean_ok = all(s["n_out_of_band"] == 0 for s in clean_summary.values())
  result = {
    "clean": "pass" if clean_ok else "fail",
    "n_arms": len(arms),
    "n_clean_arms_in_band": sum(1 for s in clean_summary.values() if s["n_out_of_band"] == 0),
    "n_listed": len(mutant_list),
    "n_failed": sum(1 for m in mutants.values() if m["status"] == "failed"),
    "weight_source": "upstream_checkpoint",
    "oracle_dump_sha256": pinned,
    "clean_summary": clean_summary,
    "clean_stages": clean,
    "mutants": {m: v["status"] for m, v in mutants.items()},
    "mutant_detail": mutants,
    "not_compared": ["potts sampler (dlmc, debt #3121)", "packer euler sampler with self-conditioning (TBD-CALIBY)"],
  }

  def safe(v: Any) -> Any:  # noqa: ANN401
    if isinstance(v, float) and not math.isfinite(v):
      return None
    if isinstance(v, dict):
      return {k: safe(x) for k, x in v.items()}
    if isinstance(v, list):
      return [safe(x) for x in v]
    return v

  result = safe(result)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  if args.controls_out is not None:
    args.controls_out.parent.mkdir(parents=True, exist_ok=True)
    args.controls_out.write_text(
      json.dumps({"clean": result["clean"], "mutants": result["mutants"], "weights": weights}, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    Path(bth_results).parent.mkdir(parents=True, exist_ok=True)
    Path(bth_results).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
