"""AntiBMPNN parity vehicle: aminx against a SEALED dump of the real upstream.

The oracle is ``dump_antibmpnn_oracle.py`` (real ``protein_mpnn_utils.py``, real
``antibmpnn_*.pt``, real ``Running_AntiBMPNN_run.py``/helper scripts, fixture ``inputs/HuJ3``) and
``dump_antibmpnn_descriptors_oracle.py`` (real ``parsing_design_result.CDRH3_Comb`` + ``peptides``).
Neither is imported here: this script only reads their ``.npz`` dumps and verifies each dump's
sha256 against the value pinned in ``antibmpnn_parity.bth.toml`` ``[data]``.

Stages graded (clean arm), aminx value vs upstream value:

* ``parse``       ``aminx.io.upstream_pdb.parse_pdb_upstream`` vs upstream ``parse_PDB`` (HuJ3, chain B).
* ``features``    ``model.features`` edge features vs upstream ``W_e(E)`` (and ``E_idx``), primary ckpt.
* ``encoder``     node ``h_V`` after the encoder (all 11 ckpts) and edge ``h_E`` (primary ckpt).
* ``logits``      teacher-forced ``log_probs`` for the upstream decoding order (all 11 ckpts x f32/f64).
* ``sampler``     per-position sampling distribution ``softmax((logits+bias)/T)`` (omit via aminx
                  ``compile_omit_aa_bias``) vs the probabilities upstream ``ProteinMPNN.sample``
                  produced at each step, and the token each step would draw (Gumbel-max on the stored
                  noise).  Two configurations (``cdr5`` and ``knobs``), float32.
* ``published``   the SHIPPED ``antibmpnn_v_48_{000,010,020,030}`` weights (``load_model``) against the
                  upstream ``log_probs`` of the matching ``.pt``.
* ``descriptors`` ``compute_cdr_physicochemical_descriptors`` vs ``peptides`` (CSV rounded to 4 dp
                  from the real ``CDRH3_Comb``, and full precision) plus a pH sweep.

Negative controls are aminx-side code perturbations, each in a fresh interpreter, and must push their
designated stage above the fail band.  Bands: f64 <= 1e-7, f32 <= 1e-4 pass; (1e-4, 1e-3] marginal;
> 1e-3 fail.  Results go to ``$BTH_RESULTS_PATH``.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
  os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "src", _REPO / "scripts", _REPO / "scripts" / "parity"):
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from artifact_key import stable_artifact_key  # noqa: E402

log = logging.getLogger("antibmpnn_parity")

SIDECAR = Path(__file__).with_suffix(".bth.toml")
CHECKPOINTS = tuple(f"antibmpnn_{n:03d}" for n in range(0, 101, 10))
PRIMARY = "antibmpnn_000"
PUBLISHED = {"antibmpnn_000": "v_48_000", "antibmpnn_010": "v_48_010", "antibmpnn_020": "v_48_020", "antibmpnn_030": "v_48_030"}
CONFIGS = ("cdr5", "knobs")
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
PASS_F64 = 1e-7
PASS_F32 = 1e-4
FAIL_MIN = 1e-3
#: mutant -> the stage that must leave the fail band
MUTANTS = {
  "reversed_ar_mask": "logits",
  "zero_edge_embedding": "features",
  "swap_checkpoint_noise": "logits",
  "omit_penalty_dropped": "sampler",
  "lehninger_pka_shifted": "descriptors",
  "barley_tables_swapped": "descriptors",
  "parser_wrong_atom_order": "parse",
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _pinned() -> dict[str, Any]:
  return tomllib.loads(SIDECAR.read_text(encoding="utf-8"))["data"]


def _maxabs(a: Any, b: Any) -> float:  # noqa: ANN401
  x = np.asarray(a, dtype=np.float64)
  y = np.asarray(b, dtype=np.float64)
  if x.shape != y.shape:
    msg = f"shape mismatch {x.shape} vs {y.shape}"
    raise ValueError(msg)
  return float(np.max(np.abs(x - y))) if x.size else 0.0


def _verify_inputs(oracle_dir: Path, root: Path) -> dict[str, str]:
  """Fail loudly unless every sealed input matches the sha256 pinned in the sidecar."""
  pinned = _pinned()
  checks = {
    oracle_dir / "antibmpnn_oracle.npz": pinned["oracle_dump_sha256"],
    oracle_dir / "antibmpnn_descriptors_oracle.npz": pinned["descriptors_dump_sha256"],
    root / "inputs" / "HuJ3" / "HuJ3.pdb": pinned["fixture_sha256"],
  }
  for name in CHECKPOINTS:
    checks[root / "antibmpnn_model_weights" / f"{name}.pt"] = pinned["checkpoints"][name]
  weights: dict[str, str] = {}
  for path, want in checks.items():
    got = _sha256(path)
    if got != want:
      msg = f"{path} sha256 {got} != pinned {want}"
      raise SystemExit(msg)
    if path.suffix == ".pt":
      weights[stable_artifact_key(path)] = got
  return weights


# --------------------------------------------------------------------------------------------------
# aminx side
# --------------------------------------------------------------------------------------------------
def _forward(model: Any, x: Any, mask: Any, ridx: Any, cidx: Any, seq: Any, order: Any) -> dict[str, Any]:  # noqa: ANN401
  """features -> encoder -> teacher-forced conditional decoder, exposing every stage."""
  import jax
  import jax.numpy as jnp

  from aminx.model.decoder import conditional_decoder_layer_edge_features, pack_conditional_decoder_static_edges
  from aminx.utils import autoregression

  edge, nbr, _, _ = model.features(jax.random.PRNGKey(0), x, mask, ridx, cidx, backbone_noise=jnp.asarray(0.0, dtype=x.dtype))
  node, enc_edge = model.encoder(edge, nbr, mask, inference=True)
  one_hot = jax.nn.one_hot(seq, 21, dtype=x.dtype)
  ar = autoregression.ar_mask_from_decoding_order(order.astype(jnp.int32))
  seq_edge, mask_bw, masked = pack_conditional_decoder_static_edges(node, enc_edge, nbr, one_hot, model.w_s_embed.weight, ar, mask)
  h = node
  for layer in model.decoder.layers:
    h = layer(h, conditional_decoder_layer_edge_features(h, seq_edge, nbr, mask_bw, masked), mask, inference=True)
  logits = h @ model.w_out.weight.T + model.w_out.bias
  return {"edge": edge, "nbr": nbr, "node": node, "enc_edge": enc_edge, "logits": logits, "log_probs": jax.nn.log_softmax(logits, axis=-1)}


def _load_pt(root: Path, name: str, dtype: Any) -> Any:  # noqa: ANN401
  import jax
  import jax.numpy as jnp

  from convert_weights import convert_mpnn_checkpoint

  if os.environ.get("ANTIBMPNN_MUTANT") == "swap_checkpoint_noise" and name == PRIMARY:
    name = "antibmpnn_030"
  model = convert_mpnn_checkpoint(root / "antibmpnn_model_weights" / f"{name}.pt")
  import equinox as eqx

  model = jax.tree_util.tree_map(lambda v: v.astype(dtype) if eqx.is_inexact_array(v) else v, model)
  if os.environ.get("ANTIBMPNN_MUTANT") == "zero_edge_embedding":
    model = eqx.tree_at(lambda m: m.features.w_e.weight, model, jnp.zeros_like(model.features.w_e.weight))
  return model


def _inputs(d: Any, cfg: str, dtype: Any) -> tuple[Any, ...]:  # noqa: ANN401
  import jax.numpy as jnp

  g = lambda k: d[f"feat/{cfg}/{k}"][0]  # noqa: E731
  return (
    jnp.asarray(g("X"), dtype=dtype),
    jnp.asarray(g("mask"), dtype=dtype),
    jnp.asarray(g("residue_idx"), dtype=jnp.int32),
    jnp.asarray(g("chain_encoding_all"), dtype=jnp.int32),
    jnp.asarray(g("S"), dtype=jnp.int32),
  )


def _stage_forward(d: Any, root: Path, ckpts: tuple[str, ...], dtype: Any, pre: str) -> dict[str, float]:  # noqa: ANN401
  """features / encoder / logits against the dump for one precision."""
  import jax.numpy as jnp

  out = {"features": 0.0, "encoder": 0.0, "logits": 0.0, "nbr_mismatch": 0.0}
  for name in ckpts:
    model = _load_pt(root, name, dtype)
    cfg = "cdr5"
    x, mask, ridx, cidx, seq = _inputs(d, cfg, dtype)
    p = f"{name}/{pre}/{cfg}/"
    order = jnp.asarray(d[p + "order_tf"][0])
    r = _forward(model, x, mask, ridx, cidx, seq, order)
    out["nbr_mismatch"] += float(np.sum(np.asarray(r["nbr"]) != d[p + "E_idx"][0]))
    out["encoder"] = max(out["encoder"], _maxabs(r["node"], d[p + "h_V_enc"][0]))
    out["logits"] = max(out["logits"], _maxabs(r["log_probs"], d[p + "log_probs_tf"][0]))
    if name == PRIMARY:
      out["features"] = max(out["features"], _maxabs(r["edge"], d[p + "h_E0"][0]))
      out["encoder"] = max(out["encoder"], _maxabs(r["enc_edge"], d[p + "h_E_enc"][0]))
  return out


def _stage_sampler(d: Any, root: Path, dtype: Any) -> tuple[float, int]:  # noqa: ANN401
  """Sampling distribution + drawn tokens vs upstream ``ProteinMPNN.sample`` (float32 only)."""
  import jax
  import jax.numpy as jnp

  from aminx.host import omit_aa_bias

  model = _load_pt(root, PRIMARY, dtype)
  worst = 0.0
  token_mismatch = 0
  for cfg in CONFIGS:
    x, mask, ridx, cidx, _ = _inputs(d, cfg, dtype)
    p = f"{PRIMARY}/f32/{cfg}/"
    s_sample = jnp.asarray(d[p + "S_sample"][0], dtype=jnp.int32)
    order = jnp.asarray(d[p + "order_sample"][0])
    r = _forward(model, x, mask, ridx, cidx, s_sample, order)
    designed = (d[f"feat/{cfg}/chain_M"][0] * d[f"feat/{cfg}/chain_M_pos"][0] * d[f"feat/{cfg}/mask"][0]).astype(np.float32)
    temp = float(d[f"feat/{cfg}/temperature"])
    omit_letters = tuple(str(d[f"feat/{cfg}/omit"]))
    length = designed.shape[0]
    omit_mask = d[f"feat/{cfg}/omit_AA_mask"][0]
    per_pos = {int(i): "".join(ALPHABET[j] for j in np.nonzero(omit_mask[i])[0]) for i in range(length) if omit_mask[i].any()}
    bias = jnp.asarray(d[f"feat/{cfg}/bias_aa"], dtype=dtype)[None, :] + jnp.asarray(
      d[f"feat/{cfg}/bias_by_res_all"][0], dtype=dtype
    )
    fixed_row = jnp.asarray(1.0 - designed)
    # NB: cfg "cdr5" has no per-position omit; its omit_AA_mask is all zero (upstream passes None).
    total = omit_aa_bias.compile_omit_aa_bias(
      bias, fixed_mask_row=fixed_row, omit_aa=("".join(omit_letters),), omit_aa_per_position=per_pos or None, seq_len=length
    )
    probs = jax.nn.softmax((r["logits"] + total) / temp, axis=-1) * jnp.asarray(designed)[:, None]
    worst = max(worst, _maxabs(probs, d[p + "probs_sample"][0]))
    # token each step draws: argmax(log p + g[step]) at designed positions, native elsewhere
    g = d[p + "gumbel"]
    native = d[f"feat/{cfg}/S"][0]
    ord_np = np.asarray(order)
    lp = np.log(np.maximum(np.asarray(probs, dtype=np.float64), 1e-300))
    for step, pos in enumerate(ord_np):
      tok = int(np.argmax(lp[pos] + g[step])) if designed[pos] > 0 else int(native[pos])
      token_mismatch += int(tok != int(d[p + "S_sample"][0][pos]))
  return worst, token_mismatch


def _stage_published(d: Any, dtype: Any) -> float:  # noqa: ANN401
  """The shipped Hub weights vs upstream log-probs of the matching .pt."""
  import jax
  import jax.numpy as jnp
  import equinox as eqx

  from aminx.io.weights import load_model

  worst = 0.0
  for name, version in PUBLISHED.items():
    model = load_model(model_weights="antibmpnn", model_version=version)
    model = jax.tree_util.tree_map(lambda v: v.astype(dtype) if eqx.is_inexact_array(v) else v, model)
    x, mask, ridx, cidx, seq = _inputs(d, "cdr5", dtype)
    p = f"{name}/f32/cdr5/"
    r = _forward(model, x, mask, ridx, cidx, seq, jnp.asarray(d[p + "order_tf"][0]))
    worst = max(worst, _maxabs(r["log_probs"], d[p + "log_probs_tf"][0]))
  return worst


def _stage_descriptors(dd: Any) -> dict[str, float]:  # noqa: ANN401
  import jax.numpy as jnp

  from aminx.model import cdr_bias

  def tokens(seq: str) -> Any:  # noqa: ANN401
    """MPNN alphabet token ids for a one-letter sequence."""
    return jnp.asarray([ALPHABET.index(c) for c in seq], dtype=jnp.int32)

  csv = {"PD1_volume": 0.0, "PD2_hydro": 0.0, "PD3_charge": 0.0}
  for i, seq in enumerate(dd["csv/combination"]):
    got = cdr_bias.compute_cdr_physicochemical_descriptors(tokens(str(seq)), ph=7.4)
    for key, val in (("PD1_volume", got.pd1_volume), ("PD2_hydro", got.pd2_hydro), ("PD3_charge", got.pd3_charge)):
      csv[key] = max(csv[key], abs(float(val) - float(dd[f"csv/{key}"][i])))
  direct = 0.0
  for i, seq in enumerate(dd["direct/seq"]):
    got = cdr_bias.compute_cdr_physicochemical_descriptors(tokens(str(seq)), ph=7.4)
    direct = max(
      direct,
      abs(float(got.pd1_volume) - float(dd["direct/PD1"][i])),
      abs(float(got.pd2_hydro) - float(dd["direct/PD2"][i])),
      abs(float(got.pd3_charge) - float(dd["direct/charge74"][i])),
    )
  probe = tokens(str(dd["probe/seq"]))
  sweep = 0.0
  for ph, want in zip(dd["probe/ph"], dd["probe/charge"], strict=True):
    got = cdr_bias.compute_cdr_physicochemical_descriptors(probe, ph=float(ph))
    sweep = max(sweep, abs(float(got.pd3_charge) - float(want)))
  return {"csv": max(csv.values()), "direct": direct, "ph_sweep": sweep}


def _stage_parse(d: Any, root: Path) -> float:  # noqa: ANN401
  from aminx.io import upstream_pdb

  parsed = upstream_pdb.parse_pdb_upstream(root / "inputs" / "HuJ3" / "HuJ3.pdb")[0]
  worst = 0.0
  for atom in ("N", "CA", "C", "O"):
    worst = max(worst, _maxabs(parsed["coords_chain_B"][f"{atom}_chain_B"], d[f"parse/{atom}_chain_B"]))
  if str(parsed["seq_chain_B"]) != str(d["parse/seq_chain_B"]):
    worst = max(worst, 1.0)
  return worst


def _apply_patch(mutant: str | None) -> None:
  if mutant in (None, "clean"):
    return
  os.environ["ANTIBMPNN_MUTANT"] = mutant
  if mutant == "reversed_ar_mask":
    import jax.numpy as jnp

    from aminx.utils import autoregression

    original = autoregression.ar_mask_from_decoding_order
    autoregression.ar_mask_from_decoding_order = lambda order, *a, **k: original(jnp.flip(order, -1), *a, **k)
  elif mutant == "omit_penalty_dropped":
    from aminx.host import omit_aa_bias

    omit_aa_bias.compile_omit_aa_bias = lambda bias, **_k: bias
  elif mutant == "lehninger_pka_shifted":
    from aminx.model import cdr_bias

    # real pKa edits (a 1.4 shift on His moves its pH 7.4 charge from 0.04 to 0.5)
    for key, delta in (("H", 1.4), ("C", 1.0), ("N_term", -1.5)):
      cdr_bias.LEHNINGER_PKA[key] += delta
  elif mutant == "barley_tables_swapped":
    from aminx.model import cdr_bias

    cdr_bias._PD1_TABLE, cdr_bias._PD2_TABLE = cdr_bias._PD2_TABLE, cdr_bias._PD1_TABLE  # noqa: SLF001
  elif mutant == "parser_wrong_atom_order":
    from aminx.io import upstream_pdb

    original = upstream_pdb._parse_biounits  # noqa: SLF001

    def swapped(*a: Any, **k: Any) -> Any:  # noqa: ANN401
      """Return N and CA columns exchanged."""
      xyz, seq = original(*a, **k)
      if isinstance(xyz, str):
        return xyz, seq
      return (xyz[:, [1, 0, 2, 3], :] if xyz.shape[1] == 4 else xyz), seq

    upstream_pdb._parse_biounits = swapped  # noqa: SLF001
  elif mutant not in ("zero_edge_embedding", "swap_checkpoint_noise"):
    msg = f"unknown mutant {mutant}"
    raise SystemExit(msg)


def _run_arm(args: argparse.Namespace) -> dict[str, Any]:
  import contextlib

  import jax
  import jax.numpy as jnp

  _apply_patch(args.arm)
  oracle_dir, root = args.oracle_dir, args.antibmpnn_root
  d = np.load(oracle_dir / "antibmpnn_oracle.npz")
  dd = np.load(oracle_dir / "antibmpnn_descriptors_oracle.npz")

  @contextlib.contextmanager
  def x64(enabled: bool):  # noqa: ANN202
    """Toggle ``jax_enable_x64`` for one precision pass."""
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", enabled)
    try:
      yield
    finally:
      jax.config.update("jax_enable_x64", prev)

  clean = args.arm in (None, "clean")
  ckpts = CHECKPOINTS if clean else (PRIMARY,)
  t0 = time.time()
  metrics: dict[str, Any] = {}
  with x64(False):
    f32 = _stage_forward(d, root, ckpts, jnp.float32, "f32")
    metrics["f32"] = f32
    metrics["sampler"], metrics["sampler_token_mismatch"] = _stage_sampler(d, root, jnp.float32)
    metrics["descriptors"] = _stage_descriptors(dd)
  if clean:
    with x64(True):
      metrics["f64"] = _stage_forward(d, root, ckpts, jnp.float64, "f64")
    with x64(False):
      metrics["published"] = _stage_published(d, jnp.float32)
  metrics["parse"] = _stage_parse(d, root)
  metrics["seconds"] = time.time() - t0
  return metrics


def _stage_value(metrics: dict[str, Any], stage: str) -> float:
  if stage in ("features", "encoder", "logits"):
    return float(metrics["f32"][stage])
  if stage == "descriptors":
    return float(max(metrics["descriptors"].values()))
  return float(metrics[stage])


def _parent(args: argparse.Namespace) -> dict[str, Any]:
  weights = _verify_inputs(args.oracle_dir, args.antibmpnn_root)
  work = args.work_dir or Path(os.environ.get("TMPDIR", "/tmp")) / "antibmpnn_parity"
  work.mkdir(parents=True, exist_ok=True)
  script = str(Path(__file__).resolve())
  mutants = [m for m in args.mutants.split(",") if m]
  measured: dict[str, dict[str, Any]] = {}
  for arm in ("clean", *mutants):
    out = work / f"payload_{arm}.json"
    out.unlink(missing_ok=True)
    cmd = [sys.executable, script, "--arm", arm, "--oracle-dir", str(args.oracle_dir), "--antibmpnn-root", str(args.antibmpnn_root), "--payload-out", str(out)]
    done = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=args.arm_timeout)
    if done.returncode != 0:
      log.error("arm %s failed: %s", arm, done.stderr[-800:])
      measured[arm] = {"error": done.stderr[-800:]}
      continue
    measured[arm] = json.loads(out.read_text(encoding="utf-8"))
  clean = measured["clean"]
  if "error" in clean:
    verdict = "error"
    flat: dict[str, Any] = {}
  else:
    f64 = max(clean["f64"]["features"], clean["f64"]["encoder"], clean["f64"]["logits"])
    f32 = max(
      clean["f32"]["features"], clean["f32"]["encoder"], clean["f32"]["logits"], clean["sampler"],
      clean["published"], clean["descriptors"]["direct"], clean["descriptors"]["ph_sweep"], clean["descriptors"]["csv"],
    )
    parse = clean["parse"]
    ok = (
      f64 < PASS_F64 and f32 < PASS_F32 and parse < 1e-9 and clean["sampler_token_mismatch"] == 0
      and clean["f32"]["nbr_mismatch"] == 0 and clean["f64"]["nbr_mismatch"] == 0
    )
    marginal = f64 < 1e-5 and f32 < FAIL_MIN and parse < 1e-9 and clean["sampler_token_mismatch"] == 0
    verdict = "pass" if ok else "marginal" if marginal else "fail"
    flat = {
      "max_abs_f64": f64, "max_abs_f32": f32, "parse_max_abs": parse,
      "features_f32": clean["f32"]["features"], "encoder_f32": clean["f32"]["encoder"], "logits_f32": clean["f32"]["logits"],
      "sampler_f32": clean["sampler"], "published_f32": clean["published"], "descriptors_csv": clean["descriptors"]["csv"],
      "descriptors_direct": clean["descriptors"]["direct"], "descriptors_ph_sweep": clean["descriptors"]["ph_sweep"],
      "sampler_token_mismatch": clean["sampler_token_mismatch"], "n_checkpoints": len(CHECKPOINTS),
    }
  statuses: dict[str, str] = {}
  for m in mutants:
    payload = measured[m]
    if "error" in payload:
      statuses[m] = "error"
    else:
      stage = MUTANTS[m]
      value = _stage_value(payload, stage)
      statuses[m] = "failed" if (value > FAIL_MIN or (stage == "sampler" and payload["sampler_token_mismatch"] > 0)) else "passed"
      log.info("mutant %s stage %s value %.3e -> %s", m, stage, value, statuses[m])
  if args.controls_out is not None:
    args.controls_out.parent.mkdir(parents=True, exist_ok=True)
    args.controls_out.write_text(json.dumps({"clean": verdict, "mutants": statuses, "weights": weights}, indent=2) + "\n", encoding="utf-8")
  summary = {
    "clean": verdict, "n_listed": len(mutants), "n_failed": sum(s == "failed" for s in statuses.values()),
    "oracle_verified": 1, "weight_source": "upstream_checkpoint", **flat,
    "mutant_values": {m: {"stage": MUTANTS[m], "value": (_stage_value(measured[m], MUTANTS[m]) if "error" not in measured[m] else None)} for m in mutants},
  }
  log.info("summary %s", json.dumps(summary, default=str))
  return summary


def main() -> None:
  """Entry point: parent (arms + verdict) or, with ``--arm``, one fresh-interpreter arm."""
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--oracle-dir", type=Path, default=Path(os.environ.get("ANTIBMPNN_ORACLE_DIR", "")) if os.environ.get("ANTIBMPNN_ORACLE_DIR") else None)
  parser.add_argument("--antibmpnn-root", type=Path, default=Path(os.environ.get("AMINX_ANTIBMPNN_ROOT", "~/repos/AntiBMPNN")).expanduser())
  parser.add_argument("--mutants", default=",".join(MUTANTS))
  parser.add_argument("--controls-out", type=Path, default=None)
  parser.add_argument("--arm", default=None)
  parser.add_argument("--payload-out", type=Path, default=None)
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument("--arm-timeout", type=float, default=3600.0)
  args = parser.parse_args()
  if args.oracle_dir is None:
    msg = "set --oracle-dir or $ANTIBMPNN_ORACLE_DIR"
    raise SystemExit(msg)
  if args.arm is not None:
    payload = _run_arm(args)
    args.payload_out.write_text(json.dumps(payload), encoding="utf-8")
    return
  summary = _parent(args)
  results = os.environ.get("BTH_RESULTS_PATH")
  if not results:
    msg = "antibmpnn_parity requires $BTH_RESULTS_PATH (run under bth)"
    raise SystemExit(msg)
  Path(results).write_text(json.dumps(summary, default=str), encoding="utf-8")


if __name__ == "__main__":
  main()
