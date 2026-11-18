"""Does a sequence's score depend on what else is in its batch? (debt #3487)

tev_design measured that scoring one sequence alone vs in a batch of two moved its NLL by about
2.9e-6 nats, while duplicates inside one batch agreed exactly. That is either float reassociation
(the batched program lowers its matmuls differently; bounded, harmless) or a cross-row
dependence (padding or partner content leaking into a row; a correctness bug at any size). This
script tells them apart and finds the first tensor that differs.

Design (all on the runner's own score kernels, ``make_score_split_fns`` + xtrax ``chunked_map``,
exactly the composition ``runner.score`` uses; a fidelity check pins that equivalence):

* **Content invariance (the discriminator).** At a FIXED batch size C, the target row must be
  bitwise identical whatever its partners contain or how long they are (shorter partners are
  X-padded). Any difference is a cross-row dependence: shapes are identical, so reassociation
  cannot explain it.
* **Duplicates.** ``[target, target]`` rows must agree bitwise.
* **Alone vs batched.** Per-position max |delta logit| of the target row against C=1, for
  C in {2, 4, 8, 32}, three partner contents, three partner lengths, the full-context and a
  causal ar_mask. Reports max, quantiles and the fraction of positions that are bit-exact.
* **Growth.** the largest per-C max |delta| over the smallest nonzero one (1.0 if all exact).
* **Planted leak (instrument control).** The content-invariance metric is recomputed on a target
  row with ``1e-6 * mean(partner logits)`` added. It MUST register; if not, a "content
  invariant" verdict means nothing.
* **Localisation.** Each decoder layer is wrapped in a pass-through that records its inputs and
  output with ``jax.debug.callback`` (non-perturbing: the logits with capture must equal the
  logits without, bitwise, which is checked). The first tensor in execution order where the
  target's C=1 and C=2 values differ is reported; then that layer's sub-ops are replayed from
  the C=1 inputs, direct vs vmapped over [target, partner], to name the first op whose output
  changes from batching alone.

Configurations are process-level (``--matmul-precision``, ``--x64``, and the backend via
``JAX_PLATFORMS``), so each is a separate tracked run.

task_id: 261118_aminx-campaign-teacher-forced-scoring
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

logger = logging.getLogger("score_batch_composition")

_BATCH_SIZES = (2, 4, 8, 32)
_CONTENT_SEEDS = (0, 1, 2)
_PLANT_EPS = 1e-6
_X = 20  # MPNN alphabet index of X, the padding token the runner uses


def _git_hash() -> str:
  try:
    return subprocess.run(
      ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def main() -> None:  # noqa: PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--structures", nargs="+", default=["tests/data/1ubq.pdb", "tests/data/1mbn.pdb"])
  parser.add_argument("--max-lengths", nargs="+", type=int, default=[128, 256])
  parser.add_argument("--checkpoint", default="proteinmpnn_v_48_020")
  parser.add_argument("--matmul-precision", choices=["default", "highest"], default="default")
  parser.add_argument("--x64", action="store_true")
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  import jax  # noqa: PLC0415

  # x64 is switched on only AFTER each model load: equinox builds the deserialisation skeleton in
  # the default float dtype, so loading under x64 refuses the float32 checkpoint. The loaded
  # model and the structure arrays are then cast to float64 explicitly (_to_float).
  if args.matmul_precision != "default":
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

  import equinox as eqx  # noqa: PLC0415
  import jax.numpy as jnp  # noqa: PLC0415
  import numpy as np  # noqa: PLC0415
  from xtrax.transforms.map import chunked_map  # noqa: PLC0415

  from aminx.host.prep import prep_protein_stream_and_model  # noqa: PLC0415
  from aminx.host.runner import score as runner_score  # noqa: PLC0415
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415
  from aminx.scoring.score import make_score_split_fns  # noqa: PLC0415
  from aminx.utils.aa_convert import protein_sequence_to_string, string_to_protein_sequence  # noqa: PLC0415
  from aminx.utils.autoregression import ar_mask_from_decoding_order  # noqa: PLC0415

  float_dtype = jnp.float64 if args.x64 else jnp.float32

  def _to_float(tree: Any) -> Any:  # noqa: ANN401
    return jax.tree.map(
      lambda x: x.astype(float_dtype) if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating) else x,
      tree,
    )

  per_structure: list[dict[str, Any]] = []
  all_pos_deltas: list[np.ndarray] = []
  content_dev_max = 0.0
  planted_dev_min = float("inf")
  dup_equal = True
  kernel_matches_runner = True
  capture_nonperturbing = True
  max_by_c: dict[int, float] = dict.fromkeys(_BATCH_SIZES, 0.0)
  max_causal_c2 = 0.0
  first_tensor = "none"
  first_op = "none"

  for structure, max_length in zip(args.structures, args.max_lengths, strict=True):
    spec = ScoringSpecification(
      inputs=[structure],
      checkpoint_id=args.checkpoint,
      sequences_to_score=["A"],
      max_length=max_length,
      length_bucketing=False,
      return_logits=True,
    )
    jax.config.update("jax_enable_x64", False)
    iterator, model = prep_protein_stream_and_model(spec)
    batch = next(iter(iterator))
    if args.x64:
      jax.config.update("jax_enable_x64", True)
    coords, mask = _to_float(batch.coordinates), _to_float(batch.mask)
    residue_index, chain_index = batch.residue_index, batch.chain_index
    model = _to_float(model)
    struct_len = int(coords.shape[1])
    real = np.asarray(batch.mask[0]) > 0
    length = int(real.sum())
    target = np.full(struct_len, _X, np.int32)
    target[:length] = np.clip(np.asarray(batch.aatype[0])[:length], 0, 20)
    rng_order = np.random.default_rng(7).permutation(length).astype(np.int32)
    causal = np.zeros((struct_len, struct_len), np.int8)
    causal[:length, :length] = np.asarray(ar_mask_from_decoding_order(jnp.asarray(rng_order)))
    causal = jnp.asarray(causal)

    def _partners(n: int, plen: int, seed: int, *, _sl: int = struct_len) -> np.ndarray:
      rng = np.random.default_rng(1000 + seed)
      out = np.full((n, _sl), _X, np.int32)
      out[:, :plen] = rng.integers(0, 20, size=(n, plen))
      return out

    def _kernel(
      mdl: Any,  # noqa: ANN401
      seqs: np.ndarray,
      ar_mask: Any = None,  # noqa: ANN401
      *,
      _coords: Any = coords,  # noqa: ANN401
      _mask: Any = mask,  # noqa: ANN401
      _ri: Any = residue_index,  # noqa: ANN401
      _ci: Any = chain_index,  # noqa: ANN401
    ) -> np.ndarray:
      split = make_score_split_fns(mdl)
      seqs_j = jnp.asarray(seqs)
      keys = jax.random.split(jax.random.PRNGKey(0), seqs_j.shape[0])

      def per_struct(c: Any, m: Any, r: Any, ch: Any) -> Any:  # noqa: ANN401
        enc = split.encode_structure(c, m, r, ch)

        def one(item: dict[str, Any]) -> Any:  # noqa: ANN401
          return split.score_candidate(
            item["key"], jax.nn.one_hot(item["seq"], 21, dtype=float_dtype), enc, c, m, r, ch,
            ar_mask=ar_mask,
          )

        return chunked_map(one, {"key": keys, "seq": seqs_j})

      _, logits, _ = jax.vmap(per_struct)(_coords, _mask, _ri, _ci)
      return np.asarray(logits[0])

    alone = _kernel(model, target[None])[0]
    alone_causal = _kernel(model, target[None], causal)[0]

    # Fidelity: the kernel composition IS runner.score's (float32 only; the runner has no x64 path).
    if not args.x64:
      for seqs in (target[None], np.concatenate([target[None], _partners(1, length, 0)])):
        strings = [protein_sequence_to_string(jnp.asarray(s[:length])) for s in seqs]
        # Alphabet guard: the runner re-encodes these letters; the tokens must survive the trip.
        back = [np.asarray(string_to_protein_sequence(t)) for t in strings]
        if not all(np.array_equal(b, s[:length]) for b, s in zip(back, seqs, strict=True)):
          msg = "token -> letter -> token round trip changed the sequence (alphabet mismatch)"
          raise RuntimeError(msg)
        r_spec = ScoringSpecification(
          inputs=[structure], checkpoint_id=args.checkpoint, sequences_to_score=strings,
          max_length=max_length, length_bucketing=False, return_logits=True,
        )
        r_logits = np.asarray(runner_score(r_spec)["logits"][0])
        k_logits = _kernel(model, seqs)
        same = bool(np.array_equal(r_logits[:, :length], k_logits[:, :length]))
        kernel_matches_runner &= same
        logger.info("%s fidelity C=%d kernel==runner: %s", structure, len(seqs), same)

    struct_rows: dict[str, Any] = {"structure": structure, "length": length, "struct_len": struct_len}
    for c in _BATCH_SIZES:
      reference_at_c: np.ndarray | None = None
      for seed in _CONTENT_SEEDS:
        for plen in (length, length - 15, length // 2):
          seqs = np.concatenate([target[None], _partners(c - 1, plen, seed)])
          out = _kernel(model, seqs)
          row = out[0][:length]
          pos_delta = np.abs(row - alone[:length]).max(-1)
          all_pos_deltas.append(pos_delta)
          max_by_c[c] = max(max_by_c[c], float(pos_delta.max()))
          if reference_at_c is None:
            reference_at_c = row
          else:
            content_dev_max = max(content_dev_max, float(np.abs(row - reference_at_c).max()))
            planted = row + _PLANT_EPS * out[1:, :length].mean(0)
            planted_ref = reference_at_c  # the unplanted reference at the same C
            planted_dev_min = min(planted_dev_min, float(np.abs(planted - planted_ref).max()))
      logger.info("%s C=%d max|dlogit| vs alone = %.3e", structure, c, max_by_c[c])

    dup = _kernel(model, np.stack([target, target]))
    dup_equal &= bool(np.array_equal(dup[0], dup[1]))
    causal_two = _kernel(model, np.concatenate([target[None], _partners(1, length, 0)]), causal)
    max_causal_c2 = max(max_causal_c2, float(np.abs(causal_two[0][:length] - alone_causal[:length]).max()))

    # ---- localisation (first structure only) -----------------------------------------------------
    if structure == args.structures[0]:
      records: list[tuple[str, np.ndarray]] = []

      def _record(name: str, value: Any) -> None:  # noqa: ANN401
        records.append((name, np.asarray(value)))

      class _Capture(eqx.Module):
        inner: Any
        name: str = eqx.field(static=True)

        def __call__(self, node_features: Any, layer_edge_features: Any, *a: Any, **k: Any) -> Any:  # noqa: ANN401
          jax.debug.callback(partial(_record, f"{self.name}.in_node"), node_features)
          jax.debug.callback(partial(_record, f"{self.name}.in_edge"), layer_edge_features)
          out = self.inner(node_features, layer_edge_features, *a, **k)
          jax.debug.callback(partial(_record, f"{self.name}.out"), out)
          return out

      cap_model = eqx.tree_at(
        lambda m: m.decoder.layers,
        model,
        tuple(_Capture(layer, f"decoder.layers[{i}]") for i, layer in enumerate(model.decoder.layers)),
      )
      def _rows(val: np.ndarray, core: int, n: int) -> list[np.ndarray]:
        """Split one callback value into per-candidate core tensors, in candidate order.

        Size-1 axes (the structure vmap, the S=1 state map) are squeezed; if a candidate axis of
        size ``n`` remains, it is found by shape (L, K, D are all far from n), not assumed to lead.
        """
        v = np.squeeze(val)
        if v.ndim == core:
          return [v]
        axes = [a for a in range(v.ndim) if v.shape[a] == n]
        if v.ndim != core + 1 or len(axes) != 1:
          msg = f"cannot attribute captured value of shape {val.shape} to {n} candidates"
          raise RuntimeError(msg)
        return [np.take(v, i, axis=axes[0]) for i in range(n)]

      def _captured(seqs: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        records.clear()
        out = _kernel(cap_model, seqs)
        jax.effects_barrier()
        n = int(seqs.shape[0])
        grouped: dict[str, list[np.ndarray]] = {}
        order: list[str] = []
        for name, val in records:
          if name not in grouped:
            order.append(name)
            grouped[name] = []
          grouped[name].extend(_rows(val, 3 if name.endswith(".in_edge") else 2, n))
        flat: dict[str, Any] = {name: grouped[name] for name in order}
        flat["__raw_shapes__"] = {}
        for name, val in records:
          flat["__raw_shapes__"].setdefault(name, []).append(list(val.shape))
        for name in order:
          if len(flat[name]) == 1 and n > 1:
            flat[name] = flat[name] * n  # unbatched under vmap: shared by every candidate (e.g. encoder h_V)
          if len(flat[name]) != n:
            msg = f"{name}: captured {len(flat[name])} rows for {n} candidates"
            raise RuntimeError(msg)
        flat["__order__"] = order
        return flat, out

      # Localise at the smallest batch size whose target row actually differs from C=1.
      loc_c = next(
        (c for c in _BATCH_SIZES
         if np.abs(_kernel(model, np.concatenate([target[None], _partners(c - 1, length, 0)]))[0]
                   - alone).max() > 0),
        None,
      )
      struct_rows["localisation_batch_size"] = loc_c
      partner = _partners(1, length, 0)
      group = np.concatenate([target[None], _partners((loc_c or 2) - 1, length, 0)])
      cap1, out1 = _captured(target[None])
      cap1p, _ = _captured(partner)
      cap2, out2 = _captured(group)
      pair = group
      capture_nonperturbing = bool(
        np.array_equal(out1, _kernel(model, target[None])) and np.array_equal(out2, _kernel(model, pair)),
      )
      table = []
      for name in cap1["__order__"]:  # type: ignore[index]
        a = cap1[name][0]
        # Unordered debug callbacks fire once per vmapped element in no guaranteed order, so the
        # target's row is attributed by content: the candidate row nearest the C=1 capture. The
        # partners differ from the target at O(1); the effect under study is ~1e-6. The margin
        # (second-nearest distance) is recorded so an ambiguous attribution is visible.
        dists = sorted((float(np.abs(a[:length] - r[:length]).max()), i) for i, r in enumerate(cap2[name]))
        b = cap2[name][dists[0][1]]
        margin = dists[1][0] if len(dists) > 1 else float("inf")
        d = float(np.abs(a - b).max())
        # Rows beyond the chain are padding: masked out of every NLL and of the logits a caller
        # reads. A difference confined to them is not a divergence of the scored quantity.
        d_real = float(np.abs(a[:length] - b[:length]).max())
        table.append({
          "tensor": name, "max_abs_diff": d, "real_rows_max_abs_diff": d_real,
          "finite": bool(np.isfinite(a).all()), "nonconstant": bool(np.ptp(a) > 0),
          "attribution_margin": margin,
          "raw_shapes_c1": cap1["__raw_shapes__"].get(name, [])[:3],
          "raw_shapes_cn": cap2["__raw_shapes__"].get(name, [])[:3],
        })
        if margin <= 1e3 * max(d_real, 1e-12) and margin != float("inf") and d_real > 0:
          msg = f"{name}: ambiguous attribution (diff {d_real:.3e}, margin {margin:.3e})"
          raise RuntimeError(msg)
        if first_tensor == "none" and d_real > 0:
          first_tensor = name
      logit_d = float(np.abs(out1[0][:length] - out2[0][:length]).max())
      table.append({"tensor": "logits", "max_abs_diff": logit_d, "real_rows_max_abs_diff": logit_d})
      if first_tensor == "none" and logit_d > 0:
        first_tensor = "logits(w_out)"
      struct_rows["first_divergence_table"] = table
      logger.info("first divergent tensor (C=1 vs C=%s): %s", loc_c, first_tensor)

      # Projection replay: the real _project_logits on the captured final decoder features, run
      # direct (as the C=1 program does) and vmapped over C (as the batched program does).
      if first_tensor == "logits(w_out)":
        from aminx.inference.decode._kernel import _project_logits  # noqa: PLC0415

        last = f"decoder.layers[{len(model.decoder.layers) - 1}].out"
        h_t = jnp.asarray(cap1[last][0])[None]
        h_p = jnp.asarray(cap1p[last][0])[None]
        direct = np.asarray(jax.jit(lambda h: _project_logits(model, h))(h_t))[0]
        proj_rows = {"direct_matches_alone_logits": bool(np.array_equal(direct[:length], alone[:length]))}
        w = np.asarray(model.w_out.weight, np.float64)
        bias = np.asarray(model.w_out.bias, np.float64)
        exact = np.asarray(cap1[last][0], np.float64)[:length] @ w.T + bias
        proj_rows["direct_err_vs_f64_exact"] = float(np.abs(direct[:length] - exact).max())
        for c in _BATCH_SIZES:
          stack = jnp.concatenate([h_t] + [h_p] * (c - 1))
          batched = np.asarray(jax.jit(jax.vmap(lambda h: _project_logits(model, h[None])[0]))(stack))
          proj_rows[f"c{c}_vs_direct"] = float(np.abs(batched[0][:length] - direct[:length]).max())
          proj_rows[f"c{c}_err_vs_f64_exact"] = float(np.abs(batched[0][:length] - exact).max())
        group_out = _kernel(model, group)
        replay_group = np.asarray(jax.jit(jax.vmap(lambda h: _project_logits(model, h[None])[0]))(
          jnp.concatenate([h_t] + [h_p] * (len(group) - 1))))
        proj_rows["batched_replay_matches_program"] = bool(
          np.array_equal(replay_group[0][:length], group_out[0][:length]))
        struct_rows["projection_replay"] = proj_rows
        first_op = "w_out projection (vmapped Linear: batched dot lowering)" if any(
          proj_rows[f"c{c}_vs_direct"] > 0 for c in _BATCH_SIZES) else "none"
        logger.info("projection replay: %s", proj_rows)

      # Sub-op replay of the first divergent decoder layer from the C=1 inputs: direct vs vmap(2).
      if first_tensor.startswith("decoder.layers["):
        idx = int(first_tensor.split("[")[1].split("]")[0])
        layer = model.decoder.layers[idx]
        key = f"decoder.layers[{idx}]"
        h_t, e_t = cap1[f"{key}.in_node"][0], cap1[f"{key}.in_edge"][0]
        h_p, e_p = cap1p[f"{key}.in_node"][0], cap1p[f"{key}.in_edge"][0]
        m_row = jnp.asarray(mask[0])

        def _subops(h: Any, e: Any, _layer: Any = layer, _m: Any = m_row) -> dict[str, Any]:  # noqa: ANN401
          expand = jnp.tile(jnp.expand_dims(h, -2), [1, e.shape[1], 1])
          mlp_input = jnp.concatenate([expand, e], -1)
          message = jax.vmap(jax.vmap(_layer.message_mlp))(mlp_input)
          acc = jnp.promote_types(message.dtype, jnp.float32)
          agg = (jnp.sum(message.astype(acc), -2) / 30.0).astype(message.dtype)
          h1 = jax.vmap(_layer.norm1)(h + agg)
          dense = jax.vmap(_layer.dense)(h1)
          h2 = jax.vmap(_layer.norm2)(h1 + dense)
          return {"message_mlp": message, "aggregate": agg, "norm1": h1, "dense": dense, "norm2": h2, "out": _m[:, None] * h2}

        direct = jax.jit(_subops)(jnp.asarray(h_t), jnp.asarray(e_t))
        n_rep = loc_c or 2
        batched = jax.jit(jax.vmap(_subops))(
          jnp.asarray(np.stack([h_t] + [h_p] * (n_rep - 1))), jnp.asarray(np.stack([e_t] + [e_p] * (n_rep - 1))),
        )
        sub_table = []
        for name in ("message_mlp", "aggregate", "norm1", "dense", "norm2", "out"):
          d = float(np.abs(np.asarray(direct[name]) - np.asarray(batched[name][0])).max())
          sub_table.append({"op": name, "max_abs_diff": d})
          if first_op == "none" and d > 0:
            first_op = name
        replay_matches = bool(np.array_equal(np.asarray(direct["out"]), cap1[f"{key}.out"][0]))
        struct_rows["subop_replay"] = {"layer": idx, "table": sub_table, "replay_matches_capture": replay_matches}
        logger.info("first divergent sub-op in %s: %s (replay==capture %s)", key, first_op, replay_matches)
    per_structure.append(struct_rows)

  deltas = np.concatenate(all_pos_deltas)
  # Growth: the largest per-C maximum over the smallest NONZERO one (1.0 when every C is exact).
  # A ratio to C=2 alone is undefined where C=2 happens to lower like C=1 (exact on CPU).
  nonzero = [v for v in max_by_c.values() if v > 0]
  growth = max(nonzero) / min(nonzero) if nonzero else 1.0
  payload = {
    "content_invariant": content_dev_max == 0.0,
    "content_dev_max": content_dev_max,
    "planted_detected": planted_dev_min > 0.0,
    "planted_dev_min": planted_dev_min,
    "dup_rows_equal": dup_equal,
    "kernel_matches_runner": kernel_matches_runner if not args.x64 else True,
    "capture_nonperturbing": capture_nonperturbing,
    "max_abs_dlogit_vs_alone": float(deltas.max()),
    "p50_pos_dlogit": float(np.quantile(deltas, 0.5)),
    "p99_pos_dlogit": float(np.quantile(deltas, 0.99)),
    "frac_positions_exact": float((deltas == 0).mean()),
    "max_by_c": {str(c): v for c, v in max_by_c.items()},
    "max_causal_c2": max_causal_c2,
    "growth_ratio": float(growth),
    "first_divergent_tensor": first_tensor,
    "first_divergent_op": first_op,
    "backend": jax.default_backend(),
    "device_kind": jax.devices()[0].device_kind,
    "matmul_precision": args.matmul_precision,
    "x64": bool(args.x64),
    "jax_version": jax.__version__,
    "git_hash": _git_hash(),
    "per_structure": per_structure,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2))
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    Path(results_path).write_text(json.dumps({k: v for k, v in payload.items() if k != "per_structure"}))
  logger.info("wrote %s", args.out)


if __name__ == "__main__":
  main()
