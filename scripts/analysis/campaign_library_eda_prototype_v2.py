"""Prototype of knob-faceted campaign library EDA over a finished aminx campaign.

Reads a finished campaign read-only (manifest + per-row zarr sinks), joins each row's knobs to
its sampled library, and writes long-format tables of library statistics faceted by those knobs,
plus a run-telemetry completion table. It is the throwaway prototype for the respec of
``xtrax.eda`` (task 261106_campaign-eda-respec); the code is split the way the spec splits it:

* ENGINE (destined for xtrax): manifest view, knob detection, sink-schema binding, completion.
* METRICS (destined for aminx): target residue tables and protein library statistics.

Controls (pre-registered in the sidecar):

* positive: reproduce tev_design's recorded per-model ``floor`` and ``mean_offdiag_jsd`` from
  the same stores with its exact method;
* negative: on the instrument arms (ProteinMPNN, SolubleMPNN), recovery against a
  composition-preserving shuffle of the native sequence must fall to chance while true recovery
  does not.

v2 changes from v1 (run 64c6845b, outcome fail): native residues come from residue tables built
with the campaign's own parser (``build_campaign_residue_tables.py``), never from re-parsing here;
the negative control is scoped to the instrument arms, and every arm gets a degeneracy flag.

Nothing is written outside ``--out-dir``; the campaign stores are opened with ``mode="r"``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.distance import jensenshannon

log = logging.getLogger("campaign_library_eda_prototype")

# --------------------------------------------------------------------------------------------
# ENGINE: campaign view, knob detection, sink schema (destined for xtrax)
# --------------------------------------------------------------------------------------------

#: Row fields that identify or locate a unit rather than configure it; never knobs.
BOOKKEEPING_FIELDS = frozenset({
  "manifest_row_hash", "output_h5_path", "job_id", "job_index", "chunk_id", "config_hash",
  "dataset_fingerprint", "environment_image", "git_sha", "planner_version", "sample_start",
  "sample_count", "sampling_spec",
})


@dataclass(frozen=True)
class SinkArray:
  """A consumer-declared meaning for one sink array (zarr dimension_names are absent here)."""

  name: str
  dims: tuple[str, ...]
  alphabet: str | None = None


#: The aminx sampling sink, as declared by the consumer for the engine.
AMINX_SAMPLING_SINK = (
  SinkArray("sequences", ("sample", "grid", "temperature", "residue"), alphabet="ACDEFGHIKLMNPQRSTVWYX"),
  SinkArray("logits", ("sample", "grid", "temperature", "residue", "token"), alphabet="ACDEFGHIKLMNPQRSTVWYX"),
  SinkArray("fixed_mask", ("residue",)),
  SinkArray("fixed_tokens", ("residue",), alphabet="ACDEFGHIKLMNPQRSTVWYX"),
)


def detect_row_knobs(rows: list[dict]) -> dict[str, list[str]]:
  """Top-level row fields (minus bookkeeping) whose value differs across rows."""
  values: dict[str, set[str]] = defaultdict(set)
  for row in rows:
    for key, val in row.items():
      if key not in BOOKKEEPING_FIELDS:
        values[key].add(json.dumps(val, sort_keys=True))
  return {k: sorted(v) for k, v in sorted(values.items()) if len(v) > 1}


def open_row_store(row: dict) -> zarr.Group:
  """Open a row's sink read-only through the manifest's own path."""
  return zarr.open_group(row["output_h5_path"], mode="r")


def read_sink(row: dict) -> dict[str, np.ndarray]:
  group = open_row_store(row)
  (struct_key,) = list(group.group_keys())
  s = group[struct_key]
  out = {}
  for arr in AMINX_SAMPLING_SINK:
    a = np.asarray(s[arr.name])
    if a.ndim != len(arr.dims):
      msg = f"{row['output_h5_path']}: {arr.name} has {a.ndim} dims, schema declares {arr.dims}"
      raise ValueError(msg)
    out[arr.name] = a
  return out


# --------------------------------------------------------------------------------------------
# METRICS: target residue tables and library statistics (destined for aminx)
# --------------------------------------------------------------------------------------------

AF2_ORDER = "ARNDCQEGHILKMFPSTWYVX"
MPNN_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
AF2_TO_MPNN = np.array([MPNN_ORDER.index(c) for c in AF2_ORDER])


@dataclass(frozen=True)
class TargetTable:
  path: str
  sha256: str
  native: np.ndarray  # (L,) MPNN-order tokens, as aminx's parser orders residues
  residue_label: list[str]  # e.g. "H46/A0"


def load_target_tables(path: Path, expect_aminx: str, expect_proxide: str) -> tuple[dict[str, TargetTable], dict]:
  """Residue tables built with the campaign's parser; refuse a table from any other build."""
  doc = json.loads(path.read_text())
  ident = doc["parser_identity"]
  if (ident["aminx"], ident["proxide"]) != (expect_aminx, expect_proxide):
    msg = f"residue tables came from {ident}, campaign ran aminx {expect_aminx} / proxide {expect_proxide}"
    raise ValueError(msg)
  if doc["alphabet"] != MPNN_ORDER:
    raise ValueError(f"residue tables use alphabet {doc['alphabet']!r}, sinks use {MPNN_ORDER!r}")
  out = {}
  for p, t in doc["tables"].items():
    digest = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest != t["sha256"]:
      raise ValueError(f"{p} changed since its residue table was built")
    out[p] = TargetTable(path=p, sha256=digest, native=np.asarray(t["native_mpnn"]), residue_label=t["residue_label"])
  return out, ident


INSTRUMENT_ARMS = frozenset({"protein_mpnn_v48_020", "soluble_mpnn_v48_020"})


def designable_mask(length: int, fixed_mask: np.ndarray) -> np.ndarray:
  """Real residues (not padding) that the run did not fix."""
  mask = np.zeros(fixed_mask.shape[0], dtype=bool)
  mask[:length] = True
  return mask & ~fixed_mask.astype(bool)


def recovery(seqs: np.ndarray, native: np.ndarray, mask: np.ndarray) -> np.ndarray:
  """(sample, temperature) fraction of designable positions equal to native."""
  ref = np.zeros(mask.shape[0], dtype=seqs.dtype)
  ref[: native.shape[0]] = native
  hit = (seqs == ref) & mask
  return hit.sum(-1) / mask.sum()


def derived_nll(seqs: np.ndarray, logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
  """(sample, temperature) mean -log p(sampled token) under the stored logits, designable only."""
  lse = np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1)) + logits.max(-1)
  tok = np.take_along_axis(logits, seqs[..., None].astype(np.int64), axis=-1)[..., 0]
  nll = lse - tok
  return (nll * mask).sum(-1) / mask.sum()


def pairwise_identity(seqs: np.ndarray, mask: np.ndarray) -> np.ndarray:
  """(temperature,) mean identity over sample pairs within one row's library, designable only."""
  n = seqs.shape[0]
  iu = np.triu_indices(n, k=1)
  same = (seqs[:, None] == seqs[None, :]) & mask  # (n, n, T, L)
  ident = same.sum(-1) / mask.sum()
  return ident[iu].mean(0)


def mean_prob_entropy_bits(logits: np.ndarray) -> np.ndarray:
  """(L,) entropy in bits of the softmax averaged over samples and temperatures."""
  flat = logits.reshape(-1, logits.shape[-2], logits.shape[-1]).astype(np.float64)
  p = np.exp(flat - flat.max(-1, keepdims=True))
  p /= p.sum(-1, keepdims=True)
  mp = p.mean(0)
  return -(mp * np.log2(np.clip(mp, 1e-300, None))).sum(-1)


# Positive control: tev_design analyze_state_distinguishability_ar.py, reproduced verbatim.
def tev_mean_probs(logits: np.ndarray) -> np.ndarray:
  flat = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
  probs = np.exp(flat - flat.max(axis=-1, keepdims=True))
  probs /= probs.sum(axis=-1, keepdims=True)
  return probs.mean(axis=0)


def tev_floor_effect(by_state_rep: dict[str, dict[int, np.ndarray]]) -> tuple[float, float]:
  states = sorted(by_state_rep)
  reps = sorted(next(iter(by_state_rep.values())))
  floor_vals = []
  for s in states:
    for i, ri in enumerate(reps):
      for rj in reps[i + 1 :]:
        floor_vals.append(float(np.nanmean(jensenshannon(by_state_rep[s][ri], by_state_rep[s][rj], base=2, axis=-1))))
  n = len(states)
  effect = np.zeros((n, n))
  for i in range(n):
    for j in range(i + 1, n):
      per_r = [float(np.nanmean(jensenshannon(by_state_rep[states[i]][r], by_state_rep[states[j]][r], base=2, axis=-1))) for r in reps]
      effect[i, j] = effect[j, i] = float(np.mean(per_r))
  return float(np.mean(floor_vals)), float(effect[np.triu_indices_from(effect, k=1)].mean())


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

ARM_LABELS = {
  "protein_mpnn_v48_020": "ProteinMPNN",
  "soluble_mpnn_v48_020": "SolubleMPNN",
  "ligand_mpnn_v32_020_25_unconditioned": "LigandMPNN, no context",
  "ligand_mpnn_v32_020_25_ligand_only": "LigandMPNN, ligand",
  "ligand_mpnn_v32_020_25_sc_only": "LigandMPNN, side chains",
  "ligand_mpnn_v32_020_25_ligand_and_sc": "LigandMPNN, ligand + side chains",
}


def write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def main() -> int:  # noqa: PLR0915
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--manifest", type=Path, required=True)
  ap.add_argument("--reference-results", type=Path, required=True, help="tev_design analyze_state_distinguishability_ar.json")
  ap.add_argument("--out-dir", type=Path, required=True)
  ap.add_argument("--residue-track-arm", default="protein_mpnn_v48_020")
  ap.add_argument("--residue-track-state", default="1LVB")
  ap.add_argument("--residue-tables", type=Path, required=True)
  ap.add_argument("--expect-aminx", required=True)
  ap.add_argument("--expect-proxide", required=True)
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  args.out_dir.mkdir(parents=True, exist_ok=True)

  manifest = json.loads(args.manifest.read_text())
  rows = manifest["rows"]
  beads = manifest["metadata"]["bead_summary"]
  if sum(b["n_rows"] for b in beads) != len(rows):
    raise ValueError("bead_summary does not cover the manifest rows")
  labels = []
  it = iter(rows)
  for b in beads:
    for _ in range(b["n_rows"]):
      labels.append((next(it), b))

  knobs = detect_row_knobs(rows)
  log.info("row knobs that vary: %s", sorted(knobs))
  # The arm label must be a pure function of the arm knobs (checkpoint + conditioning).
  arm_key: dict[str, set[tuple]] = defaultdict(set)
  for row, b in labels:
    arm_key[b["model_id"]].add((row["checkpoint_id"], row["ligand_conditioning"], row["sidechain_conditioning"]))
  if any(len(v) != 1 for v in arm_key.values()):
    raise ValueError(f"arm labels are not a function of the arm knobs: {dict(arm_key)}")

  targets, parser_identity = load_target_tables(args.residue_tables, args.expect_aminx, args.expect_proxide)
  arm_counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(21))
  arm_rec: dict[str, list[float]] = defaultdict(list)
  arm_shuf: dict[str, list[float]] = defaultdict(list)
  rng = np.random.default_rng(0)
  per_seed, completion, fixed_rows = [], [], {}
  tev_probs: dict[str, dict[str, dict[int, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
  shuffled_max = 0.0
  true_min = 1.0
  designable_ok = True
  entropy_track = None
  temps_seen: set[tuple] = set()
  rows_read = 0

  for row, b in labels:
    arm, state, rep, seed = b["model_id"], b["state"], b["replicate_idx"], b["random_seed"]
    if row["random_seed"] != seed:
      raise ValueError(f"seed mismatch for {arm}/{state}/rep{rep}")
    inp = row["sampling_spec"]["inputs"][0]
    tgt = targets[inp]
    done = os.path.exists(row["output_h5_path"] + ".done.json")
    completion.append({"unit": f"{arm}|{state}|seed{seed}", "status": "done" if done else None})
    sink = read_sink(row)
    rows_read += 1
    seqs =sink["sequences"][:, 0]  # (sample, temperature, residue); grid axis has size 1 here
    logits = sink["logits"][:, 0]
    temps = row["temperature"]
    temps_seen.add(tuple(temps))
    if seqs.shape[1] != len(temps):
      raise ValueError(f"temperature axis {seqs.shape[1]} != row temperatures {temps}")
    mask = designable_mask(len(tgt.native), sink["fixed_mask"])
    n_fixed_real = int(sink["fixed_mask"][: len(tgt.native)].astype(bool).sum())
    designable_ok &= int(mask.sum()) == len(tgt.native) - n_fixed_real

    rec = recovery(seqs, tgt.native, mask)
    shuffled = tgt.native.copy()
    idx = np.where(mask[: len(shuffled)])[0]
    shuffled[idx] = shuffled[rng.permutation(idx)]
    rec_shuf = recovery(seqs, shuffled, mask)
    nll = derived_nll(seqs, logits, mask)
    ident = pairwise_identity(seqs, mask)
    arm_counts[arm] += np.bincount(seqs[..., mask].ravel(), minlength=21)[:21]
    arm_rec[arm].append(float(rec.mean()))
    arm_shuf[arm].append(float(rec_shuf.mean()))
    for ti, t in enumerate(temps):
      if arm in INSTRUMENT_ARMS:
        true_min = min(true_min, float(rec[:, ti].mean()))
        shuffled_max = max(shuffled_max, float(rec_shuf[:, ti].mean()))
      per_seed.append({
        "arm": ARM_LABELS[arm], "state": state, "seed": f"seed {seed}", "temperature": f"{t:.1f}",
        "recovery": round(float(rec[:, ti].mean()), 6),
        "recovery_shuffled_native": round(float(rec_shuf[:, ti].mean()), 6),
        "nll_per_residue": round(float(nll[:, ti].mean()), 6),
        "within_library_identity": round(float(ident[ti]), 6),
      })
    if state not in fixed_rows:
      fidx = np.where(sink["fixed_mask"].astype(bool))[0]
      fixed_rows[state] = [{
        "state": state, "slot": f"fixed {k + 1}", "index": int(i),
        "native_residue": tgt.residue_label[i] if i < len(tgt.native) else "padding",
        "pinned_to": MPNN_ORDER[int(sink["fixed_tokens"][i])],
        "native_matches_pin": int(i < len(tgt.native) and tgt.native[i] == sink["fixed_tokens"][i]),
      } for k, i in enumerate(fidx)]
    tev_probs[arm][state][rep] = tev_mean_probs(sink["logits"])
    if arm == args.residue_track_arm and state == args.residue_track_state and rep == 0:
      ent = mean_prob_entropy_bits(sink["logits"])
      entropy_track = [{"residue": i + 1, "entropy_bits": (None if sink["fixed_mask"][i] else round(float(ent[i]), 6))} for i in range(len(tgt.native))]

  # Positive control
  ref = json.loads(args.reference_results.read_text())["per_model"]
  pos_rows, max_abs_diff = [], 0.0
  for arm in sorted(tev_probs):
    floor, offdiag = tev_floor_effect(tev_probs[arm])
    for name, got in (("floor", floor), ("mean_offdiag_jsd", offdiag)):
      want = float(ref[arm][name])
      max_abs_diff = max(max_abs_diff, abs(got - want))
      pos_rows.append({"arm": arm, "statistic": name, "reproduced": got, "recorded": want, "abs_diff": abs(got - want)})

  # Grid tables (seed-mean, then state-mean) for sweep panels
  cell: dict[tuple, list[dict]] = defaultdict(list)
  for r in per_seed:
    cell[(r["arm"], r["temperature"])].append(r)
  grid = [{"arm": a, "temperature": t,
           "recovery": round(float(np.mean([x["recovery"] for x in v])), 6),
           "nll_per_residue": round(float(np.mean([x["nll_per_residue"] for x in v])), 6),
           "within_library_identity": round(float(np.mean([x["within_library_identity"] for x in v])), 6)}
          for (a, t), v in sorted(cell.items())]
  if len(temps_seen) != 1:
    raise ValueError(f"rows disagree on the temperature axis: {temps_seen}")
  low_t = min(next(iter(temps_seen)))
  pooled = [{"arm": r["arm"], "state": r["state"], "seed": r["seed"], "recovery": r["recovery"]}
            for r in per_seed if float(r["temperature"]) == low_t]
  fixed = [x for state in sorted(fixed_rows) for x in fixed_rows[state]]

  out = args.out_dir
  write_csv(out / "library_by_seed.csv", per_seed)
  write_csv(out / "library_grid.csv", grid)
  write_csv(out / "recovery_lowest_temperature.csv", pooled)
  write_csv(out / "fixed_position_check.csv", fixed)
  write_csv(out / "completion.csv", completion)
  write_csv(out / "positive_control.csv", pos_rows)
  if entropy_track is None:
    raise ValueError("residue track arm/state not found")
  write_csv(out / "residue_entropy.csv", entropy_track)
  (out / "targets.json").write_text(json.dumps({"parser_identity": parser_identity, "targets": {p: {"sha256": t.sha256, "length": len(t.native)} for p, t in targets.items()}}, indent=2))

  # Degeneracy per arm: composition entropy of designable tokens and recovery margin over shuffle.
  arm_rows = []
  for arm in sorted(arm_counts):
    c = arm_counts[arm] / arm_counts[arm].sum()
    ent = float(-(c[c > 0] * np.log2(c[c > 0])).sum())
    margin = float(np.mean(arm_rec[arm]) - np.mean(arm_shuf[arm]))
    arm_rows.append({"arm": ARM_LABELS[arm], "composition_entropy_bits": round(ent, 4),
                     "recovery": round(float(np.mean(arm_rec[arm])), 4),
                     "recovery_shuffled_native": round(float(np.mean(arm_shuf[arm])), 4),
                     "recovery_margin": round(margin, 4), "degenerate_flag": int(margin < 0.10)})
  write_csv(out / "arm_degeneracy.csv", arm_rows)
  degenerate = [r["arm"] for r in arm_rows if r["degenerate_flag"]]

  states_bad = sorted({x["state"] for x in fixed if not x["native_matches_pin"]})
  results = {
    "rows_read": rows_read,
    "rows_total": len(rows),
    "positive_control_max_abs_diff": max_abs_diff,
    "positive_control_ok": max_abs_diff <= 1e-6,
    "negative_control_shuffled_recovery_max": shuffled_max,
    "true_recovery_min": true_min,
    "negative_control_ok": shuffled_max <= 0.12 and true_min >= 0.25,
    "designable_mask_ok": bool(designable_ok),
    "row_knobs_detected": ",".join(sorted(knobs)),
    "fixed_pin_differs_from_native_states": ",".join(states_bad),
    "n_fixed_pin_differs_from_native_states": len(states_bad),
    "degenerate_arms": ";".join(degenerate),
    "n_degenerate_arms": len(degenerate),
    "parser_aminx": parser_identity["aminx"],
    "parser_proxide": parser_identity["proxide"],
  }
  results["all_rows_ok"] = results["rows_read"] == results["rows_total"]
  log.info("results: %s", json.dumps(results))
  (out / "results.json").write_text(json.dumps(results, indent=2))
  rp = os.environ.get("BTH_RESULTS_PATH")
  if rp:
    Path(rp).write_text(json.dumps(results))
  return 0


if __name__ == "__main__":
  sys.exit(main())
