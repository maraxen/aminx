"""Sealed upstream oracle dump for AntiBMPNN (run on titanix, torch env; never in CI).

Imports the REAL upstream code from a clone of https://github.com/zeysun/AntiBMPNN
(``protein_mpnn_utils.py``, ``helper_scripts/*.py``, ``Running_AntiBMPNN_run.py``) and the REAL
weights ``antibmpnn_model_weights/antibmpnn_*.pt``, runs them on the upstream example antibody
``inputs/HuJ3/HuJ3.pdb`` (one chain, B, 121 residues) and writes ONE sealed ``.npz`` plus a
``.manifest.json`` carrying every input sha256, the upstream commit and the sha256 of the dump.

Injection points (all stochastic steps are driven by arrays stored in the dump):

* decoding order: the ``randn`` argument of ``ProteinMPNN.forward`` (protein_mpnn_utils.py:1057, used
  at :1082) and of ``ProteinMPNN.sample`` (:1104, used at :1119). We pass seeded ``randn`` tensors and
  store them, so the order is ``argsort((chain_M*chain_M_pos*mask + 1e-4) * |randn|)`` on both sides.
* categorical draw: ``torch.multinomial(probs, 1)`` at protein_mpnn_utils.py:1180 is replaced, for the
  duration of ``sample`` only, by a Gumbel-max draw ``argmax(log(probs) + g[step])`` with stored
  Gumbel noise ``g``. Everything that produces ``probs`` (temperature, omit, bias, bias_by_res,
  omit_AA_mask, decoder layers) is upstream code, unmodified.
* float64: upstream hard-codes ``.float()`` at protein_mpnn_utils.py:734 and :1084 (and :1121), which
  breaks a float64 model. For the float64 pass only, ``torch.Tensor.float`` is patched to return
  float64. No upstream source line is edited. ``all_probs`` (:1130) is float32 upstream, so sampler
  probabilities are graded at the float32 band only.

The end-to-end CLI is also executed once (``Running_AntiBMPNN_run.py`` ``main`` with only
``numpy.random.randint`` pinned, because the script ignores ``--seed``: lines 26-30 draw a random
seed) to produce a real ``.fa``; ``dump_antibmpnn_descriptors_oracle.py`` consumes that file.

Usage (titanix)::

    python scripts/parity/dump_antibmpnn_oracle.py --antibmpnn-root /home/solab/repos/AntiBMPNN \\
        --out-dir /home/solab/scratch/p4-antibmpnn/oracle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("dump_antibmpnn_oracle")

SEED = 20261104
CHECKPOINTS = tuple(f"antibmpnn_{n:03d}" for n in range(0, 101, 10))
#: the checkpoint that also carries the big per-edge tensors (features / encoder edges)
PRIMARY = "antibmpnn_000"
DESIGN_POSITIONS = "102 103 104 105 106"
CONFIGS = {
  # example_scripts.sh: chain B, positions 102-106 designed, T=0.1, default omit X
  "cdr5": {"temperature": 0.1, "omit": "X", "bias_aa": {}, "per_res_bias": False, "omit_per_pos": False, "fixed": True},
  # all positions designed with every sampler knob live
  "knobs": {
    "temperature": 0.3,
    "omit": "CWX",
    "bias_aa": {"A": -1.1, "F": 0.7},
    "per_res_bias": True,
    "omit_per_pos": True,
    "fixed": False,
  },
}
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


@contextmanager
def float_as_double():  # noqa: ANN201
  """Make the unmodified upstream forward run in float64 (see module docstring)."""
  import torch

  original = torch.Tensor.float
  torch.Tensor.float = lambda self, *a, **k: self.double()  # type: ignore[method-assign]
  prev = torch.get_default_dtype()
  torch.set_default_dtype(torch.float64)
  try:
    yield
  finally:
    torch.Tensor.float = original  # type: ignore[method-assign]
    torch.set_default_dtype(prev)


@contextmanager
def gumbel_draw(noise: np.ndarray):  # noqa: ANN201
  """Replace ``torch.multinomial`` (protein_mpnn_utils.py:1180) by a Gumbel-max draw on stored noise."""
  import torch

  original = torch.multinomial
  state = {"step": 0}

  def _draw(probs: Any, num_samples: int = 1, *a: Any, **k: Any) -> Any:  # noqa: ANN401
    g = torch.tensor(noise[state["step"]], dtype=probs.dtype)[None, :]
    state["step"] += 1
    return torch.argmax(torch.log(probs) + g, dim=-1, keepdim=True)

  torch.multinomial = _draw  # type: ignore[assignment]
  try:
    yield
  finally:
    torch.multinomial = original  # type: ignore[assignment]


def build_model(pu: Any, ckpt: dict[str, Any], *, double: bool) -> Any:
  """Mirror Running_AntiBMPNN_run.py:163-170."""
  model = pu.ProteinMPNN(
    ca_only=False,
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    augment_eps=0.0,
    k_neighbors=ckpt["num_edges"],
  )
  model.load_state_dict(ckpt["model_state_dict"])
  model.eval()
  return model.double() if double else model


def featurize(pu: Any, out: Path, root: Path, *, cfg: dict[str, Any], L_hint: int | None = None) -> dict[str, Any]:
  import torch

  dataset = pu.StructureDataset(str(out / "parsed.jsonl"), truncate=None, max_length=200000, verbose=False)
  chain_id_dict = json.loads((out / "assigned.jsonl").read_text())
  fixed = json.loads((out / "fixed.jsonl").read_text()) if cfg["fixed"] else None
  protein = dataset[0]
  length = len(protein["seq"])
  rng = np.random.default_rng(SEED + 1)
  bias_by_res = None
  if cfg["per_res_bias"]:
    bias_by_res = {protein["name"]: {"B": (0.5 * rng.standard_normal((length, 21))).tolist()}}
  omit_dict = None
  if cfg["omit_per_pos"]:
    omit_dict = {protein["name"]: {"B": [[list(range(10, 21)), "GP"]]}}
  names = (
    "X S mask lengths chain_M chain_encoding_all chain_list_list visible_list_list masked_list_list "
    "masked_chain_length_list_list chain_M_pos omit_AA_mask residue_idx dihedral_mask "
    "tied_pos_list_of_lists_list pssm_coef pssm_bias pssm_log_odds_all bias_by_res_all tied_beta"
  ).split()
  feats = pu.tied_featurize(
    [protein], torch.device("cpu"), chain_id_dict, fixed, omit_dict, None, None, bias_by_res, ca_only=False
  )
  return dict(zip(names, feats, strict=True))


def sample_once(pu: Any, model: Any, f: dict[str, Any], cfg: dict[str, Any], randn: Any, noise: np.ndarray, *, double: bool) -> dict[str, Any]:
  import torch

  omit_np = np.array([aa in cfg["omit"] for aa in ALPHABET]).astype(np.float32)
  bias_np = np.zeros(21)
  for n, aa in enumerate(ALPHABET):
    if aa in cfg["bias_aa"]:
      bias_np[n] = cfg["bias_aa"][aa]
  dt = torch.float64 if double else torch.float32
  omit_mask = f["omit_AA_mask"] if cfg["omit_per_pos"] else None
  # pssm_log_odds_mask etc. mirror the runner defaults (pssm_* off)
  pssm_log_odds_mask = (f["pssm_log_odds_all"] > 0.0).float()
  with gumbel_draw(noise), torch.no_grad():
    return model.sample(
      f["X"].to(dt),
      randn.to(dt),
      f["S"],
      f["chain_M"].to(dt),
      f["chain_encoding_all"],
      f["residue_idx"],
      mask=f["mask"].to(dt),
      temperature=cfg["temperature"],
      omit_AAs_np=omit_np,
      bias_AAs_np=bias_np,
      chain_M_pos=f["chain_M_pos"].to(dt),
      omit_AA_mask=omit_mask,
      pssm_coef=f["pssm_coef"],
      pssm_bias=f["pssm_bias"],
      pssm_multi=0.0,
      pssm_log_odds_flag=False,
      pssm_log_odds_mask=pssm_log_odds_mask,
      pssm_bias_flag=False,
      bias_by_res=f["bias_by_res_all"].to(dt),
    )


def run_checkpoint(pu: Any, root: Path, name: str, feats: dict[str, dict[str, Any]], dump: dict[str, np.ndarray]) -> None:
  import torch

  ckpt = torch.load(root / "antibmpnn_model_weights" / f"{name}.pt", map_location="cpu", weights_only=False)
  dump[f"{name}/num_edges"] = np.asarray(ckpt["num_edges"])
  dump[f"{name}/noise_level"] = np.asarray(float(ckpt["noise_level"]))
  for precision in ("f32", "f64"):
    double = precision == "f64"
    ctx = float_as_double() if double else _nullcontext()
    with ctx:
      model = build_model(pu, ckpt, double=double)
      dt = torch.float64 if double else torch.float32
      for cfg_name, cfg in CONFIGS.items():
        f = feats[cfg_name]
        gen = torch.Generator().manual_seed(SEED)
        randn1 = torch.randn(f["chain_M"].shape, generator=gen)
        randn2 = torch.randn(f["chain_M"].shape, generator=gen)
        noise_rng = np.random.default_rng(SEED + 2)
        noise = noise_rng.gumbel(size=(f["X"].shape[1], 21))
        pre = f"{name}/{precision}/{cfg_name}"
        X, S, mask = f["X"].to(dt), f["S"], f["mask"].to(dt)
        chain_m = (f["chain_M"] * f["chain_M_pos"]).to(dt)
        with torch.no_grad():
          # forward stages: mirror protein_mpnn_utils.py:1061-1069 using the model's own modules
          E, E_idx = model.features(X, mask, f["residue_idx"], f["chain_encoding_all"])
          h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), dtype=dt)
          h_E = model.W_e(E)
          h_E0 = h_E.clone()
          mask_attend = pu.gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
          mask_attend = mask.unsqueeze(-1) * mask_attend
          for layer in model.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
          log_probs = model(X, S, mask, chain_m, f["residue_idx"], f["chain_encoding_all"], randn1.to(dt))
          order1 = torch.argsort((chain_m * mask + 0.0001) * torch.abs(randn1.to(dt)))
          log_probs_order = model(
            X, S, mask, chain_m, f["residue_idx"], f["chain_encoding_all"], randn1.to(dt),
            use_input_decoding_order=True, decoding_order=order1,
          )
          assert torch.equal(log_probs, log_probs_order), "decoding-order recomputation disagrees with upstream forward"
        dump[f"{pre}/E_idx"] = E_idx.numpy()
        dump[f"{pre}/h_V_enc"] = h_V.numpy()
        dump[f"{pre}/order_tf"] = order1.numpy()
        dump[f"{pre}/log_probs_tf"] = log_probs.numpy()
        if name == PRIMARY and cfg_name == "cdr5":
          dump[f"{pre}/E"] = E.numpy()
          dump[f"{pre}/h_E0"] = h_E0.numpy()
          dump[f"{pre}/h_E_enc"] = h_E.numpy()
        if double:
          # upstream sample() allocates all_probs as float32 (protein_mpnn_utils.py:1130) and scatters a
          # float64 slice into it (:1181), so the sampler is dumped and graded in float32 only.
          continue
        out = sample_once(pu, model, f, cfg, randn2, noise, double=double)
        S_new = out["S"]
        with torch.no_grad():
          lp_new = model(
            X, S_new, mask, chain_m, f["residue_idx"], f["chain_encoding_all"], randn2.to(dt),
            use_input_decoding_order=True, decoding_order=out["decoding_order"],
          )
        dump[f"{pre}/S_sample"] = S_new.numpy()
        dump[f"{pre}/probs_sample"] = out["probs"].numpy()
        dump[f"{pre}/order_sample"] = out["decoding_order"].numpy()
        dump[f"{pre}/log_probs_sample"] = lp_new.numpy()
        dump[f"{pre}/randn1"] = randn1.numpy()
        dump[f"{pre}/randn2"] = randn2.numpy()
        dump[f"{pre}/gumbel"] = noise
    log.info("%s %s done", name, precision)


class _nullcontext:  # noqa: N801 - tiny local contextmanager
  def __enter__(self) -> None:
    return None

  def __exit__(self, *exc: object) -> None:
    return None


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--antibmpnn-root", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--checkpoints", default=",".join(CHECKPOINTS))
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  import torch

  root = args.antibmpnn_root.resolve()
  out = args.out_dir.resolve()
  out.mkdir(parents=True, exist_ok=True)
  sys.path.insert(0, str(root))
  import protein_mpnn_utils as pu  # the REAL upstream module

  assert Path(pu.__file__).resolve() == root / "protein_mpnn_utils.py"
  torch.set_num_threads(4)

  pdb = root / "inputs" / "HuJ3" / "HuJ3.pdb"
  py = sys.executable
  helper = root / "helper_scripts"
  subprocess.run([py, str(helper / "parse_multiple_chains.py"), "--input_path", str(pdb.parent) + "/", "--output_path", str(out / "parsed.jsonl")], check=True)
  subprocess.run([py, str(helper / "assign_fixed_chains.py"), "--input_path", str(out / "parsed.jsonl"), "--output_path", str(out / "assigned.jsonl"), "--chain_list", "B"], check=True)
  subprocess.run([py, str(helper / "make_fixed_positions_dict.py"), "--input_path", str(out / "parsed.jsonl"), "--output_path", str(out / "fixed.jsonl"), "--chain_list", "B", "--position_list", DESIGN_POSITIONS, "--specify_non_fixed"], check=True)

  dump: dict[str, np.ndarray] = {}
  # parse stage: the upstream parser's own output (aminx.io.upstream_pdb.parse_pdb_upstream is graded against it)
  parsed = pu.parse_PDB(str(pdb))[0]
  for atom in ("N", "CA", "C", "O"):
    dump[f"parse/{atom}_chain_B"] = np.asarray(parsed["coords_chain_B"][f"{atom}_chain_B"], dtype=np.float64)
  dump["parse/seq_chain_B"] = np.asarray(parsed["seq_chain_B"])

  feats: dict[str, dict[str, Any]] = {}
  for cfg_name, cfg in CONFIGS.items():
    f = featurize(pu, out, root, cfg=cfg)
    feats[cfg_name] = f
    for key in ("X", "S", "mask", "chain_M", "chain_M_pos", "residue_idx", "chain_encoding_all", "omit_AA_mask", "bias_by_res_all"):
      dump[f"feat/{cfg_name}/{key}"] = f[key].numpy()
    dump[f"feat/{cfg_name}/temperature"] = np.asarray(cfg["temperature"])
    dump[f"feat/{cfg_name}/omit"] = np.asarray(cfg["omit"])
    dump[f"feat/{cfg_name}/bias_aa"] = np.asarray([cfg["bias_aa"].get(a, 0.0) for a in ALPHABET])

  for name in args.checkpoints.split(","):
    run_checkpoint(pu, root, name, feats, dump)

  # End-to-end CLI run (real Running_AntiBMPNN_run.py main), only numpy.random.randint pinned.
  cli_dir = out / "cli_run"
  cli = _run_real_cli(root, out, cli_dir)
  dump.update(cli)

  dump_path = out / "antibmpnn_oracle.npz"
  np.savez_compressed(dump_path, **dump)
  commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
  manifest = {
    "dump": dump_path.name,
    "dump_sha256": sha256_file(dump_path),
    "upstream_commit": commit,
    "upstream_root_name": root.name,
    "protein_mpnn_utils_sha256": sha256_file(root / "protein_mpnn_utils.py"),
    "runner_sha256": sha256_file(root / "Running_AntiBMPNN_run.py"),
    "fixture": "inputs/HuJ3/HuJ3.pdb",
    "fixture_sha256": sha256_file(pdb),
    "checkpoints": {n: sha256_file(root / "antibmpnn_model_weights" / f"{n}.pt") for n in args.checkpoints.split(",")},
    "seed": SEED,
    "cli_fa_sha256": sha256_file(next((cli_dir / "seqs").glob("*.fa"))),
    "torch": torch.__version__,
  }
  (out / "antibmpnn_oracle.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  log.info("wrote %s sha256=%s", dump_path, manifest["dump_sha256"])


def _run_real_cli(root: Path, out: Path, cli_dir: Path) -> dict[str, np.ndarray]:
  """Run the real runner's ``main`` (Running_AntiBMPNN_run.py:5) with the seed draw (:26) pinned."""
  import importlib.util
  import os

  spec = importlib.util.spec_from_file_location("antibmpnn_runner", root / "Running_AntiBMPNN_run.py")
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  args = argparse.Namespace(
    suppress_print=1, ca_only=False, path_to_model_weights="", model_name="antibmpnn_000", use_soluble_model=False,
    seed=0, save_score=1, save_probs=1, score_only=0, path_to_fasta="", conditional_probs_only=0,
    conditional_probs_only_backbone=0, unconditional_probs_only=0, backbone_noise=0.0, num_seq_per_target=20,
    batch_size=10, max_length=200000, sampling_temp="0.1", out_folder=str(cli_dir), pdb_path="",
    pdb_path_chains="", jsonl_path=str(out / "parsed.jsonl"), chain_id_jsonl=str(out / "assigned.jsonl"),
    fixed_positions_jsonl=str(out / "fixed.jsonl"), omit_AAs="X", bias_AA_jsonl="", bias_by_res_jsonl="",
    omit_AA_jsonl="", pssm_jsonl="", pssm_multi=0.0, pssm_threshold=0.0, pssm_log_odds_flag=0, pssm_bias_flag=0,
    tied_positions_jsonl="",
  )
  original = np.random.randint
  np.random.randint = lambda *a, **k: np.asarray([SEED % 999])  # type: ignore[assignment]
  cwd = os.getcwd()
  os.chdir(root / "example")  # the runner hard-codes '../antibmpnn_model_weights' (line 35)
  try:
    module.main(args)
  finally:
    os.chdir(cwd)
    np.random.randint = original  # type: ignore[assignment]
  fa = next((cli_dir / "seqs").glob("*.fa"))
  lines = fa.read_text().strip().split("\n")
  return {"cli/fa_lines": np.asarray(lines), "cli/score_global": np.load(cli_dir / "scores" / "HuJ3.npz")["global_score"]}


if __name__ == "__main__":
  main()
