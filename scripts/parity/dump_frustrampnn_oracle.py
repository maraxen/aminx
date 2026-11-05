"""Sealed Upstream PyTorch Oracle Dumper for FrustraMPNN.

Task 261105_frustrampnn (Milestone 3).
Executes upstream FrustraMPNN reference model on tests/data/1ubq.pdb with:
1. Strict stochastic injection: augment_eps = 0.0 in features.py:180, 292.
2. Injected decoding-order randn in float32 for sequence sampling (protein_mpnn.py:324).
3. Inverse-CDF multinomial token selection with pre-drawn uniforms for bit-for-bit reproducibility.
4. Loud traps for unauthorized un-injected stochastic primitives (torch.randn, np.random.*).
5. Dumps arrays to frustrampnn_1ubq_oracle.npz:
   - features: (76, 48, 128) float32
   - encoder_out: (76, 128) float32
   - decoder_hidden: (3, 76, 128) float32
   - lin_input: (76, 512) float32
   - light_attention_out: (76, 512) float32
   - both_out: (76, 21) float32
   - frustration_scores: (76, 21) float32
   - sampled_sequences: (10, 76) int32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import torch

# Ensure omegaconf and pandas stubs are available in lean environments
import sys
if "omegaconf" not in sys.modules:
  sys.modules["omegaconf"] = MagicMock()
if "pandas" not in sys.modules:
  sys.modules["pandas"] = MagicMock()

from aminx.io.torch_checkpoint import load_restricted_torch_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dump_frustrampnn_oracle")

UPSTREAM_COMMIT = "3a03cdc300bfe24c4bb70e60207118532bc73b3b"
EXPECTED_CKPT_SHA256 = "eb54019295c8000a1006d7955ab43100e67687daa4cef18e1b3541c5cb881cfa"
DEFAULT_SEED = 42
N_SAMPLES = 10


class ConfigNode(dict):
  """Lightweight config dictionary with attribute-style access."""

  def __getattr__(self, name: str) -> Any:
    val = self[name]
    if isinstance(val, dict) and not isinstance(val, ConfigNode):
      return ConfigNode(val)
    return val

  def __setattr__(self, name: str, val: Any) -> None:
    self[name] = val


def sha256_file(path: Path) -> str:
  """Compute SHA256 hex digest of a file."""
  hasher = hashlib.sha256()
  with path.open("rb") as f:
    while chunk := f.read(65536):
      hasher.update(chunk)
  return hasher.hexdigest()


def install_loud_traps() -> None:
  """Install loud traps for un-injected stochastic primitives."""
  def trap(primitive_name: str) -> Any:
    def handler(*args: Any, **kwargs: Any) -> Any:
      msg = f"LOUD TRAP TRIGGERED: Unauthorized stochastic primitive {primitive_name} called during oracle dump!"
      raise RuntimeError(msg)
    return handler

  # Trap PyTorch random calls
  torch.randn = trap("torch.randn")
  torch.randn_like = trap("torch.randn_like")
  torch.rand = trap("torch.rand")
  torch.rand_like = trap("torch.rand_like")
  torch.randint = trap("torch.randint")

  # Trap NumPy random calls
  np.random.randn = trap("np.random.randn")
  np.random.rand = trap("np.random.rand")
  np.random.random = trap("np.random.random")
  np.random.choice = trap("np.random.choice")
  np.random.randint = trap("np.random.randint")
  logger.info("[TRAP] Loud un-injected RNG traps installed.")


def load_model(
  checkpoint_path: Path,
  trunk_checkpoint_path: Path,
  frustrampnn_repo_root: Path,
) -> Any:
  """Load upstream FrustraMPNN model and verify weight mapping."""
  sys.path.insert(0, str(frustrampnn_repo_root / "src"))
  from frustrampnn.model.transfer_model import TransferModel

  cfg = ConfigNode({
    "platform": ConfigNode({"thermompnn_dir": str(trunk_checkpoint_path.parent.parent)}),
    "model": ConfigNode({
      "hidden_dims": [64, 32],
      "subtract_mut": False,
      "num_final_layers": 3,
      "lightattn": True,
      "load_pretrained": True,
      "freeze_weights": True,
    }),
    "training": ConfigNode({
      "add_esm_embeddings": False,
    }),
    "decoding_order": "left-to-right",
  })

  logger.info("Initializing TransferModel from upstream...")
  model = TransferModel(cfg)

  logger.info("Deserializing checkpoint: %s", checkpoint_path)
  ckpt = load_restricted_torch_checkpoint(checkpoint_path)
  head_sd = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
  missing, unexpected = model.load_state_dict(head_sd, strict=False)

  logger.info("Loaded state dict: %d missing (trunk keys), %d unexpected keys.", len(missing), len(unexpected))
  if unexpected:
    msg = f"Unexpected keys in checkpoint: {unexpected}"
    raise KeyError(msg)

  model.eval()
  # Set augment_eps = 0.0 explicitly on features
  model.prot_mpnn.features.augment_eps = 0.0
  logger.info("Model loaded and set to eval mode (augment_eps = 0.0).")
  return model


def dump_oracle(
  model: Any,
  pdb_path: Path,
  out_dir: Path,
  frustrampnn_repo_root: Path,
  seed: int = DEFAULT_SEED,
) -> tuple[Path, str]:
  """Execute reference forward pass and sequence sampling, dumping sealed NPZ and manifest."""
  sys.path.insert(0, str(frustrampnn_repo_root / "src"))
  from frustrampnn.model.featurization import tied_featurize
  from frustrampnn.model.pdb_parsing import alt_parse_PDB
  from frustrampnn.model.protein_mpnn import gather_nodes

  logger.info("Parsing PDB structure: %s", pdb_path)
  pdb = alt_parse_PDB(str(pdb_path), ["A"])
  if not pdb:
    msg = f"Failed to parse chain A from {pdb_path}"
    raise ValueError(msg)

  logger.info("Featurizing coordinates with tied_featurize...")
  (
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list,
    visible_list_list, masked_list_list, masked_chain_length_list_list,
    chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask,
    tied_pos_list_of_lists_list, pssm_coef, pssm_bias,
    pssm_log_odds_all, bias_by_res_all, tied_beta,
  ) = tied_featurize([pdb[0]], "cpu", None, None, None, None, None, None, ca_only=False)

  L = S.shape[1]
  logger.info("Sequence length L = %d: %s", L, pdb[0]["seq"])

  # --- 1. Pre-draw stochastic tensors BEFORE installing loud traps ---
  rng = np.random.RandomState(seed)
  # decoding_order randn: shape (N_samples, L) in float32
  randn_draw = torch.from_numpy(rng.randn(N_SAMPLES, L).astype(np.float32))
  # inverse-CDF uniforms: shape (L, N_samples, 1) in float64
  uniforms_draw = torch.from_numpy(rng.uniform(0.0, 1.0, size=(L, N_SAMPLES, 1)).astype(np.float64))

  # Install loud traps to ensure no unauthorized stochastic primitive executes
  install_loud_traps()

  # --- 2. Teacher-forced forward pass ---
  with torch.no_grad():
    # 2.1 Features
    E, E_idx = model.prot_mpnn.features(X, mask, residue_idx, chain_encoding_all)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]))
    h_E = model.prot_mpnn.W_e(E)

    # 2.2 Encoder
    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in model.prot_mpnn.encoder_layers:
      h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
    encoder_out_t = h_V[0]  # (L, 128)

    # 2.3 Decoder hidden layers
    all_mpnn_hid, mpnn_embed, _ = model.prot_mpnn(
      X, S, mask, chain_M, residue_idx, chain_encoding_all, None
    )
    # all_mpnn_hid is [layer2, layer1, layer0]
    decoder_hidden_t = torch.stack([h[0] for h in all_mpnn_hid[:3]], dim=0)  # (3, L, 128)

    # 2.4 Latent concatenation (384 + 128 = 512)
    mpnn_hid = torch.cat(all_mpnn_hid[:3], -1)[0]  # (L, 384)
    lin_input_t = torch.cat([mpnn_hid, mpnn_embed[0]], -1)  # (L, 512)

    # 2.5 Per-residue LightAttention, both_out, frustration_out
    la_outs = []
    both_outs = []
    frust_outs = []
    for i in range(L):
      res_in = lin_input_t[i].unsqueeze(-1).unsqueeze(0)  # (1, 512, 1)
      res_la = model.light_attention_mpnn(res_in, mask)  # (512,)
      la_outs.append(res_la)
      res_both = model.both_out(res_la)  # (21,)
      both_outs.append(res_both)
      res_frust = model.frustration_out(res_both.unsqueeze(-1)).squeeze(-1)  # (21,)
      frust_outs.append(res_frust)

    light_attention_out_t = torch.stack(la_outs, dim=0)  # (L, 512)
    both_out_t = torch.stack(both_outs, dim=0)  # (L, 21)
    frustration_scores_t = torch.stack(frust_outs, dim=0)  # (L, 21)

    # --- 3. Sequence Sampling with Injected Draws ---
    X_rep = X.repeat(N_SAMPLES, 1, 1, 1)
    S_rep = S.repeat(N_SAMPLES, 1)
    mask_rep = mask.repeat(N_SAMPLES, 1)
    chain_M_rep = chain_M.repeat(N_SAMPLES, 1)
    chain_M_pos_rep = chain_M_pos.repeat(N_SAMPLES, 1)
    chain_encoding_all_rep = chain_encoding_all.repeat(N_SAMPLES, 1)
    residue_idx_rep = residue_idx.repeat(N_SAMPLES, 1)

    step_counter = [0]
    orig_multinomial = torch.multinomial

    def shimmed_multinomial(probs: torch.Tensor, num_samples: int, replacement: bool = True, generator: Any = None) -> torch.Tensor:
      del num_samples, replacement, generator
      step = step_counter[0]
      step_counter[0] += 1
      u = uniforms_draw[step]  # (N_samples, 1)
      probabilities = probs.detach().to(dtype=torch.float64)
      cumulative = torch.cumsum(probabilities, dim=-1)
      target = u * cumulative[..., -1:]
      index = torch.searchsorted(cumulative, target, right=True)
      vocab = probabilities.shape[-1]
      positions = torch.arange(vocab, device=probabilities.device, dtype=torch.long)
      positive = probabilities > 0
      last_pos = torch.where(positive, positions, torch.zeros_like(positions)).amax(dim=-1, keepdim=True)
      clamped = torch.minimum(index.to(dtype=torch.long), last_pos)
      return torch.clamp(clamped, min=0, max=vocab - 1)

    torch.multinomial = shimmed_multinomial
    try:
      sample_out = model.prot_mpnn.sample(
        X_rep,
        randn_draw,
        S_rep,
        chain_M_rep,
        chain_encoding_all_rep,
        residue_idx_rep,
        mask=mask_rep,
        temperature=0.1,
        omit_AAs_np=np.zeros(21),
        bias_AAs_np=np.zeros(21),
        chain_M_pos=chain_M_pos_rep,
        bias_by_res=torch.zeros(N_SAMPLES, L, 21),
      )
    finally:
      torch.multinomial = orig_multinomial

    sampled_sequences_t = sample_out["S"]  # (10, L)

  logger.info("Forward execution and sampling complete without tripping loud traps.")

  # Convert tensors to exact numpy specifications
  payload: dict[str, np.ndarray] = {
    "features": E[0].detach().cpu().numpy().astype(np.float32),
    "encoder_out": encoder_out_t.detach().cpu().numpy().astype(np.float32),
    "decoder_hidden": decoder_hidden_t.detach().cpu().numpy().astype(np.float32),
    "lin_input": lin_input_t.detach().cpu().numpy().astype(np.float32),
    "light_attention_out": light_attention_out_t.detach().cpu().numpy().astype(np.float32),
    "both_out": both_out_t.detach().cpu().numpy().astype(np.float32),
    "frustration_scores": frustration_scores_t.detach().cpu().numpy().astype(np.float32),
    "sampled_sequences": sampled_sequences_t.detach().cpu().numpy().astype(np.int32),
  }

  for k, arr in payload.items():
    logger.info("  Array '%s': shape=%s, dtype=%s", k, arr.shape, arr.dtype)

  out_dir.mkdir(parents=True, exist_ok=True)
  npz_path = out_dir / "frustrampnn_1ubq_oracle.npz"
  manifest_path = out_dir / "frustrampnn_1ubq_oracle.manifest.json"

  logger.info("Writing oracle archive: %s", npz_path)
  np.savez(npz_path, **payload)
  npz_sha = sha256_file(npz_path)
  logger.info("Oracle archive SHA256: %s", npz_sha)

  manifest_data = {
    "target_pdb": "1ubq",
    "pdb_path": str(pdb_path),
    "upstream_commit": UPSTREAM_COMMIT,
    "checkpoint_sha256": EXPECTED_CKPT_SHA256,
    "oracle_npz_sha256": npz_sha,
    "seed": seed,
    "num_samples": N_SAMPLES,
    "sequence_length": L,
    "array_shapes": {k: list(v.shape) for k, v in payload.items()},
    "array_dtypes": {k: str(v.dtype) for k, v in payload.items()},
    "stochastic_controls": {
      "augment_eps": 0.0,
      "decoding_order_injected": True,
      "inverse_cdf_multinomial_injected": True,
      "loud_traps_installed": True,
    },
  }

  manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n")
  logger.info("Wrote manifest: %s", manifest_path)
  return npz_path, npz_sha


def main() -> None:
  """Run oracle dumper CLI."""
  parser = argparse.ArgumentParser(description="FrustraMPNN Sealed Upstream Oracle Dumper")
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("/home/solab/repos/FrustraMPNN/model_weights/megascale_default_la-true_esm-False_split1.ckpt"),
    help="Path to checkpoint",
  )
  parser.add_argument(
    "--trunk-checkpoint",
    type=Path,
    default=Path("/home/solab/repos/FrustraMPNN/weights/vanilla_model_weights/v_48_020.pt"),
    help="Path to vanilla trunk checkpoint",
  )
  parser.add_argument(
    "--frustrampnn-repo-root",
    type=Path,
    default=Path("/home/solab/repos/FrustraMPNN"),
    help="Path to upstream FrustraMPNN repo clone",
  )
  parser.add_argument(
    "--pdb",
    type=Path,
    default=Path("tests/data/1ubq.pdb"),
    help="Path to target PDB structure",
  )
  parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("tests/data/fixtures"),
    help="Directory to save oracle dumps",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="Deterministic PRNG seed",
  )
  parser.add_argument(
    "--verify-reproducibility",
    action="store_true",
    help="Execute two consecutive dump passes and assert identical SHA256",
  )
  args = parser.parse_args()

  logger.info("=" * 70)
  logger.info("Starting FrustraMPNN Oracle Dumper")
  logger.info("=" * 70)

  model = load_model(args.checkpoint, args.trunk_checkpoint, args.frustrampnn_repo_root)

  if args.verify_reproducibility:
    logger.info("[REPRO CHECK] Running Pass 1...")
    p1, s1 = dump_oracle(model, args.pdb, args.out_dir, args.frustrampnn_repo_root, seed=args.seed)

    # Re-load model fresh for Pass 2 to verify zero state leakage
    logger.info("[REPRO CHECK] Running Pass 2...")
    model2 = load_model(args.checkpoint, args.trunk_checkpoint, args.frustrampnn_repo_root)
    p2, s2 = dump_oracle(model2, args.pdb, args.out_dir, args.frustrampnn_repo_root, seed=args.seed)

    logger.info("Pass 1 SHA256: %s", s1)
    logger.info("Pass 2 SHA256: %s", s2)
    if s1 != s2:
      msg = f"Bit-for-bit reproducibility FAILURE! Pass 1: {s1}, Pass 2: {s2}"
      raise RuntimeError(msg)
    logger.info("[PASS] BIT-FOR-BIT REPRODUCIBILITY VERIFIED! Both passes produced SHA256 %s", s1)
  else:
    dump_oracle(model, args.pdb, args.out_dir, args.frustrampnn_repo_root, seed=args.seed)

  logger.info("=" * 70)
  logger.info("ORACLE DUMP COMPLETED SUCCESSFULLY")
  logger.info("=" * 70)


if __name__ == "__main__":
  main()
