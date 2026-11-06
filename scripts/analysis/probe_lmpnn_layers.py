#!/usr/bin/env python3
"""Activation parity bisection probe for aminx vs LigandMPNN reference on 1BC8.

Compares:
1. Inputs & features: coordinates, ligand context, E_idx.
2. Encoder: backbone encoder, context encoder, V_C fusion, final h_V.
3. Decoder at step 0 (rank 0, position 57).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Setup paths
repo_root = Path(__file__).resolve().parent.parent.parent
ref_root = Path("/home/marielle/projects/aminx/reference_ligandmpnn_clone")
sys.path.insert(0, str(ref_root))
sys.path.insert(0, str(repo_root / "scripts" / "parity"))

import e2e_run_api_parity as e2e

def main():
  os.environ["REFERENCE_PATH"] = str(ref_root)
  torch, model_utils, _ = e2e._import_reference_torch()
  print("=" * 70)
  print("LMPNN Bisection Probe: aminx vs upstream LigandMPNN")
  print("=" * 70)

  # 1. Load upstream reference context
  ctx = e2e.ref_context("lmpnn", "base", batch=1)
  fd = ctx["fd"]
  ref_model = ctx["model"]
  order = ctx["out"]["decoding_order"][0].numpy()
  length = fd["mask"].shape[1]
  designed = e2e._designed(fd)
  print(f"Sequence length: {length}, Designed: {designed.sum()}")
  print(f"Rank 0 position: {order[0]}")

  # 2. Extract upstream intermediate activations
  with torch.no_grad():
    # Features
    V_ref, E_ref, E_idx_ref, Y_nodes_ref, Y_edges_ref, Y_m_ref = ref_model.features(fd)
    
    # Backbone encoder
    h_V_ref = torch.zeros((E_ref.shape[0], E_ref.shape[1], E_ref.shape[-1]))
    h_E_ref = ref_model.W_e(E_ref)
    h_E_context_ref = ref_model.W_v(V_ref)

    mask = fd["mask"]
    mask_attend = model_utils.gather_nodes(mask.unsqueeze(-1), E_idx_ref).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in ref_model.encoder_layers:
      h_V_ref, h_E_ref = layer(h_V_ref, h_E_ref, E_idx_ref, mask, mask_attend)

    # Context encoder
    h_V_C_ref = ref_model.W_c(h_V_ref)
    Y_m_edges_ref = Y_m_ref[:, :, :, None] * Y_m_ref[:, :, None, :]
    Y_nodes_ref = ref_model.W_nodes_y(Y_nodes_ref)
    Y_edges_ref = ref_model.W_edges_y(Y_edges_ref)
    for i in range(len(ref_model.context_encoder_layers)):
      Y_nodes_ref = ref_model.y_context_encoder_layers[i](
        Y_nodes_ref, Y_edges_ref, Y_m_ref, Y_m_edges_ref
      )
      h_E_context_cat_ref = torch.cat([h_E_context_ref, Y_nodes_ref], -1)
      h_V_C_ref = ref_model.context_encoder_layers[i](
        h_V_C_ref, h_E_context_cat_ref, mask, Y_m_ref
      )

    h_V_final_ref = h_V_ref + ref_model.V_C_norm(ref_model.V_C(h_V_C_ref))

    # Single-step decoder pass under sequence (Lane T teacher forcing)
    # aminx 1 sample
    kw = e2e.aminx_kwargs("lmpnn", ctx, temperature=0.1)
    seqs, aminx_logits = e2e.aminx_sample(kw, num_samples=1, seed=5, order=order, length=length)

    fd2 = dict(fd)
    fd2["S"] = torch.as_tensor(seqs[:1].astype(np.int64))
    fd2["batch_size"] = 1
    fd2["randn"] = ctx["randn_row"]
    ref_score_out = ref_model.score(fd2, use_sequence=True)
    ref_lp = ref_score_out["log_probs"].numpy()[0]
    ref_logits = ref_score_out["logits"].numpy()[0]

  # 3. Load aminx model directly and extract intermediate representations
  from aminx.io.weights import load_model
  aminx_model = load_model("ligandmpnn_v_32_020_25")
  print("\nAminx model loaded successfully:", type(aminx_model))

  # Compare aminx weights vs reference checkpoint
  cp = ref_root / "model_params" / "ligandmpnn_v_32_020_25.pt"
  ckpt = torch.load(cp, map_location="cpu")
  sd = ckpt["model_state_dict"]

  # Check W_out weight
  w_out_ref = sd["W_out.weight"].numpy() # (21, 128)
  w_out_aminx = np.asarray(aminx_model.w_out.weight)
  print(f"W_out weight shape: ref={w_out_ref.shape}, aminx={w_out_aminx.shape}")
  print(f"W_out weight max diff: {np.abs(w_out_ref - w_out_aminx).max():.3e}")

  w_s_ref = sd["W_s.weight"].numpy() # (128, 21) or (21, 128)
  w_s_aminx = np.asarray(aminx_model.w_s_embed.weight)
  print(f"W_s weight shape: ref={w_s_ref.shape}, aminx={w_s_aminx.shape}")
  if w_s_ref.shape != w_s_aminx.shape:
    w_s_ref = w_s_ref.T
  print(f"W_s weight max diff: {np.abs(w_s_ref - w_s_aminx).max():.3e}")

  # Compare encoder outputs
  from aminx.inference.encode import make_encode_fn
  encode_fn = make_encode_fn(aminx_model)
  # Prepare aminx bundle
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.types.configs import InferenceConfig
  # Run bundle builder
  from aminx.host.runner import prep_protein_stream_and_model
  from aminx.run.specs import SamplingSpecification
  spec = SamplingSpecification(**kw, num_samples=1, random_seed=5, decoding_order_fn=e2e._order_fn(order, length))
  stream, _ = prep_protein_stream_and_model(spec)
  batch = next(iter(stream))
  bundle = build_inference_bundle(batch, InferenceConfig(), model=aminx_model)
  enc_out = encode_fn(bundle.states[0], bundle.coordinates[0], bundle.mask[0])
  h_V_aminx = np.asarray(enc_out.node_features)[:length]
  h_E_aminx = np.asarray(enc_out.edge_features)[:length, :32]
  E_idx_aminx = np.asarray(enc_out.neighbor_indices)[:length, :32]

  print("\n--- Level 3 & 4 Parity Checks ---")
  print(f"E_idx max diff: {np.abs(E_idx_ref[0].numpy() - E_idx_aminx).max()}")
  print(f"Encoder h_E max diff: {np.abs(h_E_ref[0].numpy() - h_E_aminx).max():.3e}")
  print(f"Encoder h_V (final) max diff: {np.abs(h_V_final_ref[0].numpy() - h_V_aminx).max():.3e}")

  # Compare logit differences at Rank 0 (Position 57)
  p0 = order[0]
  aminx_lp = e2e._lsm(aminx_logits[0])
  diff_p0 = np.abs(aminx_lp[p0, :20] - ref_lp[p0, :20])
  print(f"\n--- Position {p0} (Rank 0) Detailed Inspection ---")
  print(f"Ref logprobs[:5]:   {ref_lp[p0, :5]}")
  print(f"Aminx logprobs[:5]: {aminx_lp[p0, :5]}")
  print(f"Diff[:5]:           {diff_p0[:5]}")
  print(f"Max diff at pos {p0}: {diff_p0.max():.6f}")

  # Check position 91 (the worst position)
  diff_p91 = np.abs(aminx_lp[91, :20] - ref_lp[91, :20])
  print(f"\n--- Position 91 (Rank 85, worst) Detailed Inspection ---")
  print(f"Ref logprobs[:5]:   {ref_lp[91, :5]}")
  print(f"Aminx logprobs[:5]: {aminx_lp[91, :5]}")
  print(f"Diff[:5]:           {diff_p91[:5]}")
  print(f"Max diff at pos 91: {diff_p91.max():.6f}")


if __name__ == "__main__":
  main()
