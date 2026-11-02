#!/usr/bin/env python3
"""Step-by-step feature and encoder parity between upstream and aminx."""

import os
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
ref_root = Path("/home/marielle/projects/aminx/reference_ligandmpnn_clone")
sys.path.insert(0, str(ref_root))
sys.path.insert(0, str(repo_root / "scripts" / "parity"))

import e2e_run_api_parity as e2e

def main():
  os.environ["REFERENCE_PATH"] = str(ref_root)
  torch, model_utils, _ = e2e._import_reference_torch()

  ctx = e2e.ref_context("lmpnn", "base", batch=1)
  fd = ctx["fd"]
  ref_model = ctx["model"]
  order = ctx["out"]["decoding_order"][0].numpy()
  length = fd["mask"].shape[1]

  from aminx.io.weights import load_model
  aminx_model = load_model("ligandmpnn_v_32_020_25")

  # 1. Compare Features
  with torch.no_grad():
    V_ref, E_ref, E_idx_ref, Y_nodes_ref, Y_edges_ref, Y_m_ref = ref_model.features(fd)

  # Call aminx features directly
  import jax
  import jax.numpy as jnp

  coords = jnp.asarray(fd["X"][0].numpy())
  mask = jnp.asarray(fd["mask"][0].numpy())
  residue_index = jnp.asarray(fd["R_idx"][0].numpy())
  chain_index = jnp.asarray(fd["chain_labels"][0].numpy())
  Y = jnp.asarray(fd["Y"][0].numpy())
  Y_t = jnp.asarray(fd["Y_t"][0].numpy())
  Y_m = jnp.asarray(fd["Y_m"][0].numpy())

  V_ax, E_ax, E_idx_ax, Y_n_ax, Y_e_ax, Y_m_ax = aminx_model.features(
    _key=jax.random.PRNGKey(0),
    structure_coordinates=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    ligand_coords=Y,
    ligand_atom_types=Y_t,
    ligand_mask=Y_m,
  )

  print("=== FEATURE PARITY ===")
  print(f"E_idx max diff: {np.abs(E_idx_ref[0].numpy() - np.asarray(E_idx_ax)).max()}")
  print(f"E max diff:     {np.abs(E_ref[0].numpy() - np.asarray(E_ax)).max():.3e}")
  print(f"V max diff:     {np.abs(V_ref[0].numpy() - np.asarray(V_ax)).max():.3e}")
  print(f"Y_nodes max diff: {np.abs(Y_nodes_ref[0].numpy() - np.asarray(Y_n_ax)).max():.3e}")
  print(f"Y_edges max diff: {np.abs(Y_edges_ref[0].numpy() - np.asarray(Y_e_ax)).max():.3e}")
  print(f"Y_m max diff:     {np.abs(Y_m_ref[0].numpy() - np.asarray(Y_m_ax)).max():.3e}")

  with torch.no_grad():
    h_E_ref = ref_model.W_e(E_ref)
  print(f"E after W_e max diff: {np.abs(h_E_ref[0].numpy() - np.asarray(E_ax)).max():.3e}")

  # 2. Compare Backbone Encoder
  with torch.no_grad():
    h_V_ref = torch.zeros((E_ref.shape[0], E_ref.shape[1], E_ref.shape[-1]))
    mask_ref = fd["mask"]
    mask_attend = model_utils.gather_nodes(mask_ref.unsqueeze(-1), E_idx_ref).squeeze(-1)
    mask_attend = mask_ref.unsqueeze(-1) * mask_attend
    for layer in ref_model.encoder_layers:
      h_V_ref, h_E_ref = layer(h_V_ref, h_E_ref, E_idx_ref, mask_ref, mask_attend)

  # Aminx backbone encoder
  h_V_ax = jnp.zeros((E_ax.shape[0], aminx_model.node_features_dim))
  h_E_ax = E_ax
  mask_2d = mask[:, None] * mask[None, :]
  mask_attend_ax = jnp.take_along_axis(mask_2d, E_idx_ax.astype(jnp.int32), axis=1)
  for layer in aminx_model.encoder.layers:
    h_V_ax, h_E_ax = layer(h_V_ax, h_E_ax, E_idx_ax, mask, mask_attend_ax, inference=True)

  print("\n=== BACKBONE ENCODER PARITY ===")
  print(f"Backbone h_E max diff: {np.abs(h_E_ref[0].numpy() - np.asarray(h_E_ax)).max():.3e}")
  print(f"Backbone h_V max diff: {np.abs(h_V_ref[0].numpy() - np.asarray(h_V_ax)).max():.3e}")

  # 3. Compare Context Encoder
  with torch.no_grad():
    h_V_C_ref = ref_model.W_c(h_V_ref)
    h_E_context_ref = ref_model.W_v(V_ref)
    Y_m_edges_ref = Y_m_ref[:, :, :, None] * Y_m_ref[:, :, None, :]
    Y_nodes_ref = ref_model.W_nodes_y(Y_nodes_ref)
    Y_edges_ref = ref_model.W_edges_y(Y_edges_ref)
    for i in range(len(ref_model.context_encoder_layers)):
      Y_nodes_ref = ref_model.y_context_encoder_layers[i](
        Y_nodes_ref, Y_edges_ref, Y_m_ref, Y_m_edges_ref
      )
      h_E_context_cat_ref = torch.cat([h_E_context_ref, Y_nodes_ref], -1)
      h_V_C_ref = ref_model.context_encoder_layers[i](
        h_V_C_ref, h_E_context_cat_ref, mask_ref, Y_m_ref
      )
    h_V_final_ref = h_V_ref + ref_model.V_C_norm(ref_model.V_C(h_V_C_ref))

  # Aminx context encoder
  h_V_C_ax = jax.vmap(aminx_model.w_c)(h_V_ax)
  h_E_context_ax = jax.vmap(jax.vmap(aminx_model.w_v))(V_ax)
  Y_m_edges_ax = Y_m_ax[..., None] * Y_m_ax[..., None, :]
  Y_nodes_ax = jax.vmap(jax.vmap(aminx_model.w_nodes_y))(Y_n_ax)
  Y_edges_ax = jax.vmap(jax.vmap(jax.vmap(aminx_model.w_edges_y)))(Y_e_ax)

  print("\n=== CONTEXT ENCODER INITIAL PROJECTIONS ===")
  print(f"h_V_C initial max diff:     {np.abs(h_V_C_ref[0].numpy() - np.asarray(h_V_C_ax)).max():.3e}")
  print(f"h_E_context initial max diff: {np.abs(h_E_context_ref[0].numpy() - np.asarray(h_E_context_ax)).max():.3e}")
  print(f"Y_nodes initial max diff:   {np.abs(Y_nodes_ref[0].numpy() - np.asarray(Y_nodes_ax)).max():.3e}")
  print(f"Y_edges initial max diff:   {np.abs(Y_edges_ref[0].numpy() - np.asarray(Y_edges_ax)).max():.3e}")

  for i in range(len(aminx_model.context_encoder)):
    Y_nodes_ax = jax.vmap(
      lambda node, edge, mask_l, mask_e: aminx_model.y_context_encoder[i](
        node, edge, mask_l, attention_mask=mask_e, inference=True
      ),
    )(Y_nodes_ax, Y_edges_ax, Y_m_ax, Y_m_edges_ax)
    h_E_context_cat_ax = jnp.concatenate([h_E_context_ax, Y_nodes_ax], axis=-1)
    h_V_C_ax = aminx_model.context_encoder[i](
      h_V_C_ax, h_E_context_cat_ax, mask, attention_mask=Y_m_ax, inference=True
    )
    print(f"\n--- Layer {i} ---")
    print(f"Y_nodes layer {i} max diff: {np.abs(Y_nodes_ref[0].numpy() - np.asarray(Y_nodes_ax)).max():.3e}")
    print(f"h_V_C layer {i} max diff:   {np.abs(h_V_C_ref[0].numpy() - np.asarray(h_V_C_ax)).max():.3e}")

  h_V_final_ref = h_V_ref + ref_model.V_C_norm(ref_model.V_C(h_V_C_ref))
  h_V_final_ax = h_V_ax + jax.vmap(aminx_model.v_c_norm)(jax.vmap(aminx_model.v_c)(h_V_C_ax))
  print("\n=== FINAL ENCODER OUTPUT ===")
  print(f"Final h_V max diff: {np.abs(h_V_final_ref[0].numpy() - np.asarray(h_V_final_ax)).max():.3e}")

if __name__ == "__main__":
  main()
