#!/usr/bin/env python3
"""Test call_conditional with L=93 vs L=96 padding."""

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
    p0 = int(order[0])

    fd2 = dict(fd)
    fd2["S"] = fd["S"]
    fd2["batch_size"] = 1
    fd2["randn"] = ctx["randn_row"]
    with torch.no_grad():
        ref_score = ref_model.score(fd2, use_sequence=True)
        ref_score_lp = ref_score["log_probs"].numpy()[0]
        ref_score_logits = ref_score["logits"].numpy()[0]

    import jax
    import jax.numpy as jnp
    from aminx.io.weights import load_model
    aminx_model = load_model("ligandmpnn_v_32_020_25")

    # 1. Unpadded L=93
    coords_93 = jnp.asarray(fd["X"][0].numpy())
    mask_93 = jnp.asarray(fd["mask"][0].numpy())
    r_idx_93 = jnp.asarray(fd["R_idx"][0].numpy())
    chain_93 = jnp.asarray(fd["chain_labels"][0].numpy())
    Y = jnp.asarray(fd["Y"][0].numpy())
    Y_t = jnp.asarray(fd["Y_t"][0].numpy())
    Y_m = jnp.asarray(fd["Y_m"][0].numpy())

    # 2. Padded L=96
    coords_96 = jnp.pad(coords_93, ((0, 3), (0, 0), (0, 0)))
    mask_96 = jnp.pad(mask_93, (0, 3), constant_values=0.0)
    r_idx_96 = jnp.pad(r_idx_93, (0, 3), constant_values=100)
    chain_96 = jnp.pad(chain_93, (0, 3), constant_values=1)
    Y_96 = jnp.pad(Y, ((0, 3), (0, 0), (0, 0)))
    Y_t_96 = jnp.pad(Y_t, ((0, 3), (0, 0)))
    Y_m_96 = jnp.pad(Y_m, ((0, 3), (0, 0)))

    # Encode both:
    def encode(coords, mask, r_idx, chain, y, y_t, y_m):
        V, E, E_idx, Y_n, Y_e, Y_m_ = aminx_model.features(
            _key=jax.random.PRNGKey(0),
            structure_coordinates=coords,
            mask=mask,
            residue_index=r_idx,
            chain_index=chain,
            ligand_coords=y,
            ligand_atom_types=y_t,
            ligand_mask=y_m,
        )
        h_V = jnp.zeros((E.shape[0], aminx_model.node_features_dim))
        h_E = E
        mask_2d = mask[:, None] * mask[None, :]
        mask_attend = jnp.take_along_axis(mask_2d, E_idx.astype(jnp.int32), axis=1)
        for layer in aminx_model.encoder.layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend, inference=True)

        h_V_C = jax.vmap(aminx_model.w_c)(h_V)
        h_E_context = jax.vmap(jax.vmap(aminx_model.w_v))(V)
        Y_m_edges = Y_m_[..., None] * Y_m_[..., None, :]
        Y_nodes = jax.vmap(jax.vmap(aminx_model.w_nodes_y))(Y_n)
        Y_edges = jax.vmap(jax.vmap(jax.vmap(aminx_model.w_edges_y)))(Y_e)
        for i in range(len(aminx_model.context_encoder)):
            Y_nodes = jax.vmap(
                lambda node, edge, mask_l, mask_e: aminx_model.y_context_encoder[i](
                    node, edge, mask_l, attention_mask=mask_e, inference=True
                ),
            )(Y_nodes, Y_edges, Y_m_, Y_m_edges)
            h_E_context_cat = jnp.concatenate([h_E_context, Y_nodes], axis=-1)
            h_V_C = aminx_model.context_encoder[i](
                h_V_C, h_E_context_cat, mask, attention_mask=Y_m_, inference=True
            )
        h_V_final = h_V + jax.vmap(aminx_model.v_c_norm)(jax.vmap(aminx_model.v_c)(h_V_C))
        return h_V_final, h_E, E_idx

    h_V_93, h_E_93, E_idx_93 = encode(coords_93, mask_93, r_idx_93, chain_93, Y, Y_t, Y_m)
    h_V_96, h_E_96, E_idx_96 = encode(coords_96, mask_96, r_idx_96, chain_96, Y_96, Y_t_96, Y_m_96)

    print(f"Encoder h_V diff on [:93]: {np.abs(h_V_93 - h_V_96[:93]).max():.6e}")

    # Build ar_mask for 93 and 96
    from aminx.utils.autoregression import ar_mask_from_decoding_order
    order_93 = jnp.asarray(order)
    order_96 = jnp.concatenate([order_93, jnp.arange(93, 96, dtype=jnp.int32)])

    ar_mask_93 = ar_mask_from_decoding_order(order_93)
    ar_mask_96 = ar_mask_from_decoding_order(order_96)

    seq_93 = jnp.asarray(fd["S"][0].numpy())
    seq_96 = jnp.pad(seq_93, (0, 3), constant_values=0)

    oh_93 = jax.nn.one_hot(seq_93, 21)
    oh_96 = jax.nn.one_hot(seq_96, 21)

    dec_93 = aminx_model.decoder.call_conditional(
        h_V_93, h_E_93, E_idx_93, mask_93, ar_mask_93, oh_93, aminx_model.w_s_embed.weight, inference=True
    )
    dec_96 = aminx_model.decoder.call_conditional(
        h_V_96, h_E_96, E_idx_96, mask_96, ar_mask_96, oh_96, aminx_model.w_s_embed.weight, inference=True
    )

    lp_93 = jax.nn.log_softmax(jax.vmap(aminx_model.w_out)(dec_93), axis=-1)
    lp_96 = jax.nn.log_softmax(jax.vmap(aminx_model.w_out)(dec_96), axis=-1)

    diff_93_vs_ref = np.abs(np.asarray(lp_93)[:, :20] - ref_score_lp[:, :20]).max()
    diff_96_vs_ref = np.abs(np.asarray(lp_96[:93])[:, :20] - ref_score_lp[:, :20]).max()
    diff_93_vs_96 = np.abs(np.asarray(lp_93) - np.asarray(lp_96[:93])).max()

    print(f"Diff 93 vs ref_score: {diff_93_vs_ref:.6e}")
    print(f"Diff 96 vs ref_score: {diff_96_vs_ref:.6e}")
    print(f"Diff 93 vs 96:        {diff_93_vs_96:.6e}")

if __name__ == "__main__":
    main()
