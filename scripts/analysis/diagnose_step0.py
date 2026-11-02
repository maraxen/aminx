#!/usr/bin/env python3
"""Diagnose Step 0 decoder activations between aminx and PyTorch."""

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
    designed = e2e._designed(fd)
    p0 = int(order[0])
    print(f"Total length: {length}, Designed: {designed.sum()}, Step 0 position (p0): {p0}")

    kw = e2e.aminx_kwargs("lmpnn", ctx, temperature=0.1)
    seqs, aminx_logits = e2e.aminx_sample(kw, num_samples=1, seed=5, order=order, length=length)

    fd2 = dict(fd)
    fd2["S"] = torch.as_tensor(seqs[:1].astype(np.int64))
    fd2["batch_size"] = 1
    fd2["randn"] = ctx["randn_row"]
    with torch.no_grad():
        ref_score = ref_model.score(fd2, use_sequence=True)
        ref_score_lp = ref_score["log_probs"].numpy()[0]
        ref_score_logits = ref_score["logits"].numpy()[0]

    # In PyTorch, what is sample() producing?
    with torch.no_grad():
        ref_sample = ref_model.sample(fd)
        ref_sample_lp = ref_sample["log_probs"].numpy()[0]
        # In sample, all_log_probs is [B, L, 21]
    
    aminx_lp = e2e._lsm(aminx_logits[0])

    print("\n--- AT STEP 0 (POSITION p0) ---")
    print(f"Ref score logprobs[:5]:   {ref_score_lp[p0, :5]}")
    print(f"Ref sample logprobs[:5]:  {ref_sample_lp[p0, :5]}")
    print(f"Aminx sample logprobs[:5]: {aminx_lp[p0, :5]}")

    diff_score_sample = np.abs(ref_score_lp[p0, :20] - ref_sample_lp[p0, :20])
    diff_aminx_score = np.abs(aminx_lp[p0, :20] - ref_score_lp[p0, :20])
    diff_aminx_sample = np.abs(aminx_lp[p0, :20] - ref_sample_lp[p0, :20])

    print(f"Diff ref_score vs ref_sample at p0: {diff_score_sample.max():.6e}")
    print(f"Diff aminx vs ref_score at p0:      {diff_aminx_score.max():.6e}")
    print(f"Diff aminx vs ref_sample at p0:     {diff_aminx_sample.max():.6e}")

    # Now let's compare raw logits
    print("\n--- RAW LOGITS AT p0 ---")
    print(f"Ref score logits[:5]:   {ref_score_logits[p0, :5]}")
    print(f"Aminx returned logits[:5]: {aminx_logits[0, p0, :5]}")
    print(f"Notice aminx logits are shifted by: {aminx_logits[0, p0, :5] - ref_score_logits[p0, :5]}")

    # Now let's test what aminx model.decoder.call_conditional gives on the initial sequence at p0!
    import jax
    import jax.numpy as jnp
    from aminx.io.weights import load_model
    aminx_model = load_model("ligandmpnn_v_32_020_25")

    # Initial sequence in aminx sample is all UNDRAWN_TOKEN (-1)
    init_seq_oh = jax.nn.one_hot(jnp.full((length,), -1, dtype=jnp.int32), 21)
    
    # Run aminx features & encoder
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
    # Backbone encoder
    h_V_ax = jnp.zeros((E_ax.shape[0], aminx_model.node_features_dim))
    h_E_ax = E_ax
    mask_2d = mask[:, None] * mask[None, :]
    mask_attend_ax = jnp.take_along_axis(mask_2d, E_idx_ax.astype(jnp.int32), axis=1)
    for layer in aminx_model.encoder.layers:
        h_V_ax, h_E_ax = layer(h_V_ax, h_E_ax, E_idx_ax, mask, mask_attend_ax, inference=True)

    # Context encoder
    h_V_C_ax = jax.vmap(aminx_model.w_c)(h_V_ax)
    h_E_context_ax = jax.vmap(jax.vmap(aminx_model.w_v))(V_ax)
    Y_m_edges_ax = Y_m_ax[..., None] * Y_m_ax[..., None, :]
    Y_nodes_ax = jax.vmap(jax.vmap(aminx_model.w_nodes_y))(Y_n_ax)
    Y_edges_ax = jax.vmap(jax.vmap(jax.vmap(aminx_model.w_edges_y)))(Y_e_ax)
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
    h_V_final_ax = h_V_ax + jax.vmap(aminx_model.v_c_norm)(jax.vmap(aminx_model.v_c)(h_V_C_ax))

    # Build ar_mask using ar_mask_from_decoding_order
    from aminx.utils.autoregression import ar_mask_from_decoding_order
    order_jax = jnp.asarray(order)
    ar_mask = ar_mask_from_decoding_order(order_jax)

    # Now call call_conditional with init_seq_oh
    dec_step0 = aminx_model.decoder.call_conditional(
        h_V_final_ax,
        h_E_ax,
        E_idx_ax,
        mask,
        ar_mask,
        init_seq_oh,
        aminx_model.w_s_embed.weight,
        inference=True,
    )
    logits_step0 = jax.vmap(aminx_model.w_out)(dec_step0)
    lp_step0 = np.asarray(jax.nn.log_softmax(logits_step0, axis=-1))

    print(f"\n--- MANUAL call_conditional WITH init_seq_oh AT p0 ---")
    print(f"Logits[:5]: {logits_step0[p0, :5]}")
    print(f"Logprobs[:5]: {lp_step0[p0, :5]}")
    print(f"Diff vs ref_score at p0:  {np.abs(lp_step0[p0, :20] - ref_score_lp[p0, :20]).max():.6e}")
    print(f"Diff vs aminx_sample at p0: {np.abs(lp_step0[p0, :20] - aminx_lp[p0, :20]).max():.6e}")

    # Now call call_conditional with the FULL sampled sequence (teacher forcing)
    full_seq_oh = jax.nn.one_hot(jnp.asarray(seqs[0]), 21)
    dec_full = aminx_model.decoder.call_conditional(
        h_V_final_ax,
        h_E_ax,
        E_idx_ax,
        mask,
        ar_mask,
        full_seq_oh,
        aminx_model.w_s_embed.weight,
        inference=True,
    )
    logits_full = jax.vmap(aminx_model.w_out)(dec_full)
    lp_full = np.asarray(jax.nn.log_softmax(logits_full, axis=-1))

    print(f"\n--- MANUAL call_conditional WITH full_seq_oh (TEACHER FORCING) ---")
    print(f"Max diff vs ref_score ACROSS ALL 93 DESIGNED POSITIONS:")
    diff_all = np.abs(lp_full[designed, :20] - ref_score_lp[designed, :20])
    print(f"Max abs logprob diff: {diff_all.max():.6e}")
    print(f"Diff at p0: {diff_all[0].max():.6e}")
    print(f"Diff at p91: {np.abs(lp_full[91, :20] - ref_score_lp[91, :20]).max():.6e}")

if __name__ == "__main__":
    main()
