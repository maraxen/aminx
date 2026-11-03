#!/usr/bin/env python3
"""Compare plan.encode vs manual encode."""

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
    _, _, _ = e2e._import_reference_torch()
    ctx = e2e.ref_context("lmpnn", "base", batch=1)
    fd = ctx["fd"]
    order = ctx["out"]["decoding_order"][0].numpy()
    p0 = int(order[0])

    kw = e2e.aminx_kwargs("lmpnn", ctx, temperature=0.1)

    from aminx.run.specs import SamplingSpecification
    from aminx.host.prep import prep_protein_stream_and_model
    from aminx.host.plan import make_inference_plan
    from aminx.inference.bundle_builder import build_inference_bundle
    from aminx.host.kernel_dispatch import _structure_bundle_kwargs, _prepare_ligand_context, _prepare_fixed_controls
    import jax
    import jax.numpy as jnp

    spec = SamplingSpecification(**kw, num_samples=1, random_seed=5, decoding_order_fn=e2e._order_fn(order, 93))
    protein_iterator, model = prep_protein_stream_and_model(spec)
    batched_ensemble = next(iter(protein_iterator))
    plan = make_inference_plan(model, spec, purpose="sample")

    per_structure = {
        "coords": jnp.asarray(batched_ensemble.coordinates),
        "mask": jnp.asarray(batched_ensemble.mask),
        "residue_index": jnp.asarray(batched_ensemble.residue_index),
        "chain_index": jnp.asarray(batched_ensemble.chain_index),
    }
    fixed_mask, fixed_tokens = _prepare_fixed_controls(spec, batched_ensemble=batched_ensemble, batch_structure_ids=None)
    ligand_context = _prepare_ligand_context(spec, batched_ensemble=batched_ensemble, batch_size=1, seq_len=96, canonical_structure_ids=None, batch_structure_ids=None)
    per_structure["fixed_mask"] = fixed_mask
    per_structure["fixed_tokens"] = fixed_tokens

    structure_kwargs = _structure_bundle_kwargs(
        0, per_structure=per_structure, ligand_context=ligand_context,
        shared={"bias": None, "state_weights": None, "mode": "sample_ar", "inference": True}
    )
    bundle, config = build_inference_bundle(**structure_kwargs)
    enc = plan.encode(bundle, jax.random.PRNGKey(0), config)

    # Manual encode from fd:
    coords_93 = jnp.asarray(fd["X"][0].numpy())
    mask_93 = jnp.asarray(fd["mask"][0].numpy())
    r_idx_93 = jnp.asarray(fd["R_idx"][0].numpy())
    chain_93 = jnp.asarray(fd["chain_labels"][0].numpy())
    Y = jnp.asarray(fd["Y"][0].numpy())
    Y_t = jnp.asarray(fd["Y_t"][0].numpy())
    Y_m = jnp.asarray(fd["Y_m"][0].numpy())

    from aminx.io.weights import load_model
    aminx_model = load_model("ligandmpnn_v_32_020_25")
    V, E, E_idx, Y_n, Y_e, Y_m_ = aminx_model.features(
        _key=jax.random.PRNGKey(0),
        structure_coordinates=coords_93,
        mask=mask_93,
        residue_index=r_idx_93,
        chain_index=chain_93,
        ligand_coords=Y,
        ligand_atom_types=Y_t,
        ligand_mask=Y_m,
    )
    h_V = jnp.zeros((E.shape[0], aminx_model.node_features_dim))
    h_E = E
    mask_2d = mask_93[:, None] * mask_93[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, E_idx.astype(jnp.int32), axis=1)
    for layer in aminx_model.encoder.layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask_93, mask_attend, inference=True)
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
            h_V_C, h_E_context_cat, mask_93, attention_mask=Y_m_, inference=True
        )
    h_V_final_manual = h_V + jax.vmap(aminx_model.v_c_norm)(jax.vmap(aminx_model.v_c)(h_V_C))

    h_V_final_plan = np.asarray(enc.node_features[0, :93])
    diff_h_V = np.abs(h_V_final_manual - h_V_final_plan)
    print(f"Max diff h_V (manual vs plan.encode): {diff_h_V.max():.6e}")
    print(f"Diff h_V at p0 ({p0}): {diff_h_V[p0].max():.6e}")
    print(f"Worst residue in h_V diff: {diff_h_V.max(axis=-1).argmax()}, max diff: {diff_h_V.max():.6e}")

    # Check coords difference:
    coords_plan = np.asarray(batched_ensemble.coordinates[0, :93, :4])
    print(f"Coords diff: {np.abs(coords_plan - fd['X'][0].numpy()).max():.6e}")

    # Check ligand coords difference:
    Y_plan = np.asarray(bundle.ligand.ligand_coords[0, :93])
    print(f"Ligand coords diff: {np.abs(Y_plan - fd['Y'][0].numpy()).max():.6e}")

    # Check ligand mask difference:
    Y_m_plan = np.asarray(bundle.ligand.ligand_mask[0, :93])
    print(f"Ligand mask diff: {np.abs(Y_m_plan - fd['Y_m'][0].numpy()).max():.6e}")

    # Check ligand atom types difference:
    Y_t_plan = np.asarray(bundle.ligand.ligand_atom_types[0, :93])
    print(f"Ligand atom types diff: {np.abs(Y_t_plan - fd['Y_t'][0].numpy()).max():.6e}")

if __name__ == "__main__":
    main()
