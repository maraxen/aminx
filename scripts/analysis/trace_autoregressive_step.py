#!/usr/bin/env python3
"""Trace exact autoregressive steps in aminx to see where divergence happens."""

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

    kw = e2e.aminx_kwargs("lmpnn", ctx, temperature=0.1)
    
    # We load aminx model and prepare the plan and bundle
    from aminx.run.specs import SamplingSpecification
    from aminx.host.prep import prep_protein_stream_and_model
    from aminx.host.plan import make_inference_plan
    from aminx.inference.bundle_builder import build_inference_bundle, with_decoding_order
    from aminx.host.kernel_dispatch import _structure_bundle_kwargs, decoding_order_key
    import jax
    import jax.numpy as jnp

    spec = SamplingSpecification(**kw, num_samples=1, random_seed=5, decoding_order_fn=e2e._order_fn(order, length))
    protein_iterator, model = prep_protein_stream_and_model(spec)
    batched_ensemble = next(iter(protein_iterator))
    plan = make_inference_plan(model, spec, purpose="sample")

    # Build bundle as kernel_dispatch does:
    per_structure = {
        "coords": jnp.asarray(batched_ensemble.coordinates),
        "mask": jnp.asarray(batched_ensemble.mask),
        "residue_index": jnp.asarray(batched_ensemble.residue_index),
        "chain_index": jnp.asarray(batched_ensemble.chain_index),
    }
    from aminx.host.kernel_dispatch import _prepare_ligand_context, _prepare_fixed_controls
    fixed_mask_for_vmap, fixed_tokens_for_vmap = _prepare_fixed_controls(
        spec,
        batched_ensemble=batched_ensemble,
        batch_structure_ids=None,
    )
    ligand_context = _prepare_ligand_context(
        spec,
        batched_ensemble=batched_ensemble,
        batch_size=1,
        seq_len=batched_ensemble.coordinates.shape[1],
        canonical_structure_ids=None,
        batch_structure_ids=None,
    )
    per_structure["fixed_mask"] = fixed_mask_for_vmap
    per_structure["fixed_tokens"] = fixed_tokens_for_vmap
    
    structure_kwargs = _structure_bundle_kwargs(
        0,
        per_structure=per_structure,
        ligand_context=ligand_context,
        shared={
            "bias": jnp.asarray(spec.run_spec.sampling.bias, dtype=jnp.float32) if spec.run_spec.sampling.bias is not None else None,
            "state_weights": None,
            "mode": "sample_ar",
            "inference": True,
        },
    )

    bundle, config = build_inference_bundle(
        **structure_kwargs,
        backbone_noise=jnp.asarray(spec.run_spec.sampling.backbone_noise),
        temperature=jnp.asarray(spec.run_spec.sampling.temperature),
    )

    base_key = jax.random.PRNGKey(spec.run_spec.sampling.random_seed)
    encode_key = jax.random.fold_in(base_key, 0)
    enc = plan.encode(bundle, encode_key, config)

    # Apply order to bundle
    sample_key = jax.random.fold_in(base_key, 0) # for sample 0
    order_key = decoding_order_key(sample_key)
    ordered_bundle = with_decoding_order(bundle, order_key, e2e._order_fn(order, length))

    # Now let's inspect enc and ordered_bundle
    print(f"enc.node_features shape: {enc.node_features.shape}")
    print(f"ordered_bundle.conditioning.ar_mask shape: {ordered_bundle.conditioning.ar_mask.shape}")
    print(f"ordered_bundle.wave shape: {ordered_bundle.wave.group_positions.shape}")

    # Let's inspect step 0 of plan.decode_fn!
    # In AutoregressiveDecode:
    ar_decode = plan.decode_fn
    print(f"decode_fn type: {type(ar_decode)}")
    print(f"incremental: {ar_decode.incremental}")

    # Let's run ar_decode step 0 directly!
    # Materialize init_sequence:
    init_seq = jnp.full((enc.node_features.shape[1],), -1, dtype=jnp.int32)
    seq_oh = jax.nn.one_hot(init_seq, 21) # (L, 21)
    
    # Run _decode_one_step on enc:
    from aminx.inference.decode._kernel import _decode_one_step
    decoded_step0 = _decode_one_step(
        model=ar_decode.model,
        node_features=enc.node_features[0],
        edge_features=enc.edge_features[0],
        neighbor_indices=enc.neighbor_indices[0],
        mask=enc.mask[0],
        ar_mask=ordered_bundle.conditioning.ar_mask[0],
        sequence_oh=seq_oh,
        key=None,
        inference=True,
    )
    from aminx.inference.decode.autoregressive import _project_logits
    logits_step0 = _project_logits(ar_decode.model, decoded_step0[None, ...])[0]
    lp_step0 = jax.nn.log_softmax(logits_step0, axis=-1)

    p0 = int(order[0])
    print(f"\nStep 0 Position: {p0}")
    print(f"Decoded step 0 logits[:5] at p0: {logits_step0[p0, :5]}")
    print(f"Decoded step 0 logprobs[:5] at p0: {lp_step0[p0, :5]}")

    # Run ref score at p0
    fd2 = dict(fd)
    fd2["S"] = fd["S"]
    fd2["batch_size"] = 1
    fd2["randn"] = ctx["randn_row"]
    with torch.no_grad():
        ref_score = ref_model.score(fd2, use_sequence=True)
        ref_score_lp = ref_score["log_probs"].numpy()[0]
        ref_score_logits = ref_score["logits"].numpy()[0]

    print(f"Ref score logits[:5] at p0: {ref_score_logits[p0, :5]}")
    print(f"Ref score logprobs[:5] at p0: {ref_score_lp[p0, :5]}")
    print(f"Diff logprobs at p0: {np.abs(np.asarray(lp_step0)[p0, :20] - ref_score_lp[p0, :20]).max():.6e}")

    # Now let's check what _fuse_and_sample in AutoregressiveDecode does at step 0!
    from aminx.inference.decode.autoregressive import _fuse_and_sample
    cond = ordered_bundle.conditioning
    wave = ordered_bundle.wave
    pos0 = wave.group_positions[0, :, 0] # (G,)
    group_id = cond.tie_group_map[0, pos0]
    mask_group = (cond.tie_group_map[0][None, :] == group_id[:, None]) # (G, L)

    token, avg_stored = _fuse_and_sample(
        logits_step0[None, ...], # (S=1, L, 21)
        cond.bias,
        mask_group,
        cond.fixed_mask,
        cond.fixed_tokens,
        group_id,
        None,
        plan.stage_set,
        cond.temperature,
        None,
    )
    print(f"avg_stored shape: {avg_stored.shape}")
    print(f"avg_stored[0, :5]: {avg_stored[0, :5]}")
    print(f"Diff avg_stored[0, :20] vs ref_score_lp at p0: {np.abs(np.asarray(avg_stored)[0, :20] - ref_score_lp[p0, :20]).max():.6e}")

    # Now let's run the full plan.decode
    res = plan.decode(enc, ordered_bundle, sample_key, config)
    res_logits = np.asarray(res.logits)
    res_lp = e2e._lsm(res_logits)
    print(f"\nFull plan.decode res.logits[p0, :5]: {res_logits[p0, :5]}")
    print(f"Full plan.decode res_lp[p0, :5]: {res_lp[p0, :5]}")
    print(f"Diff res_lp vs ref_score_lp at p0: {np.abs(res_lp[p0, :20] - ref_score_lp[p0, :20]).max():.6e}")

if __name__ == "__main__":
    main()
