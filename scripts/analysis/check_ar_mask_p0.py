#!/usr/bin/env python3
"""Check ar_mask at p0."""

import os
import sys
from pathlib import Path
import numpy as np

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
    print(f"p0 = {p0}")

    kw = e2e.aminx_kwargs("lmpnn", ctx, temperature=0.1)

    from aminx.run.specs import SamplingSpecification
    from aminx.host.prep import prep_protein_stream_and_model
    from aminx.host.plan import make_inference_plan
    from aminx.inference.bundle_builder import build_inference_bundle, with_decoding_order
    from aminx.host.kernel_dispatch import _structure_bundle_kwargs, _prepare_ligand_context, _prepare_fixed_controls, decoding_order_key
    import jax
    import jax.numpy as jnp

    spec = SamplingSpecification(**kw, num_samples=1, random_seed=5, decoding_order_fn=e2e._order_fn(order, 93))
    protein_iterator, model = prep_protein_stream_and_model(spec)
    batched_ensemble = next(iter(protein_iterator))

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

    base_key = jax.random.PRNGKey(5)
    sample_key = jax.random.fold_in(base_key, 0)
    order_key = decoding_order_key(sample_key)
    ordered_bundle = with_decoding_order(bundle, order_key, e2e._order_fn(order, 93))

    mask_p0 = ordered_bundle.conditioning.ar_mask[0, p0]
    print(f"ar_mask at p0 sum: {mask_p0.sum()}")
    print(f"Non-zero elements in ar_mask at p0: {np.where(mask_p0 > 0)[0]}")

    # What about the order in ordered_bundle?
    wave = ordered_bundle.wave
    print(f"wave wave_idx 0 group_positions: {wave.group_positions[0, 0, 0]}")

    # Now let's check what e2e._order_fn returned!
    fn = e2e._order_fn(order, 93)
    ord_out, _ = fn(order_key, 96)
    print(f"ord_out[:5]: {ord_out[:5]}")
    print(f"ord_out[0]: {ord_out[0]}")

if __name__ == "__main__":
    main()
