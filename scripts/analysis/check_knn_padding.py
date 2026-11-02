#!/usr/bin/env python3
"""Check KNN neighbors and padding interaction for 1BC8."""

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

    # Reference E_idx
    with torch.no_grad():
        _, _, E_idx_ref, _, _, _ = ctx["model"].features(fd)
    ref_knn_p0 = E_idx_ref[0, p0].numpy()
    aminx_knn_p0 = np.asarray(enc.neighbor_indices[0, p0])

    print(f"Position p0: {p0}")
    print(f"Ref KNN (len 30):   {ref_knn_p0}")
    print(f"Aminx KNN (len 32): {aminx_knn_p0}")
    print(f"Are ref KNN first 30 in aminx KNN? {np.isin(ref_knn_p0, aminx_knn_p0).all()}")

    # Check padding mask:
    print(f"Mask values: {np.asarray(enc.mask[0])}")
    print(f"Are any padded indices (93, 94, 95) in aminx KNN at p0? {np.isin([93, 94, 95], aminx_knn_p0)}")
    
    # Check all 93 positions for padded indices in their KNN:
    has_padded_neighbor = []
    for i in range(93):
        knn = np.asarray(enc.neighbor_indices[0, i])
        if np.isin([93, 94, 95], knn).any():
            has_padded_neighbor.append(i)
    print(f"Number of positions with padded neighbors in their KNN: {len(has_padded_neighbor)}")
    print(f"Positions with padded neighbors: {has_padded_neighbor[:10]}")

    # Check the coordinates of padded residues:
    print(f"Coords at 92 (real): {np.asarray(batched_ensemble.coordinates[0, 92, 0])}")
    print(f"Coords at 93 (padded): {np.asarray(batched_ensemble.coordinates[0, 93, 0])}")

    # NOW: In LigandMPNN upstream reference: HOW MANY NEIGHBORS (top_k)?
    print(f"Upstream top_k: {E_idx_ref.shape[-1]}")
    print(f"Aminx num_neighbors: {enc.neighbor_indices.shape[-1]}")

if __name__ == "__main__":
    main()
