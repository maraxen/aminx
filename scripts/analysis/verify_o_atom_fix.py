#!/usr/bin/env python3
"""Verify whether extracting O from index 4 when shape[-2] > 4 fixes LigandMPNN parity."""

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
    import jax
    import jax.numpy as jnp
    from aminx.io.weights import load_model

    spec = SamplingSpecification(**kw, num_samples=1, random_seed=5, decoding_order_fn=e2e._order_fn(order, 93))
    protein_iterator, model = prep_protein_stream_and_model(spec)
    batched_ensemble = next(iter(protein_iterator))

    aminx_model = load_model("ligandmpnn_v_32_020_25")

    # The 37-atom coordinates from prep
    coords_37 = jnp.asarray(batched_ensemble.coordinates[0]) # (96, 37, 3)
    mask = jnp.asarray(batched_ensemble.mask[0]) # (96,)
    r_idx = jnp.asarray(batched_ensemble.residue_index[0]) # (96,)
    chain = jnp.asarray(batched_ensemble.chain_index[0]) # (96,)

    # Extract 4 backbone atoms properly: (N=0, CA=1, C=2, O=4)
    N = coords_37[:, 0, :]
    Ca = coords_37[:, 1, :]
    C = coords_37[:, 2, :]
    O_wrong = coords_37[:, 3, :] # CB!
    O_correct = coords_37[:, 4, :] # Real O!

    coords_4_wrong = jnp.stack([N, Ca, C, O_wrong], axis=1) # (96, 4, 3)
    coords_4_correct = jnp.stack([N, Ca, C, O_correct], axis=1) # (96, 4, 3)

    # Reference features from fd:
    fd_X = fd["X"][0].numpy()
    print("Diff coords_4_correct[:93] vs fd['X']:", np.abs(np.asarray(coords_4_correct[:93]) - fd_X).max())
    print("Diff coords_4_wrong[:93] vs fd['X']:  ", np.abs(np.asarray(coords_4_wrong[:93]) - fd_X).max())

    # Now let's test features with coords_4_correct vs coords_4_wrong:
    from aminx.host.kernel_dispatch import _prepare_ligand_context
    ligand_context = _prepare_ligand_context(spec, batched_ensemble=batched_ensemble, batch_size=1, seq_len=96, canonical_structure_ids=None, batch_structure_ids=None)
    y = ligand_context["Y"][0]
    y_t = ligand_context["Y_t"][0]
    y_m = ligand_context["Y_m"][0]

    # Run features with coords_4_wrong (what aminx currently does):
    V_wrong, E_wrong, E_idx_w, _, _, _ = aminx_model.features(
        _key=jax.random.PRNGKey(0),
        structure_coordinates=coords_4_wrong,
        mask=mask,
        residue_index=r_idx,
        chain_index=chain,
        ligand_coords=y,
        ligand_atom_types=y_t,
        ligand_mask=y_m,
    )

    # Run features with coords_4_correct (the fix):
    V_corr, E_corr, E_idx_c, _, _, _ = aminx_model.features(
        _key=jax.random.PRNGKey(0),
        structure_coordinates=coords_4_correct,
        mask=mask,
        residue_index=r_idx,
        chain_index=chain,
        ligand_coords=y,
        ligand_atom_types=y_t,
        ligand_mask=y_m,
    )

    # Reference features from PyTorch model.features(fd):
    with torch.no_grad():
        V_ref, E_ref, E_idx_ref, _, _, _ = ctx["model"].features(fd)
        V_ref = V_ref[0].numpy()
        E_ref = ctx["model"].W_e(E_ref)[0].numpy() # E after W_e
        E_idx_ref = E_idx_ref[0].numpy()

    print("\n=== FEATURE DIFFS VS PYTORCH REFERENCE ===")
    print("V wrong vs ref:  ", np.abs(np.asarray(V_wrong[:93]) - V_ref).max())
    print("V correct vs ref:", np.abs(np.asarray(V_corr[:93]) - V_ref).max())

    print("E wrong vs ref:  ", np.abs(np.asarray(E_wrong[:93]) - E_ref).max())
    print("E correct vs ref:", np.abs(np.asarray(E_corr[:93]) - E_ref).max())

if __name__ == "__main__":
    main()
