#!/usr/bin/env python3
"""Check coordinate differences between aminx parsed PDB and reference fd['X']."""

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
    X_ref = fd["X"][0].numpy() # (93, 4, 3)

    kw = e2e.aminx_kwargs("lmpnn", ctx, temperature=0.1)
    from aminx.run.specs import SamplingSpecification
    from aminx.host.prep import prep_protein_stream_and_model

    spec = SamplingSpecification(**kw, num_samples=1, random_seed=5)
    protein_iterator, model = prep_protein_stream_and_model(spec)
    batched_ensemble = next(iter(protein_iterator))

    X_aminx = np.asarray(batched_ensemble.coordinates[0, :93, :4]) # (93, 4, 3)

    diff_per_res = np.abs(X_aminx - X_ref).max(axis=(1, 2))
    print(f"Residues with coordinate difference > 1e-4:")
    bad_res = np.where(diff_per_res > 1e-4)[0]
    print(f"Count: {len(bad_res)}, Indices: {bad_res}")
    for r in bad_res:
        print(f"\nResidue {r}: max diff = {diff_per_res[r]:.4f}")
        print(f"Ref X (N, CA, C, O):\n{X_ref[r]}")
        print(f"Aminx X (N, CA, C, O):\n{X_aminx[r]}")

    # Check residue names / numbers in aminx vs ref
    print(f"\nRef chain_labels: {fd['chain_labels'][0].numpy()}")
    print(f"Ref R_idx: {fd['R_idx'][0].numpy()[:10]}")
    print(f"Aminx R_idx: {np.asarray(batched_ensemble.residue_index[0, :10])}")

if __name__ == "__main__":
    main()
