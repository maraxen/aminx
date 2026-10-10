"""P4d: dump upstream ProtonPottsMPNN (v6) features for the two cells that discriminate the backbone set.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §40. Run in the
`aminx-oracles-protonpotts` environment::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/protonpotts/dump_protonpotts_features_p4d.py --out <dir> \\
        --cell only_o_1hgu=<1HGU.pdb> --cell only_o_1olr=<1OLR.pdb>

WHY. The `protonpotts_features` wave perturbs each featurizer rule and requires the comparison to reject
it. The perturbation "O is not a backbone atom" was NOT rejected by any of the nine sealed cells: at the
0.8 backbone threshold, every residue in 1CQW whose O fails also has a failing N, CA or C, so the cells
cannot tell whether O counts. Upstream's source says it does
(`chain_type_to_atom_names={PROTEINS: ["N", "CA", "C", "O"]}`), but a rule read from source and never
discriminated by data is exactly what the wave exists to expose.

A scan of the 137 available structures at the correct 0.8 threshold found exactly the residues needed:
1HGU and 1OLR each have one residue whose N, CA and C are above 0.8 and whose O is not.

This is the P4c dumper unchanged; only the cells differ. Logic lives in `dump_protonpotts_features_p4c.py`.
"""

from __future__ import annotations

import sys

from dump_protonpotts_features_p4c import main

if __name__ == "__main__":
  sys.exit(main())
