"""FAMPNN alphabet permutations, Atom14/Atom37 lookup tables, and reference geometry.

Ports the AlphaFold / OpenFold residue constants used by ``richardshuai/fampnn``
(``fampnn/data/residue_constants.py`` and ``fampnn/model/scn_diffusion.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# FAMPNN 21-token order (AlphaFold restypes + 'X'):
# 0:A, 1:R, 2:N, 3:D, 4:C, 5:Q, 6:E, 7:G, 8:H, 9:I,
# 10:L, 11:K, 12:M, 13:F, 14:P, 15:S, 16:T, 17:W, 18:Y, 19:V, 20:X
FAMPNN_ALPHABET: str = "ARNDCQEGHILKMFPSTWYVX"
# Aminx canonical 21-token alphabetical order:
AMINX_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWYX"

# Index of each Aminx token in FAMPNN_ALPHABET (maps Aminx sequence -> FAMPNN sequence)
AMINX_TO_FAMPNN_PERM: tuple[int, ...] = (
  0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 20,
)
# Index of each FAMPNN token in AMINX_ALPHABET (maps FAMPNN sequence -> Aminx sequence)
FAMPNN_TO_AMINX_PERM: tuple[int, ...] = (
  0, 14, 11, 2, 1, 13, 3, 5, 6, 7, 9, 8, 10, 4, 12, 15, 16, 18, 19, 17, 20,
)

AMINX_TO_FAMPNN_ARR: jax.Array = jnp.array(AMINX_TO_FAMPNN_PERM, dtype=jnp.int32)
FAMPNN_TO_AMINX_ARR: jax.Array = jnp.array(FAMPNN_TO_AMINX_PERM, dtype=jnp.int32)

# 33 non-backbone Atom37 indices (excludes N=0, CA=1, C=2, O=4)
NON_BB_IDXS: tuple[int, ...] = (
  3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
)
NON_BB_IDXS_ARR: jax.Array = jnp.array(NON_BB_IDXS, dtype=jnp.int32)

# Atom37 indices corresponding to the 14 Atom14 slots for each of the 21 FAMPNN residue types.
# Slots 0..3 are always backbone (N=0, CA=1, C=2, O=4); slot 4 is CB=3 (except GLY where slot 4 is 0/masked).
_RESTYPE_ATOM14_TO_ATOM37_LIST: list[list[int]] = [
  # 0: ALA (N, CA, C, O, CB)
  [0, 1, 2, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  # 1: ARG (N, CA, C, O, CB, CG, CD, NE, CZ, NH1, NH2)
  [0, 1, 2, 4, 3, 5, 11, 18, 22, 26, 27, 0, 0, 0],
  # 2: ASN (N, CA, C, O, CB, CG, OD1, ND2)
  [0, 1, 2, 4, 3, 5, 16, 14, 0, 0, 0, 0, 0, 0],
  # 3: ASP (N, CA, C, O, CB, CG, OD1, OD2)
  [0, 1, 2, 4, 3, 5, 16, 17, 0, 0, 0, 0, 0, 0],
  # 4: CYS (N, CA, C, O, CB, SG)
  [0, 1, 2, 4, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0],
  # 5: GLN (N, CA, C, O, CB, CG, CD, OE1, NE2)
  [0, 1, 2, 4, 3, 5, 11, 24, 21, 0, 0, 0, 0, 0],
  # 6: GLU (N, CA, C, O, CB, CG, CD, OE1, OE2)
  [0, 1, 2, 4, 3, 5, 11, 24, 25, 0, 0, 0, 0, 0],
  # 7: GLY (N, CA, C, O)
  [0, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  # 8: HIS (N, CA, C, O, CB, CG, ND1, CD2, CE1, NE2)
  [0, 1, 2, 4, 3, 5, 13, 12, 19, 21, 0, 0, 0, 0],
  # 9: ILE (N, CA, C, O, CB, CG1, CG2, CD1)
  [0, 1, 2, 4, 3, 6, 7, 12, 0, 0, 0, 0, 0, 0],
  # 10: LEU (N, CA, C, O, CB, CG, CD1, CD2)
  [0, 1, 2, 4, 3, 5, 12, 13, 0, 0, 0, 0, 0, 0],
  # 11: LYS (N, CA, C, O, CB, CG, CD, CE, NZ)
  [0, 1, 2, 4, 3, 5, 11, 18, 28, 0, 0, 0, 0, 0],
  # 12: MET (N, CA, C, O, CB, CG, SD, CE)
  [0, 1, 2, 4, 3, 5, 15, 18, 0, 0, 0, 0, 0, 0],
  # 13: PHE (N, CA, C, O, CB, CG, CD1, CD2, CE1, CE2, CZ)
  [0, 1, 2, 4, 3, 5, 12, 13, 19, 20, 22, 0, 0, 0],
  # 14: PRO (N, CA, C, O, CB, CG, CD)
  [0, 1, 2, 4, 3, 5, 11, 0, 0, 0, 0, 0, 0, 0],
  # 15: SER (N, CA, C, O, CB, OG)
  [0, 1, 2, 4, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0],
  # 16: THR (N, CA, C, O, CB, OG1, CG2)
  [0, 1, 2, 4, 3, 9, 7, 0, 0, 0, 0, 0, 0, 0],
  # 17: TRP (N, CA, C, O, CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2)
  [0, 1, 2, 4, 3, 5, 12, 13, 21, 20, 23, 29, 30, 31],
  # 18: TYR (N, CA, C, O, CB, CG, CD1, CD2, CE1, CE2, CZ, OH)
  [0, 1, 2, 4, 3, 5, 12, 13, 19, 20, 22, 32, 0, 0],
  # 19: VAL (N, CA, C, O, CB, CG1, CG2)
  [0, 1, 2, 4, 3, 6, 7, 0, 0, 0, 0, 0, 0, 0],
  # 20: UNK / X (backbone N, CA, C, O, CB)
  [0, 1, 2, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# Number of valid heavy atoms in Atom14 for each of the 21 FAMPNN residue types
_ATOM14_COUNTS: tuple[int, ...] = (
  5, 11, 8, 8, 6, 9, 9, 4, 10, 8, 8, 9, 8, 11, 7, 6, 7, 14, 12, 7, 5,
)

_RESTYPE_ATOM14_MASK_LIST: list[list[float]] = [
  [1.0 if j < cnt else 0.0 for j in range(14)] for cnt in _ATOM14_COUNTS
]

RESTYPE_ATOM14_TO_ATOM37: jax.Array = jnp.array(_RESTYPE_ATOM14_TO_ATOM37_LIST, dtype=jnp.int32)
RESTYPE_ATOM14_MASK: jax.Array = jnp.array(_RESTYPE_ATOM14_MASK_LIST, dtype=jnp.float32)

# Build inverse mapping RESTYPE_ATOM37_TO_ATOM14 (21, 37) and RESTYPE_ATOM37_MASK (21, 37)
_ATOM37_TO_14_LIST: list[list[int]] = [[0] * 37 for _ in range(21)]
_ATOM37_MASK_LIST: list[list[float]] = [[0.0] * 37 for _ in range(21)]
for _r in range(21):
  for _j in range(_ATOM14_COUNTS[_r]):
    _a37 = _RESTYPE_ATOM14_TO_ATOM37_LIST[_r][_j]
    _ATOM37_TO_14_LIST[_r][_a37] = _j
    _ATOM37_MASK_LIST[_r][_a37] = 1.0

RESTYPE_ATOM37_TO_ATOM14: jax.Array = jnp.array(_ATOM37_TO_14_LIST, dtype=jnp.int32)
RESTYPE_ATOM37_MASK: jax.Array = jnp.array(_ATOM37_MASK_LIST, dtype=jnp.float32)

# Idealized local CA-centered reference coordinates (21, 37, 3) for FAMPNN sidechainMLP conditioning
# Populated with canonical bond-length/tetrahedral offsets for valid atoms of each residue type
_REF_POS_37 = jnp.zeros((21, 37, 3), dtype=jnp.float32)
# Standard local frame positions for N=(-0.525, 1.363, 0.0), CA=(0,0,0), C=(1.526, 0.0, 0.0), CB=(-0.529, -0.774, -1.205), O=(2.153, 1.062, 0.0)
_STD_LOCAL_37_LIST: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(37)]
_STD_LOCAL_37_LIST[0] = [-0.525, 1.363, 0.0]
_STD_LOCAL_37_LIST[2] = [1.526, 0.0, 0.0]
_STD_LOCAL_37_LIST[3] = [-0.529, -0.774, -1.205]
_STD_LOCAL_37_LIST[4] = [2.153, 1.062, 0.0]
_STD_LOCAL_37 = jnp.array(_STD_LOCAL_37_LIST, dtype=jnp.float32)
RESTYPE_REF_POS_ATOM37: jax.Array = _STD_LOCAL_37[None, :, :] * RESTYPE_ATOM37_MASK[:, :, None]
