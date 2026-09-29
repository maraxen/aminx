"""Canonical PyTree bundles for Aminx inference.

These bundles form the strict boundary between host-side preparation and
accelerator-side JIT kernels. All Optional fields are resolved to concrete
zero-filled arrays by the host before entering JIT.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class GeometryBundle(eqx.Module):
  """Stacked backbone geometry for one or more conformational states.

  Backbone coordinates, masks, and residue/chain indices assembled from
  protein structures. Forms the JIT boundary: host-side prepares geometry,
  passes as traced PyTree into JAX kernels. Static fields (n_states, n_canonical,
  n_flat) are not traced; Optional fields are resolved to zero-filled arrays
  before JIT entry.

  Parameters
  ----------
  coords : Float[Array, "S L 4 3"]
      Backbone atom coordinates in canonical order [N, CA, C, O] per residue.
      S = num conformational states, L = sequence length.
  mask : Float[Array, "S L"]
      Per-residue backbone validity mask (1.0 = valid, 0.0 = masked).
  residue_index : Int[Array, "S L"]
      Absolute residue indices in the PDB structure.
  chain_index : Int[Array, "S L"]
      Chain assignment for inter-chain masking.
  n_states : int
      Number of conformational states. Static (not a JAX array).
  n_canonical : int
      Number of canonical amino acids (19 or 20). Static (not a JAX array).
  n_flat : int
      Flat supersystem size = sum of all residues across all states.
      Static (not a JAX array).
  structure_mapping : Int[Array, "S L"] | None
      Optional structure-specific residue mapping for multi-model systems.
      None for standard single-model inputs.
  physics_features : Float[Array, "S L P"] | None
      Optional membrane physics features (one-hot per-residue labels).
      P = physics feature dimension (depends on physics model).
      None for soluble/ligand-only models (no membrane conditioning).

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  coords: Float[Array, "S L 4 3"]
  mask: Float[Array, "S L"]
  residue_index: Int[Array, "S L"]
  chain_index: Int[Array, "S L"]
  n_states: int = eqx.field(static=True)
  n_canonical: int = eqx.field(static=True)
  n_flat: int = eqx.field(static=True)
  structure_mapping: Int[Array, "S L"] | None = None
  # Membrane physics features: one-hot of per-residue labels, shape (S, L, P).
  # None for soluble/ligand models (no physics conditioning).
  physics_features: Float[Array, "S L P"] | None = None
  # Side-chain atom coordinates (full 37-atom set): shape (S, L, 37, 3).
  # None when side-chain context conditioning is disabled (no GPU transfer).
  atom_37: Float[Array, "S L 37 3"] | None = None
  # Side-chain atom validity mask: shape (S, L, 37).
  # None when side-chain context conditioning is disabled (no GPU transfer).
  atom_37_mask: Float[Array, "S L 37"] | None = None
  # Per-residue designability mask (1.0 = designable, 0.0 = fixed).
  # Derived from fixed_mask complement; None means all residues designable.
  chain_mask: Float[Array, "S L"] | None = None


class ConditioningBundle(eqx.Module):
  """Sequence-level conditioning input — fully resolved, no Optional fields.

  All fields are pre-populated by the host: unconditional/autoregressive modes
  pass zeros for sequence_oh, conditional modes pass one-hot, masked regions
  pass fixed_tokens, etc. JAX kernels never see None; zero-filled arrays are
  passed instead for missing modalities.

  Parameters
  ----------
  fixed_mask : Float[Array, "L"]
      Per-position fixation mask (1.0 = fixed, 0.0 = designable).
      Shape: L = sequence length.
  fixed_tokens : Int[Array, "L"]
      Amino acid index for fixed residues (0 = not fixed, actual index = fixed).
  bias : Float[Array, "L V"]
      Per-position per-amino-acid logit bias. Shape: V = vocab size (21).
      Applied during decode_step before sampling or scoring.
  tie_group_map : Int[Array, "S L"]
      Tied-position group assignment. Positions with same group id
      receive same logit distribution (for symmetric design patterns).
      Shape: S = num states.
  state_position_map : Int[Array, "S L"]
      Per-state residue correspondence into the shared reference frame (state 0's
      own numbering). state_position_map[s, i] is the index in state s's native
      numbering corresponding to reference position i, or -1 if state s has no
      residue at that reference position (indel). Default is the identity map
      (state_position_map[s, i] = i for all s), reproducing naive same-index
      cross-state fusion. Populate via aminx.utils.align.build_state_position_map
      when states have genuinely different native lengths/numbering (e.g.
      different PDB depositions of the same protein) so per-position multistate
      fusion (_apply_logit_transform, ar_logit_transform) combines logits for the
      same physical residue rather than the same raw array index.
  state_weights : Float[Array, "S"]
      Per-state contribution weights for logit fusion in multi-state models.
      Traced leaf; summed to 1.0 across states before fusion.
  sequence_oh : Float[Array, "L V"]
      One-hot encoded input sequence for conditional decoding.
      All zeros for unconditional or autoregressive-only modes.
  ar_mask : Float[Array, "S L L"]
      Autoregressive attention mask. 1.0 = visible, 0.0 = masked.
      All-ones for purely conditional (non-AR) scoring.
  temperature : Float[Array, ""]
      Sampling temperature (scalar). Default 1.0 = no temperature scaling.
      Traced leaf; used by sample_step if present.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  fixed_mask: Float[Array, "L"]
  fixed_tokens: Int[Array, "L"]
  bias: Float[Array, "L V"]
  tie_group_map: Int[Array, "S L"]
  state_position_map: Int[Array, "S L"]
  state_weights: Float[Array, "S"]
  sequence_oh: Float[Array, "L V"]  # zeros for unconditional/AR
  ar_mask: Float[Array, "S L L"]  # full 1s for purely conditional
  temperature: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.array(1.0))


class LigandBundle(eqx.Module):
  """Ligand atomic context for structure-conditioned sequence design.

  Per-state ligand coordinates, atom types, and validity masks. All arrays
  are zero-filled when no ligand is present; never None inside JIT.

  Parameters
  ----------
  ligand_coords : Float[Array, "S L_lig A 3"]
      Ligand atom 3D coordinates gathered per protein residue.
      L_lig = protein sequence length (each protein residue has A nearest
      ligand atoms); A = num nearest ligand atoms per protein residue.
  ligand_atom_types : Int[Array, "S L_lig A"]
      Ligand atom element type indices (integer) per protein residue.
  ligand_mask : Float[Array, "S L_lig A"]
      Validity mask for ligand atoms (1.0 = valid ligand atom, 0.0 = padding).

  References
  ----------
  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  .. [LigandMPNN-code] Dauparas, J. LigandMPNN source code (commit 3870631).
     https://github.com/dauparas/LigandMPNN

  """

  ligand_coords: Float[Array, "S L_lig A 3"]
  ligand_atom_types: Int[Array, "S L_lig A"]
  ligand_mask: Float[Array, "S L_lig A"]


class WaveScheduleBundle(eqx.Module):
  """Tied-position decoding schedule encoded as waves and groups.

  Organizes the tied-position autoregressive decode into W waves; each
  wave contains G groups of P positions. Positions with the same tie_group_id
  are decoded together (to maintain equivalent logit distributions).

  Parameters
  ----------
  group_ids : Int[Array, "W G"]
      Tie group identifier per wave and group slot.
      Shape: W = num decoding waves, G = max groups per wave.
  group_positions : Int[Array, "W G P"]
      Position indices within each group.
      Shape: P = max positions per group. Padded with -1 (masked by position_valid).
  group_valid : Bool[Array, "W G"]
      Validity mask for groups (True = group has content).
  position_valid : Bool[Array, "W G P"]
      Validity mask for positions within groups (True = position is real).

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  group_ids: Int[Array, "W G"]
  group_positions: Int[Array, "W G P"]
  group_valid: Bool[Array, "W G"]
  position_valid: Bool[Array, "W G P"]

  @staticmethod
  def from_tie_groups(
    tie_group_map: Int[Array, "L"],
    decoding_order: Int[Array, "L"],
  ) -> WaveScheduleBundle:
    """Create a schedule where tied positions are in the same wave step.

    Groups positions by tie_group_map and orders them according to
    decoding_order. Each tied position group becomes one wave.

    Parameters
    ----------
    tie_group_map : Int[Array, "L"]
        Tie group assignment per position. Same id = same group.
    decoding_order : Int[Array, "L"]
        Positions in decoding order. Determines wave sequencing.

    Returns
    -------
    WaveScheduleBundle
        Schedule with groups organized into waves.

    """
    # Map each position to its decoding step
    # (Assuming decoding_order respects ties: positions in same tie group
    # must appear consecutively or be handled as a block)
    # For now, let's group by tie_group_map.

    # Unique tie groups in order of first appearance in decoding_order
    present_groups = []
    seen_groups = set()
    for i in decoding_order.tolist():
      g = int(tie_group_map[i])
      if g not in seen_groups:
        present_groups.append(g)
        seen_groups.add(g)

    num_waves = len(present_groups)
    # Maximum positions in a group
    counts = jnp.bincount(tie_group_map)
    max_positions = int(jnp.max(counts))

    group_ids = jnp.array(present_groups)[:, None]  # (num_waves, 1)

    # group_positions: (num_waves, 1, max_positions)
    # This is tricky to do in JAX without loops if we want it general.
    # But since this is host-side factory, we can use loops.
    pos_list = []
    for g in present_groups:
      indices = jnp.where(tie_group_map == g)[0]
      # Pad to max_positions
      padded = jnp.pad(indices, (0, max_positions - len(indices)), constant_values=-1)
      pos_list.append(padded)

    group_positions = jnp.array(pos_list)[:, None, :]
    group_valid = jnp.ones((num_waves, 1), dtype=jnp.bool_)
    position_valid = group_positions != -1

    # Replace -1 with 0 to avoid index errors (masked by position_valid)
    group_positions = jnp.where(position_valid, group_positions, 0)

    return WaveScheduleBundle(
      group_ids=group_ids,
      group_positions=group_positions,
      group_valid=group_valid,
      position_valid=position_valid,
    )

  @staticmethod
  def from_colors(
    group_colors: Int[Array, " n_groups"],
    tie_group_map: Int[Array, "L"],
  ) -> WaveScheduleBundle:
    """Create a schedule from a graph coloring of tie groups (chromatic / improper-coloring arms).

    Unlike `from_tie_groups` (exactly one tie group per wave, in a fixed linear
    order), this groups tie groups that share a color into the *same* wave —
    i.e. each color class becomes one wave containing all of its (conditionally
    independent, for a proper coloring) groups. `group_colors[i]` is the color of
    the i-th unique tie group in ascending group-id order.

    Note: the multi-group waves (G > 1) this constructor produces ARE decoded
    correctly — `AutoregressiveDecode.__call__` samples every active group slot of
    a wave from one shared forward pass. This docstring previously claimed only
    slot 0 was decoded; that was true before 85d8c480 (2026-06-30) and stale after
    it. See the `aminx/inference/schedule_selector.py` module docstring for the
    constraint that does still apply (host-side construction only).

    Parameters
    ----------
    group_colors : Int[Array, "n_groups"]
        Color assigned to each unique tie group (ascending tie-group-id order).
    tie_group_map : Int[Array, "L"]
        Tie group assignment per position. Same id = same group.

    Returns
    -------
    WaveScheduleBundle
        Schedule with groups packed by color into waves (G may be > 1).

    """
    present_groups = sorted({int(g) for g in tie_group_map.tolist()})
    if len(present_groups) != len(group_colors):
      msg = (
        f"group_colors has {len(group_colors)} entries but tie_group_map has "
        f"{len(present_groups)} unique groups."
      )
      raise ValueError(msg)
    group_colors_list = [int(c) for c in group_colors.tolist()]

    num_waves = max(group_colors_list) + 1
    groups_per_wave: list[list[int]] = [[] for _ in range(num_waves)]
    for group_id, color in zip(present_groups, group_colors_list, strict=True):
      groups_per_wave[color].append(group_id)

    max_groups_per_wave = max(len(g) for g in groups_per_wave)
    counts = jnp.bincount(tie_group_map.reshape(-1))
    max_positions = int(jnp.max(counts))

    group_ids_list = [[-1] * max_groups_per_wave for _ in range(num_waves)]
    group_positions_list = [
      [[0] * max_positions for _ in range(max_groups_per_wave)] for _ in range(num_waves)
    ]
    group_valid_list = [[False] * max_groups_per_wave for _ in range(num_waves)]
    position_valid_list = [
      [[False] * max_positions for _ in range(max_groups_per_wave)] for _ in range(num_waves)
    ]

    for w, groups in enumerate(groups_per_wave):
      for slot, g in enumerate(groups):
        indices = jnp.where(tie_group_map == g)[0]
        n = int(indices.shape[0])
        group_ids_list[w][slot] = g
        group_valid_list[w][slot] = True
        for p in range(n):
          group_positions_list[w][slot][p] = int(indices[p])
          position_valid_list[w][slot][p] = True

    return WaveScheduleBundle(
      group_ids=jnp.array(group_ids_list, dtype=jnp.int32),
      group_positions=jnp.array(group_positions_list, dtype=jnp.int32),
      group_valid=jnp.array(group_valid_list, dtype=jnp.bool_),
      position_valid=jnp.array(position_valid_list, dtype=jnp.bool_),
    )

  @staticmethod
  def from_decoding_order(
    decoding_order: Int[Array, "L"],
    tie_group_map: Int[Array, "L"] | None = None,
  ) -> WaveScheduleBundle:
    """Sequential schedule that visits positions in `decoding_order` (jit/vmap-safe).

    This is `empty` with its waves permuted: W = L waves, one slot each, wave `t`
    holding position `decoding_order[t]`. Unlike `from_tie_groups` it has no
    data-dependent shapes or `.tolist()` calls, so it can be built inside a trace from a
    traced (e.g. freshly drawn) order.

    Tie groups need no special handling: `AutoregressiveDecode` resolves each slot to its
    real group via the conditioning `tie_group_map` and samples a group only at its first
    occurrence, and `generate_wave_ar_mask` places each group at its first wave. A tied
    group is therefore decoded at the step its earliest member appears in the order --
    the same semantics as `from_tie_groups`, with the later duplicate waves inert.

    Parameters
    ----------
    decoding_order : Int[Array, "L"]
        ORDER array: `decoding_order[t]` is the position decoded at step `t`. Not a rank
        array -- see `aminx.utils.autoregression.generate_ar_mask` for the distinction.
    tie_group_map : Int[Array, "L"] | None
        Tie group id per position (state-0 convention). `None` means untied (each
        position its own group).

    Returns
    -------
    WaveScheduleBundle
        Schedule of shape W = L, G = 1, P = 1.

    """
    order = jnp.asarray(decoding_order, dtype=jnp.int32)
    seq_len = order.shape[0]
    group_of_step = (
      order if tie_group_map is None else jnp.asarray(tie_group_map, dtype=jnp.int32)[order]
    )
    return WaveScheduleBundle(
      group_ids=group_of_step[:, None],
      group_positions=order[:, None, None],
      group_valid=jnp.ones((seq_len, 1), dtype=jnp.bool_),
      position_valid=jnp.ones((seq_len, 1, 1), dtype=jnp.bool_),
    )

  @staticmethod
  def empty(seq_len: int) -> WaveScheduleBundle:
    """Sequential single-position-at-a-time schedule (no tied positions).

    Parameters
    ----------
    seq_len : int
        Sequence length. Creates W=L waves, one position per wave.

    Returns
    -------
    WaveScheduleBundle
        Schedule with sequential untied decoding.

    """
    # W = L, G = 1
    group_ids = jnp.arange(seq_len)[:, None]
    group_positions = jnp.arange(seq_len)[:, None, None]
    group_valid = jnp.ones((seq_len, 1), dtype=jnp.bool_)
    position_valid = jnp.ones((seq_len, 1, 1), dtype=jnp.bool_)
    return WaveScheduleBundle(
      group_ids=group_ids,
      group_positions=group_positions,
      group_valid=group_valid,
      position_valid=position_valid,
    )


class InferenceBundle(eqx.Module):
  """Top-level container passed as single traced PyTree through JIT boundary.

  Combines geometry, conditioning, ligand context, and wave schedule into
  one immutable PyTree for inference kernels. All sub-bundles are fully
  resolved on the host (no None fields); sub-bundles are traced leaves.

  Parameters
  ----------
  geometry : GeometryBundle
      Backbone geometry (coordinates, masks, indices) for one or more states.
  conditioning : ConditioningBundle
      Sequence conditioning: fixed positions, biases, tied groups, one-hot.
  ligand : LigandBundle
      Ligand atoms and masks. Zero-filled if no ligand.
  wave : WaveScheduleBundle
      Decoding wave schedule for tied-position autoregressive decode.
  packer : PackerBundle | None
      Optional side-chain packer bundle (sequence, coordinates, ligand, masks).
      None if side-chain packing features are not needed.
  backbone_noise : Float[Array, ""]
      Backbone coordinate noise magnitude (scalar). 0.0 = no noise.
      Used by coordinate augmentation during encode/decode.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  """

  geometry: GeometryBundle
  conditioning: ConditioningBundle
  ligand: LigandBundle
  wave: WaveScheduleBundle
  packer: PackerBundle | None = None
  backbone_noise: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.array(0.0))


class EncodedFeatures(eqx.Module):
  """Single-state encoder output reused across encode/decode boundaries.

  Node and edge features extracted by the encoder from a single backbone
  structure. Typically batched via vmap when processing multiple states.

  Parameters
  ----------
  node_features : Float[Array, "L D"]
      Per-residue node embeddings. Shape: L = sequence length, D = embedding dim.
  edge_features : Float[Array, "L K D"]
      Per-residue neighbor edge embeddings.
      Shape: K = num neighbors per residue (typically 30).
  neighbor_indices : Int[Array, "L K"]
      Neighbor residue indices (used for gather operations in decode).

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  node_features: Float[Array, "L D"]
  edge_features: Float[Array, "L K D"]
  neighbor_indices: Int[Array, "L K"]


class EncoderOutput(eqx.Module):
  """Multi-state encoder output before logit fusion.

  Stacked single-state encoder outputs (via vmap) for multiple conformational
  states. Passed into logit fusion and decode functions.

  Parameters
  ----------
  node_features : Float[Array, "S L D"]
      Per-state per-residue node embeddings.
      Shape: S = num states, L = sequence length, D = embedding dim.
  edge_features : Float[Array, "S L K D"]
      Per-state per-residue neighbor edge embeddings.
  neighbor_indices : Int[Array, "S L K"]
      Neighbor residue indices (constant across states).
  mask : Float[Array, "S L"]
      Per-state per-residue validity mask.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  node_features: Float[Array, "S L D"]
  edge_features: Float[Array, "S L K D"]
  neighbor_indices: Int[Array, "S L K"]
  mask: Float[Array, "S L"]


class PackerResult(eqx.Module):
  """Von Mises–Fisher mixture parameters for side-chain torsion angles.

  Output of the side-chain packer forward pass. Encodes the predicted
  distribution over chi angles (4 per residue) as a VMF mixture model.

  Parameters
  ----------
  mean : Float[Array, "L 4 3"] | Float[Array, "S L 4 3"]
      Mean direction vectors (unit sphere). Shape: L = num residues,
      S = num conformational states, 4 = chi angles (chi1-chi4),
      3 = Cartesian coordinates.
  concentration : Float[Array, "L 4 3"] | Float[Array, "S L 4 3"]
      Concentration (kappa) parameters for VMF. Higher = sharper distribution.
  mix_logits : Float[Array, "L 4 3"] | Float[Array, "S L 4 3"]
      Logits for mixture weights across the 3-component VMF mixture.

  References
  ----------
  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  .. [LigandMPNN-code] Dauparas, J. LigandMPNN source code (commit 3870631).
     https://github.com/dauparas/LigandMPNN

  """

  mean: Float[Array, "L 4 3"] | Float[Array, "S L 4 3"]
  concentration: Float[Array, "L 4 3"] | Float[Array, "S L 4 3"]
  mix_logits: Float[Array, "L 4 3"] | Float[Array, "S L 4 3"]


class PackerBundle(eqx.Module):
  """Input features for side-chain packing (torsion angle prediction).

  Amino acid sequence, backbone coordinates, and ligand context for the
  side-chain packing forward pass. Output is a PackerResult (VMF parameters).

  Parameters
  ----------
  sequence : Int[Array, "L"] | Int[Array, "S L"]
      Amino acid sequence (indices 0–20). Shape: L = sequence length,
      S = num conformational states.
  backbone_coords : Float[Array, "L 14 3"] | Float[Array, "S L 14 3"]
      All backbone atoms (N, CA, C, O, CB, etc.). Shape: 14 = max atom types.
  backbone_mask : Float[Array, "L 14"] | Float[Array, "S L 14"]
      Validity mask for backbone atoms.
  ligand_coords : Float[Array, "L M 3"] | Float[Array, "S L M 3"]
      Ligand atom coordinates per residue. Shape: M = max ligand atoms.
  ligand_mask : Float[Array, "L M"] | Float[Array, "S L M"]
      Validity mask for ligand atoms.
  ligand_atom_types : Float[Array, "L M"] | Float[Array, "S L M"]
      Ligand atom element type indices per protein residue. Note: annotated as
      Float due to upstream type drift; semantically integer atom type indices.
  mask : Float[Array, "L"] | Float[Array, "S L"]
      Per-residue validity mask (1.0 = design-able, 0.0 = masked).
  residue_index : Int[Array, "L"] | Int[Array, "S L"]
      Absolute residue indices in PDB.
  chain_labels : Int[Array, "L"] | Int[Array, "S L"]
      Chain assignment for inter-chain masking.
  backbone_noise : Float[Array, ""]
      Backbone coordinate noise magnitude. Default 0.0 (no noise).

  References
  ----------
  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  .. [LigandMPNN-code] Dauparas, J. LigandMPNN source code (commit 3870631).
     https://github.com/dauparas/LigandMPNN

  """

  sequence: Int[Array, "L"] | Int[Array, "S L"]
  backbone_coords: Float[Array, "L 14 3"] | Float[Array, "S L 14 3"]
  backbone_mask: Float[Array, "L 14"] | Float[Array, "S L 14"]
  ligand_coords: Float[Array, "L M 3"] | Float[Array, "S L M 3"]
  ligand_mask: Float[Array, "L M"] | Float[Array, "S L M"]
  ligand_atom_types: Float[Array, "L M"] | Float[Array, "S L M"]
  mask: Float[Array, "L"] | Float[Array, "S L"]
  residue_index: Int[Array, "L"] | Int[Array, "S L"]
  chain_labels: Int[Array, "L"] | Int[Array, "S L"]
  backbone_noise: Float[Array, ""] | float = 0.0


class DecodeOutput(eqx.Module):
  """Stacked decoder outputs (sequences and logits) across K outputs.

  Output of decode-step fusion containing sampled or scored sequences and
  their corresponding logits. Used as input to post-decode fusion stages.

  Parameters
  ----------
  sequences : Int[Array, "K L"]
      Sampled or predicted amino acid sequences.
      Shape: K = number of decoded outputs, L = sequence length.
  logits : Float[Array, "K L V"]
      Per-position per-amino-acid logit scores.
      Shape: V = vocabulary size (21).

  """

  sequences: Int[Array, "K L"]
  logits: Float[Array, "K L V"]
