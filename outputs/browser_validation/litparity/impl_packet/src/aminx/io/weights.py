"""Unified loader for Aminx weights from Hugging Face Hub or local path."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import equinox as eqx
import jax
import jax.nn.initializers as init
import zstandard as zstd
from huggingface_hub import hf_hub_download

from aminx.model import Aminx, PrxteinLigandMPNN
from aminx.model.packer import Packer

HF_REPO_ID = "maraxen/aminx"

# Pinned so a code pin implies a WEIGHTS pin. The built wheel ships no
# ``model_params/*.eqx.zst``, so without this the checkpoint bytes are whatever the Hub's
# default branch happens to serve, and two machines can run different weights while reporting
# an identical aminx version. Bump deliberately, together with the consumer-side manifest.
# Override for a checkpoint newer than the pin via ``AMINX_WEIGHTS_REVISION``.
HF_REVISION = "25fb7f6e985724dee7471c3bc18522fe33b9228e"

#: Env var naming a directory that is AUTHORITATIVE for weights when set. A missing file there
#: raises instead of silently falling through to the Hub -- the silent fallback is the defect
#: shape this exists to prevent.
WEIGHTS_DIR_ENV = "AMINX_WEIGHTS_DIR"
REVISION_ENV = "AMINX_WEIGHTS_REVISION"

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

log = logging.getLogger(__name__)

NODE_FEATURES = 128
EDGE_FEATURES = 128
HIDDEN_FEATURES = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
VOCAB_SIZE = 21
LIGAND_DEFAULT_CHECKPOINT = "ligandmpnn_v_32_020_25"
# Dauparas reference LigandMPNN uses two alternating ligand-side / protein-side context blocks.
NUM_LIGAND_CONTEXT_LAYERS = 2
# Rebuild ``model_params/*.eqx.zst`` from ``.pt``: ``scripts/regenerate_packaged_eqx_zst.py``.

LEGACY_ALIAS_MAP = {
  ("original", "v_48_002"): "proteinmpnn_v_48_002.eqx.zst",
  ("original", "v_48_010"): "proteinmpnn_v_48_010.eqx.zst",
  ("original", "v_48_020"): "proteinmpnn_v_48_020.eqx.zst",
  ("original", "v_48_030"): "proteinmpnn_v_48_030.eqx.zst",
  ("soluble", "v_48_002"): "solublempnn_v_48_002.eqx.zst",
  ("soluble", "v_48_010"): "solublempnn_v_48_010.eqx.zst",
  ("soluble", "v_48_020"): "solublempnn_v_48_020.eqx.zst",
  ("soluble", "v_48_030"): "solublempnn_v_48_030.eqx.zst",
}


def get_topology_for_checkpoint(checkpoint_id: str) -> dict[str, int | bool | str]:
  """Infer model topology and type from the checkpoint filename/id."""
  topology = {
    "k_neighbors": 48,
    "num_positional_embeddings": 32,
    "atom_context_num": 16,
    "physics_feature_dim": 0,
    "model_type": "protein",
  }

  name = checkpoint_id.lower()

  if "v_32" in name:
    topology["k_neighbors"] = 32
    topology["num_positional_embeddings"] = 16
  elif "v_30" in name:
    topology["k_neighbors"] = 30
    topology["num_positional_embeddings"] = 16
  elif "v_48" in name:
    topology["k_neighbors"] = 48
    topology["num_positional_embeddings"] = 32

  if "ligandmpnn_sc" in name or "_sc_" in name:
    topology["model_type"] = "packer"
    topology["num_positional_embeddings"] = 16
  elif "ligandmpnn" in name:
    topology["model_type"] = "ligand"
    topology["num_positional_embeddings"] = 32
  elif "membrane" in name:
    topology["physics_feature_dim"] = 3

  parts = name.split("_")
  if len(parts) >= 2:
    try:
      last_part = parts[-1].replace(".eqx", "").replace(".zst", "")
      if last_part.isdigit() and int(last_part) not in [32, 48, 30]:
        topology["atom_context_num"] = int(last_part)
    except ValueError:
      pass

  return topology


@dataclass(frozen=True)
class WeightProvenance:
  """Which checkpoint bytes a process actually loads, and what they hash to.

  The built wheel carries no ``model_params/*.eqx.zst``, so the aminx version pin does NOT
  determine the weights. Record ``sha256`` alongside the version in any result whose numbers
  depend on them.
  """

  filename: str
  source: str
  """``"packaged"`` if read from ``aminx.model_params``, ``"hub"`` if downloaded."""
  path: str
  sha256: str
  hub_repo_id: str | None = None
  hub_revision: str | None = None
  """Resolved snapshot commit when ``source == "hub"``. ``None`` if it could not be read."""


def _env_or_none(name: str) -> str | None:
  """Read an env var, treating set-but-blank as a configuration ERROR rather than as unset.

  ``FOO=`` is almost always an unset variable interpolated into a shell, CI or Docker env.
  Silently ignoring it would reproduce exactly the failure these settings exist to prevent --
  a weight source that changes with no signal -- so it fails closed instead.
  """
  raw = os.environ.get(name)
  if raw is None:
    return None
  value = raw.strip()
  if not value:
    msg = (
      f"{name} is set but blank ({raw!r}). Refusing to guess: a blank value is almost always "
      f"an unset variable interpolated into an environment, and silently ignoring it would "
      f"change the weight source with no signal. Unset {name}, or give it a real value."
    )
    raise ValueError(msg)
  return value


def normalise_checkpoint_filename(name: str) -> str:
  """Return the on-disk filename for a checkpoint id.

  Callers hold the BARE id (``"proteinmpnn_v_48_020"``) -- it is what ``--checkpoint-id`` and
  every spec file carry. Normalising only inside :func:`load_weights` meant
  :func:`weight_provenance` received the bare id and 404'd for a file that was sitting on
  disk, so this lives where every entry point reaches it. Idempotent.
  """
  return name if name.endswith(".zst") else f"{name}.eqx.zst"


def _reject_unsafe_filename(filename: str) -> None:
  """Reject a checkpoint name that would escape whatever directory it is joined to.

  ``Path("/pinned") / "/etc/passwd"`` is ``/etc/passwd`` under pathlib join semantics, and
  ``checkpoint_id`` reaches here unmodified from the CLI and from shared spec files.
  """
  candidate = Path(filename)
  if candidate.is_absolute() or ".." in candidate.parts:
    msg = (
      f"checkpoint filename {filename!r} must be a plain name relative to the weights "
      f"directory; absolute paths and '..' are rejected because they escape it."
    )
    raise ValueError(msg)


def _resolve_weight_path(filename: str) -> tuple[str, str]:
  """Return ``(source, path)`` for ``filename``: packaged resource first, then the Hub.

  Single source of truth for the resolution ORDER. Both :func:`_load_weight_bytes` and
  :func:`weight_provenance` go through here, so a provenance record can never describe a
  different file from the one that was loaded.

  Order: an authoritative ``AMINX_WEIGHTS_DIR`` (fails closed), then packaged resources, then
  the Hub at :data:`HF_REVISION`.

  SCOPE, and it is narrower than it looks. This governs resolution from a ``checkpoint_id``.
  It does NOT govern :func:`load_weights`/:func:`load_model`'s ``local_path`` argument --
  exposed as ``--model-local-path`` and as ``RunSpecification.model_local_path`` -- which
  opens the file it is given directly and by design. ``AMINX_WEIGHTS_DIR`` is therefore
  authoritative for checkpoint-id resolution, not for every route by which weights can enter
  the process; :func:`load_weights` warns when both are supplied so the override is never
  silent.
  """
  _reject_unsafe_filename(filename)
  filename = normalise_checkpoint_filename(filename)
  explicit_dir = _env_or_none(WEIGHTS_DIR_ENV)
  # Validated here rather than at the point of use: resolution short-circuits on the packaged
  # branch, so a blank AMINX_WEIGHTS_REVISION would otherwise go unchecked on exactly the path
  # most installs take, and only surface once something reached the Hub.
  revision = _env_or_none(REVISION_ENV) or HF_REVISION
  if explicit_dir:
    candidate = Path(explicit_dir) / filename
    if not candidate.is_file():
      msg = (
        f"{WEIGHTS_DIR_ENV}={explicit_dir!r} is set and is authoritative for checkpoint-id "
        f"resolution, but {filename!r} is not in it. Refusing to fall back to packaged "
        f"resources or the Hub: a silent change of weight source is the failure this setting "
        f"exists to prevent. Place the file there, or unset {WEIGHTS_DIR_ENV}."
      )
      raise FileNotFoundError(msg)
    return "explicit_dir", str(candidate)

  try:
    resource_path = files("aminx.model_params").joinpath(filename)
    # ``str()`` on a Traversable is only openable when it is backed by a real file. Check that
    # here rather than returning a path the caller cannot read: a zip/egg import yields a
    # Traversable whose ``is_file()`` is True but whose string form is not a filesystem path,
    # and falling through to the Hub is recoverable where an unhandled read error is not.
    if resource_path.is_file():
      packaged = Path(str(resource_path))
      if packaged.is_file():
        return "packaged", str(packaged)
  except (TypeError, ModuleNotFoundError, OSError) as exc:
    # Logged, not swallowed. Falling through to the Hub is the right recovery when the Hub is
    # reachable, but on an air-gapped compute node it turns a local, fixable PermissionError
    # into an opaque "Hub unreachable" failure with no breadcrumb back to the real cause.
    log.debug("packaged lookup failed for %s, falling through to the Hub: %r", filename, exc)

  log.info("Resolving %s from %s at %s (cached after first use)", filename, HF_REPO_ID, revision)
  return "hub", hf_hub_download(repo_id=HF_REPO_ID, filename=filename, revision=revision)


def _hub_revision_from_path(path: str) -> str | None:
  """Read the snapshot commit out of a Hub cache path, or ``None`` if absent.

  The cache layout is ``.../snapshots/<commit>/<filename>``. Returns ``None`` rather than
  raising when the layout differs, so provenance recording never breaks weight loading.
  """
  parts = Path(path).parts
  if "snapshots" not in parts:
    return None
  # Search from the RIGHT. `list.index` finds the first match, and a cache root that itself
  # contains a "snapshots" component (HF_HOME on shared scratch, say) would then yield the
  # wrong element -- a confidently-wrong revision is worse than None, because it looks real.
  index = len(parts) - 1 - parts[::-1].index("snapshots") + 1
  if index >= len(parts):
    return None
  candidate = parts[index]
  if not _COMMIT_SHA.fullmatch(candidate):
    log.warning(
      "cache path %r does not name a commit sha where one was expected (%r); recording the "
      "revision as unknown rather than guessing",
      path,
      candidate,
    )
    return None
  return candidate


def weight_provenance(filename: str) -> WeightProvenance:
  """Identify the checkpoint bytes this process would load for ``filename``.

  Resolves through the same path as :func:`_load_weight_bytes`, so the record describes the
  file that actually executes **for the checkpoint-id route**.

  It does NOT describe a run that supplied ``local_path`` / ``--model-local-path``, nor one
  that resolved through ``checkpoint_registry_path`` (which reaches the loader as a
  ``local_path``). Those routes bypass resolution by design; both log when taken.
  """
  source, path = _resolve_weight_path(filename)
  digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
  return WeightProvenance(
    # The NORMALISED name, so the record is unambiguous whether the caller passed a bare
    # checkpoint id or a full filename.
    filename=normalise_checkpoint_filename(filename),
    source=source,
    path=str(path),
    sha256=digest,
    hub_repo_id=HF_REPO_ID if source == "hub" else None,
    hub_revision=_hub_revision_from_path(path) if source == "hub" else None,
  )


def _load_weight_bytes(filename: str) -> bytes:
  """Return raw bytes for a weight file, trying local resources before HF Hub."""
  _, path = _resolve_weight_path(filename)
  return Path(path).read_bytes()


def load_weights(
  checkpoint_id: str | None = None,
  skeleton: eqx.Module | None = None,
  key: jax.Array | None = None,
  local_path: str | None = None,
) -> eqx.Module:
  """Load weights into a skeleton from internal resources or local path."""
  if checkpoint_id is None and local_path is None:
    if skeleton is None:
      msg = "skeleton is required for reinitialization"
      raise ValueError(msg)
    if key is None:
      key = jax.random.PRNGKey(0)

    params, static = eqx.partition(skeleton, eqx.is_inexact_array)
    param_leaves = jax.tree_util.tree_leaves(params)
    keys = jax.random.split(key, len(param_leaves))

    def initialize_param(param: jax.Array, key: jax.Array) -> jax.Array:
      shape = param.shape
      if len(shape) >= 2:
        return init.glorot_normal()(key, shape, param.dtype)
      return init.normal(stddev=0.01)(key, shape, param.dtype)

    initialized_leaves = [initialize_param(p, k) for p, k in zip(param_leaves, keys, strict=True)]
    new_params = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(params), initialized_leaves,
    )
    return eqx.combine(new_params, static)

  if local_path:
    # ALWAYS announced, not only when AMINX_WEIGHTS_DIR is set. `checkpoint_registry_path`
    # (host/prep.py) also arrives here as local_path, typically with no env var in play, so
    # gating on the env var left the project's own registry route silent -- the exact failure
    # class this work exists to close.
    log.info(
      "weights loaded from an explicit local_path=%r; weight_provenance() does NOT describe "
      "this route, so recording which weights ran is the caller's responsibility here",
      local_path,
    )
    # Deliberately os.environ.get, not _env_or_none: this route does not USE the variable, so
    # a blank value has no wrong-weights consequence here and should not raise. It only means
    # there is no authoritative dir to warn about being overridden.
    if os.environ.get(WEIGHTS_DIR_ENV):
      log.warning(
        "local_path=%r overrides %s=%r. The env var is authoritative for CHECKPOINT-ID "
        "resolution only; an explicit local_path bypasses it by design. Recording which "
        "weights ran is the caller's responsibility on this route -- weight_provenance() "
        "does not describe it.",
        local_path,
        WEIGHTS_DIR_ENV,
        os.environ.get(WEIGHTS_DIR_ENV),
      )
    with open(local_path, "rb") as f:
      header = f.read(4)
      f.seek(0)
      if header == b"\x28\xb5\x2f\xfd":  # zstd magic number (little-endian)
        dctx = zstd.ZstdDecompressor()
        stream = io.BytesIO(dctx.decompress(f.read()))
        return eqx.tree_deserialise_leaves(stream, skeleton)
    return eqx.tree_deserialise_leaves(local_path, skeleton)
  filename = normalise_checkpoint_filename(checkpoint_id)

  data = _load_weight_bytes(filename)
  dctx = zstd.ZstdDecompressor()
  stream = io.BytesIO(dctx.decompress(data))
  return eqx.tree_deserialise_leaves(stream, skeleton)


def load_model(
  checkpoint_id: str | None = None,
  model_weights: str | None = "original",
  local_path: str | None = None,
  key: jax.Array | None = None,
  *,
  use_electrostatics: bool = False,
  use_vdw: bool = False,
  dropout_rate: float = 0.1,
  use_side_chain_context: bool = False,
  # legacy parameter for backwards compatibility
  model_version: str | None = None,
  ligand_l_chunk: int | None = None,
) -> eqx.Module:
  """Load a fully instantiated Aminx-family model with weights.

  Skeleton topology (ligand vs protein, ``k_neighbors``, etc.) is inferred from the
  logical **checkpoint_id** string after legacy normalization — not from the basename
  of ``local_path``. When using ``local_path`` for weight bytes, ``checkpoint_id``
  must still name the intended checkpoint family (for example ``ligandmpnn_v_32_020_25``)
  so the correct module class and hyperparameters are built before deserialization.

  **Keys:** ``key`` affects random initialization for skeleton creation.

  """
  if key is None:
    key = jax.random.PRNGKey(0)

  # Support legacy API usage where checkpoint_id was named model_version
  if checkpoint_id and ("_v_" in checkpoint_id or "mpnn" in checkpoint_id):
    # Valid checkpoint id
    pass
  elif checkpoint_id and not model_version:
    # They passed 'v_48_020' into checkpoint_id
    model_version = checkpoint_id
    checkpoint_id = None

  if not checkpoint_id:
    if model_weights and model_version:
      filename = LEGACY_ALIAS_MAP.get((model_weights, model_version))
      if filename:
        checkpoint_id = filename
      else:
        checkpoint_id = f"{model_weights}_{model_version}.eqx.zst"
    else:
      checkpoint_id = "proteinmpnn_v_48_020.eqx.zst"

  if not checkpoint_id.endswith(".zst") and not local_path:
    checkpoint_id = f"{checkpoint_id}.eqx.zst"

  # Topology always follows the logical checkpoint id (ligand vs protein, kNN, ...).
  # ``local_path`` may be a bare ``artifact.eqx`` filename with no family/version hints.
  topo = get_topology_for_checkpoint(checkpoint_id)

  physics_feature_dim = topo["physics_feature_dim"]
  if use_electrostatics or use_vdw:
    physics_feature_dim = (5 if use_electrostatics else 0) + (5 if use_vdw else 0)

  model_type = topo["model_type"]
  if model_type == "ligand":
    ligand_skeleton_kw: dict[str, int] = {}
    if ligand_l_chunk is not None:
      ligand_skeleton_kw["ligand_l_chunk"] = ligand_l_chunk
    skeleton = PrxteinLigandMPNN(
      node_features=NODE_FEATURES,
      edge_features=EDGE_FEATURES,
      hidden_features=HIDDEN_FEATURES,
      num_encoder_layers=NUM_ENCODER_LAYERS,
      num_decoder_layers=NUM_DECODER_LAYERS,
      k_neighbors=topo["k_neighbors"],
      num_positional_embeddings=topo["num_positional_embeddings"],
      num_context_layers=NUM_LIGAND_CONTEXT_LAYERS,
      ligand_mpnn_use_side_chain_context=use_side_chain_context,
      **ligand_skeleton_kw,
      key=key,
    )
  elif model_type == "packer":
    skeleton = Packer(
      node_features=NODE_FEATURES,
      edge_features=EDGE_FEATURES,
      hidden_dim=HIDDEN_FEATURES,
      num_encoder_layers=NUM_ENCODER_LAYERS,
      num_decoder_layers=NUM_DECODER_LAYERS,
      top_k=topo["k_neighbors"],
      atom_context_num=topo["atom_context_num"],
      num_positional_embeddings=topo["num_positional_embeddings"],
      key=key,
    )
  else:
    skeleton = Aminx(
      node_features=NODE_FEATURES,
      edge_features=EDGE_FEATURES,
      hidden_features=HIDDEN_FEATURES,
      physics_feature_dim=physics_feature_dim if physics_feature_dim > 0 else None,
      num_encoder_layers=NUM_ENCODER_LAYERS,
      num_decoder_layers=NUM_DECODER_LAYERS,
      vocab_size=VOCAB_SIZE,
      k_neighbors=topo["k_neighbors"],
      num_positional_embeddings=topo["num_positional_embeddings"],
      dropout_rate=dropout_rate,
      key=key,
    )

  loaded = load_weights(
    checkpoint_id=checkpoint_id,
    local_path=local_path,
    skeleton=skeleton,
  )

  return loaded
