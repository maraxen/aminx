"""Shared test fixtures."""

import json
import os
import socket
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax import random

from aminx.utils.data_structures import Protein
from aminx.types.arrays import ModelParameters


def _hf_hub_reachable() -> bool:
  """True when huggingface.co:443 accepts a TCP connection.

  The timeout is per socket. ``setdefaulttimeout`` would stick for the rest of
  the process, including a later multi-hundred-MB ``hf_hub_download``.
  """
  try:
    with socket.create_connection(("huggingface.co", 443), timeout=5):
      return True
  except OSError:
    return False


def _local_weights_exist() -> bool:
  model_params = Path(__file__).parent.parent / "src" / "aminx" / "model_params"
  return any(model_params.glob("*.eqx.zst"))


_WEIGHTS_AVAILABLE: bool = _local_weights_exist() or _hf_hub_reachable()


def pytest_configure(config: pytest.Config) -> None:
  config.addinivalue_line(
    "markers",
    "requires_weights: test requires model weight files (bundled or HF Hub)",
  )


def pytest_runtest_setup(item: pytest.Item) -> None:
  if item.get_closest_marker("requires_weights") and not _WEIGHTS_AVAILABLE:
    pytest.skip("requires_weights: no bundled weights and HF Hub unreachable")


@pytest.fixture(scope="session")
def checkpoint_local_path():
  """Resolve checkpoint_id to local path if bundled, else HF Hub cache path."""
  def _resolve(checkpoint_id: str) -> str:
    fname = checkpoint_id if checkpoint_id.endswith(".eqx.zst") else f"{checkpoint_id}.eqx.zst"
    local = Path(__file__).parent.parent / "src" / "aminx" / "model_params" / fname
    if local.exists():
      return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id="maraxen/aminx", filename=fname)
  return _resolve


def _parse_first_structure(path: Path) -> Protein:
    """Parse a single structure and skip test session if parser backend is unavailable."""
    try:
        from aminx.io.parsing import parse_input
    except ModuleNotFoundError as exc:
        pytest.skip(f"Skipping parsing-dependent tests: {exc}")
    try:
        return next(parse_input(str(path)))
    except ModuleNotFoundError as exc:
        pytest.skip(f"Skipping parsing-dependent tests: {exc}")


@pytest.fixture(scope="session")
def protein_structure() -> Protein:
    """Load a sample protein structure from a PDB file."""
    pdb_path = Path(__file__).parent / "data" / "1ubq.pdb"
    return _parse_first_structure(pdb_path)


@pytest.fixture(scope="session")
def pqr_protein_tuple() -> Protein:
    """Load a sample protein structure from a PQR file."""
    pqr_path = Path(__file__).parent / "data" / "1a00.pqr"
    return _parse_first_structure(pqr_path)


@pytest.fixture(scope="session")
def model_inputs(protein_structure: Protein) -> dict:
    """Create model inputs from a protein structure."""
    return {
        "structure_coordinates": protein_structure.coordinates,
        "mask": protein_structure.mask,
        "residue_index": protein_structure.residue_index,
        "chain_index": protein_structure.chain_index,
        "sequence": protein_structure.aatype,
    }


@pytest.fixture
def rng_key() -> random.PRNGKey:
    """Create a new random key for testing."""
    return random.PRNGKey(0)


@pytest.fixture
def mock_model_parameters() -> ModelParameters:
  """Create a complete and structurally correct set of mock model parameters."""
  # Model dimensions
  C_V = 128  # Node feature dimension
  C_E = 128  # Edge feature dimension
  INITIAL_EDGE_FEATURES = 528
  ENCODER_MLP_INPUT_DIM = C_V + C_V + C_E  # 384
  # CORRECTED: Define the decoder's specific input dimension
  DECODER_MLP_INPUT_DIM = C_V + ENCODER_MLP_INPUT_DIM # 128 + 384 = 512
  NUM_AMINO_ACIDS = 21
  MAXIMUM_RELATIVE_FEATURES = 32
  pos_enc_dim = 2 * MAXIMUM_RELATIVE_FEATURES + 2

  # Helper to create mock linear layer parameters
  def _make_linear_params(d_in, d_out):
    return {"w": jax.random.normal(jax.random.PRNGKey(0), (d_in, d_out)), "b": jax.random.normal(jax.random.PRNGKey(0), (d_out,))}

  # Helper to create mock norm layer parameters
  def _make_norm_params(dim=C_V):
    return {"scale": jnp.ones((dim,)), "offset": jnp.zeros((dim,))}

  params = {
    # Feature extraction parameters
    "protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear": _make_linear_params(
      pos_enc_dim, C_E,
    ),
    "protein_mpnn/~/protein_features/~/edge_embedding": _make_linear_params(
      INITIAL_EDGE_FEATURES, C_E,
    ),
    "protein_mpnn/~/protein_features/~/norm_edges": _make_norm_params(dim=C_E),
    # Main model parameters
    "protein_mpnn/~/W_e": _make_linear_params(C_E, C_E),
    "protein_mpnn/~/embed_token": {"W_s": jnp.ones((NUM_AMINO_ACIDS, C_V))},
    "protein_mpnn/~/W_out": _make_linear_params(C_V, NUM_AMINO_ACIDS),
  }

  # Encoder and Decoder Layers
  for i in range(3):
    enc_l_name = f"enc{i}"
    dec_l_name = f"dec{i}"
    enc_prefix = "protein_mpnn/~/enc_layer"
    if i > 0:
        enc_prefix += f"_{i}"
    dec_prefix = "protein_mpnn/~/dec_layer"
    if i > 0:
        dec_prefix += f"_{i}"

    # Encoder
    params.update(
      {
        f"{enc_prefix}/~/{enc_l_name}_norm1": _make_norm_params(),
        f"{enc_prefix}/~/{enc_l_name}_norm2": _make_norm_params(),
        f"{enc_prefix}/~/{enc_l_name}_norm3": _make_norm_params(),
        f"{enc_prefix}/~/{enc_l_name}_W1": _make_linear_params(ENCODER_MLP_INPUT_DIM, C_V),
        f"{enc_prefix}/~/{enc_l_name}_W2": _make_linear_params(C_V, C_V),
        f"{enc_prefix}/~/{enc_l_name}_W3": _make_linear_params(C_V, C_V),
        f"{enc_prefix}/~/{enc_l_name}_W11": _make_linear_params(ENCODER_MLP_INPUT_DIM, C_E),
        f"{enc_prefix}/~/{enc_l_name}_W12": _make_linear_params(C_E, C_E),
        f"{enc_prefix}/~/{enc_l_name}_W13": _make_linear_params(C_E, C_E),
        f"{enc_prefix}/~/position_wise_feed_forward/~/{enc_l_name}_dense_W_in": _make_linear_params(
          C_V, C_V,
        ),
        f"{enc_prefix}/~/position_wise_feed_forward/~/{enc_l_name}_dense_W_out": _make_linear_params(
          C_V, C_V,
        ),
      },
    )
    # Decoder
    params.update(
      {
        f"{dec_prefix}/~/{dec_l_name}_norm1": _make_norm_params(),
        f"{dec_prefix}/~/{dec_l_name}_norm2": _make_norm_params(),
        f"{dec_prefix}/~/{dec_l_name}_W1": _make_linear_params(DECODER_MLP_INPUT_DIM, C_V),
        f"{dec_prefix}/~/{dec_l_name}_W2": _make_linear_params(C_V, C_V),
        f"{dec_prefix}/~/{dec_l_name}_W3": _make_linear_params(C_V, C_V),
        f"{dec_prefix}/~/position_wise_feed_forward/~/{dec_l_name}_dense_W_in": _make_linear_params(
          C_V, C_V,
        ),
        f"{dec_prefix}/~/position_wise_feed_forward/~/{dec_l_name}_dense_W_out": _make_linear_params(
          C_V, C_V,
        ),
      },
    )

  return params

@pytest.fixture(params=[False, True], ids=["eager", "jit"])
def apply_jit(request):
    """Returns a function that conditionally JITs the input function."""
    should_jit = request.param

    def _wrapper(fn, **kwargs):
        if should_jit:
            return jax.jit(fn, **kwargs)
        return fn

    return _wrapper


@pytest.fixture
def minimal_bundle_fixture():
    """Minimal InferenceBundle for smoke testing.

    Returns a valid InferenceBundle with small but proper array shapes:
    S=1 (one state), L=4 (sequence length), K=8 (neighbors), D=16 (embed dim).
    """
    from aminx.types.bundles import (
        GeometryBundle,
        ConditioningBundle,
        LigandBundle,
        WaveScheduleBundle,
        InferenceBundle,
    )

    S, L, K, D = 1, 4, 8, 16
    V = 21  # vocab size (amino acids)

    # Minimal geometry
    geometry = GeometryBundle(
        coords=jnp.zeros((S, L, 4, 3)),  # S, L, 4 atoms, 3D
        mask=jnp.ones((S, L)),
        residue_index=jnp.arange(L)[None, :].astype(jnp.int32).repeat(S, axis=0),
        chain_index=jnp.zeros((S, L), dtype=jnp.int32),
        n_states=S,
        n_canonical=20,
        n_flat=L,
    )

    # Minimal conditioning
    conditioning = ConditioningBundle(
        fixed_mask=jnp.zeros(L),
        fixed_tokens=jnp.zeros(L, dtype=jnp.int32),
        bias=jnp.zeros((L, V)),
        tie_group_map=jnp.zeros((S, L), dtype=jnp.int32),
        state_position_map=jnp.broadcast_to(jnp.arange(L)[None, :], (S, L)),
        state_weights=jnp.ones(S),
        sequence_oh=jnp.zeros((L, V)),
        ar_mask=jnp.ones((S, L, L)),
    )

    # Minimal ligand (empty)
    ligand = LigandBundle(
        ligand_coords=jnp.zeros((S, L, 1, 3)),
        ligand_atom_types=jnp.zeros((S, L, 1), dtype=jnp.int32),
        ligand_mask=jnp.zeros((S, L, 1)),
    )

    # Minimal wave schedule
    wave = WaveScheduleBundle.empty(L)

    # Construct the bundle
    bundle = InferenceBundle(
        geometry=geometry,
        conditioning=conditioning,
        ligand=ligand,
        wave=wave,
        backbone_noise=jnp.array(0.0),
    )

    return bundle


@pytest.fixture
def minimal_encode_fn_fixture():
    """Minimal encode function fixture for testing.

    Returns a tuple (encode_fn, config) where encode_fn is a callable that
    accepts the minimal test arguments and returns an EncoderOutput.
    """
    from aminx.types.encodings import EncoderOutput
    from aminx.types.configs import InferenceConfig

    L, K, D = 4, 8, 16

    def minimal_encode_fn(*args, **kwargs):
        """Minimal encode function that returns a valid EncoderOutput."""
        return EncoderOutput(
            node_features=jnp.zeros((L, D)),
            edge_features=jnp.zeros((L, K, D)),
            neighbor_indices=jnp.zeros((L, K), dtype=jnp.int32),
        )

    # Return the function and a minimal config
    config = InferenceConfig()
    return (minimal_encode_fn, config)


def _port_wave_active() -> str | None:
  """Return AMINX_PORT_WAVE when the redsox port hooks must run."""
  wave = os.environ.get("AMINX_PORT_WAVE")
  if not wave:
    return None
  return wave


def _declared_port_wave(item: pytest.Item) -> str:
  """Wave declared on the item; unmarked tests belong to the __nonport__ bucket."""
  mark = item.get_closest_marker("port_wave")
  if mark is None or not mark.args:
    return "__nonport__"
  return str(mark.args[0])


def _redsox_ids(path: Path) -> list[str]:
  ids: list[str] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    nodeid = line.strip()
    if nodeid:
      ids.append(nodeid)
  return ids


def pytest_sessionstart(session: pytest.Session) -> None:
  """Enter the mutant context manager named by AMINX_PORT_MUTANT.

  Inert unless AMINX_PORT_WAVE is set, so ordinary test sessions are unchanged.
  """
  if _port_wave_active() is None:
    return
  mutant = os.environ.get("AMINX_PORT_MUTANT")
  if not mutant:
    return
  from port.mutant_registry import load_mutant

  context = load_mutant(mutant)
  context.__enter__()
  session.config._aminx_port_mutant_cm = context  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session) -> None:
  """Exit a mutant context entered at session start. Inert when no wave is set."""
  if _port_wave_active() is None:
    return
  context = getattr(session.config, "_aminx_port_mutant_cm", None)
  if context is None:
    return
  context.__exit__(None, None, None)


def pytest_collection_modifyitems(
  config: pytest.Config,
  items: list[pytest.Item],
) -> None:
  """Deselect items outside the active wave or the redsox id file.

  Inert unless AMINX_PORT_WAVE is set. Deselected items are removed, never skipped.
  A listed id that pytest did not collect is a UsageError.
  """
  wave = _port_wave_active()
  if wave is None:
    return
  # The ids file is the gate's selection (spec §6.6 step 1); plain per-wave runs omit it and
  # select every item declared for the active wave.
  select = os.environ.get("AMINX_REDSOX_SELECT")
  allowed: set[str] | None = None
  if select:
    ids_path = Path(select)
    if not ids_path.is_file():
      msg = f"AMINX_REDSOX_SELECT does not exist: {ids_path}"
      raise pytest.UsageError(msg)
    listed = _redsox_ids(ids_path)
    collected = {item.nodeid for item in items}
    missing = [nodeid for nodeid in listed if nodeid not in collected]
    if missing:
      preview = ", ".join(missing[:5])
      msg = f"AMINX_REDSOX_SELECT ids were not collected ({len(missing)}): {preview}"
      raise pytest.UsageError(msg)
    allowed = set(listed)
  selected: list[pytest.Item] = []
  deselected: list[pytest.Item] = []
  for item in items:
    if _declared_port_wave(item) != wave or (allowed is not None and item.nodeid not in allowed):
      deselected.append(item)
    else:
      selected.append(item)
  if deselected:
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
  item: pytest.Item,
  call: pytest.CallInfo[None],
):
  """Copy call.excinfo.typename onto report.user_properties.

  Inert unless AMINX_PORT_WAVE is set.
  """
  outcome = yield
  if _port_wave_active() is None:
    return
  report = outcome.get_result()
  if call.excinfo is None:
    return
  report.user_properties.append(("exc_type", call.excinfo.typename))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
  """Append one redsox outcome record. Inert unless AMINX_PORT_WAVE is set."""
  wave = _port_wave_active()
  if wave is None:
    return
  outcomes = os.environ.get("AMINX_REDSOX_OUTCOMES")
  if not outcomes:
    return
  exc_type = None
  for key, value in report.user_properties:
    if key == "exc_type":
      exc_type = value
  mutant = os.environ.get("AMINX_PORT_MUTANT") or None
  record = {
    "nodeid": report.nodeid,
    "wave": wave,
    "when": report.when,
    "outcome": report.outcome,
    "wasxfail": bool(getattr(report, "wasxfail", False)),
    "mutant": mutant,
    "exc_type": exc_type,
  }
  path = Path(outcomes)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True))
    handle.write("\n")


# --- jax_enable_x64 leak guard (debt #2419, #2431) --------------------------
# Two modules once set this process-global flag and never restored it, so every
# test running afterwards built float64 arrays. The damage lands far from the
# cause -- a float32 checkpoint refusing to deserialise into a freshly built
# float64 `like`, or a lax.cond whose branches disagree int64 vs int32 -- and it
# cost 17 of the 18 tests/host failures in #2419 plus a wrong root-cause filing
# before it was found.
#
# This restores the flag after any test that changes it, which is what actually
# stops the cascade, and names every offender at session end. It deliberately
# does NOT fail the leaking test: pytest_runtest_logfinish fires after fixture
# finalizers, so a correctly-restoring fixture will not trip it, but making a
# global-state check fatal on a suite this large is a bigger change than the
# bug warrants. The printed summary is the signal to go fix the fixture.
_X64_BASELINE: list[bool] = []
_X64_LEAKS: list[str] = []


def pytest_sessionstart(session: object) -> None:  # noqa: ARG001
  """Record the flag as the session found it."""
  import jax  # noqa: PLC0415

  _X64_BASELINE.append(bool(jax.config.jax_enable_x64))


def pytest_runtest_logfinish(nodeid: str, location: object) -> None:  # noqa: ARG001
  """After the whole test, including fixture teardown, put the flag back."""
  import jax  # noqa: PLC0415

  if not _X64_BASELINE:
    return
  baseline = _X64_BASELINE[0]
  if bool(jax.config.jax_enable_x64) != baseline:
    _X64_LEAKS.append(nodeid)
    jax.config.update("jax_enable_x64", baseline)


def pytest_sessionfinish(session: object, exitstatus: object) -> None:  # noqa: ARG001
  """Name the leakers, so the fixture gets fixed rather than papered over."""
  if not _X64_LEAKS:
    return
  print(  # noqa: T201
    f"\njax_enable_x64 LEAKED by {len(_X64_LEAKS)} test(s); the flag was restored "
    "after each, but scope it in the fixture (see tests/ebm/conftest.py):",
  )
  for nodeid in _X64_LEAKS:
    print(f"  {nodeid}")  # noqa: T201
