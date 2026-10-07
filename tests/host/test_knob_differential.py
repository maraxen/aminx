"""Differential harness: does a control knob a caller sets actually reach the model?

Ten times an aminx control knob has been declared on the config surface, appeared to work, and
silently never reached the model. Every one was found incidentally; none by a systematic sweep.
Existing conditioning tests miss them because they build `Protein`/bundle objects directly and
bypass the CLI -> manifest -> spec -> runner layer where all the bugs actually live.

This converts "is this knob wired?" from a code-reading judgment (which has failed every time)
into an executable assertion, driven through the REAL path:

    plan_campaign_manifest -> row["sampling_spec"] JSON -> SamplingSpecification(**payload)
                           -> [Tier B] real runner -> spy at load_model / build_inference_bundle

Tier A needs no model and no structure and runs in milliseconds; it covers the manifest-drop
class, which is 6 of the 10 known occurrences. Tier B spies at the model boundary for the hops
Tier A cannot see.

Expected behavior per field lives in `knob_observations.py` -- see that module for why declared
intent is required rather than a bare value-diff.

task_id: 260715_aminx-campaign-control-knob-audit
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from aminx.host.campaign import plan_campaign_manifest
from aminx.run.spec_json import migrate_removed_spec_keys
from aminx.run.specs import SamplingSpecification
from tests.host.knob_observations import (
  KNOWN_BROKEN_TIER_A,
  KNOWN_BROKEN_TIER_B,
  OBSERVATIONS,
  Bundle,
  Internal,
  LoadModel,
  NonSerializable,
  NotApplicable,
  Overridden,
  SpecOnly,
)

pytestmark = pytest.mark.knob_differential

_LIGAND_CKPT = "ligandmpnn_v_32_020_25"


def _plan_one_row(**spec_kwargs: Any) -> dict[str, Any]:
  """Drive the REAL planner and return the row's serialized sampling_spec."""
  base = SamplingSpecification(
    inputs=["ref.pdb"],
    checkpoint_id=_LIGAND_CKPT,
    **spec_kwargs,
  )
  rows = plan_campaign_manifest(
    base_spec=base,
    campaign_id="knob-differential",
    output_root="/tmp/knob-differential",
    designs_per_library_type=1,
    samples_chunk_size=1,
    # No arms: this harness probes whether a knob survives plan -> manifest -> spec, which
    # is orthogonal to fixing residues. Arms are covered in tests/host/test_fixed_arms.py.
    fixed_arms=None,
  )
  return dict(rows[0]["sampling_spec"])


def _reconstruct(payload: dict[str, Any]) -> SamplingSpecification:
  """Reconstruct exactly as the worker does (campaign.py:853-856).

  Deliberately mirrors the real worker rather than using spec_json: the harness must not
  diverge from the path under test, or it stops testing anything real.
  """
  worker_payload = migrate_removed_spec_keys(payload)
  return SamplingSpecification(**worker_payload)


def _round_trip(field: str, value: Any) -> Any:
  """Set `field` on base_spec, drive the real path, return the reconstructed value."""
  spec = _reconstruct(_plan_one_row(**{field: value}))
  return getattr(spec, field)


def _equal(a: Any, b: Any) -> bool:
  if a is None or b is None:
    return a is b
  try:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))
  except (TypeError, ValueError):
    return a == b


# Differential pairs: (field, value_a, value_b). Values must be legal on their own so a
# failure means "the knob was dropped", never "the spec rejected my input".
DIFFERENTIAL_PAIRS: dict[str, tuple[Any, Any]] = {
  "fixed_mask": (np.zeros(4, dtype=np.float32), np.array([1, 0, 1, 0], dtype=np.float32)),
  "fixed_positions": (np.zeros(4, dtype=np.float32), np.array([0, 1, 0, 1], dtype=np.float32)),
  "fixed_tokens": (np.zeros(4, dtype=np.int32), np.array([3, 3, 3, 3], dtype=np.int32)),
  "state_position_map": (np.zeros((1, 4), dtype=np.int32), np.arange(4, dtype=np.int32)[None, :]),
  "state_weights": (np.array([1.0], dtype=np.float32), np.array([0.5], dtype=np.float32)),
  "bias": (np.zeros((4, 21), dtype=np.float32), np.ones((4, 21), dtype=np.float32)),
  "tie_group_map": (np.zeros(4, dtype=np.int32), np.array([0, 0, 1, 1], dtype=np.int32)),
  "structure_mapping": (np.zeros(4, dtype=np.int32), np.arange(4, dtype=np.int32)),
  "multi_state_strategy": ("arithmetic_mean", "product"),
  "ligand_mpnn_use_side_chain_context": (False, True),
  "sidechain_conditioning": (False, True),
  "ligand_conditioning": (False, True),
  "use_electrostatics": (False, True),
  "use_vdw": (False, True),
  "temperature": (0.1, 0.9),
  "backbone_noise": (0.0, 0.25),
  "batch_size": (2, 8),
  "model_weights": ("original", "soluble"),
  "chain_id": ("A", "B"),
  "model_family": ("ligandmpnn", "proteinmpnn"),
}

_TIER_A_OBSERVABLE = (Bundle, LoadModel, SpecOnly)


def _tier_a_fields() -> list[str]:
  """Fields whose declared verdict implies the value must survive the manifest."""
  return sorted(
    name
    for name, verdict in OBSERVATIONS.items()
    if isinstance(verdict, _TIER_A_OBSERVABLE) and name in DIFFERENTIAL_PAIRS
  )


def _maybe_xfail(request: pytest.FixtureRequest, field: str, known_broken: Mapping[str, str]) -> None:
  """Apply a STRICT xfail marker so a fix XPASSes into a red build.

  This used to call `pytest.xfail(reason)` -- which is IMPERATIVE: it aborts the test body
  immediately and unconditionally marks xfail. The body never ran, so it could never XPASS, so
  the "a real fix forces the marker's removal" property I relied on all along **did not exist**.
  Proven when S1a fixed all 12 knobs and the suite still reported 12 xfailed; only `--runxfail`
  revealed they had started passing.

  A check that cannot fail and a check that cannot pass are the same bug wearing different hats.
  `request.node.add_marker` runs the body and detects XPASS, which is what strict xfail is for.
  """
  if field in known_broken:
    request.node.add_marker(pytest.mark.xfail(strict=True, reason=known_broken[field]))


class TestDeclarationCompleteness:
  """The drift guard: a new spec field with no declared verdict is a red build.

  This is what makes occurrence #11 structurally impossible. It is the same failure the audit
  exists to answer for -- `plan_campaign_manifest`'s hand-written literal drifting from an
  88-field dataclass with nothing enforcing the sync.
  """

  def test_every_spec_field_has_a_declared_observation(self) -> None:
    declared = set(OBSERVATIONS)
    actual = {f.name for f in fields(SamplingSpecification)}

    undeclared = actual - declared
    assert not undeclared, (
      f"Spec field(s) with no declared observation: {sorted(undeclared)}. "
      "Add an entry to tests/host/knob_observations.py declaring how this knob behaves in "
      "campaign mode and where it is observable. An undeclared field is exactly how the "
      "previous ten silent-knob bugs reached production."
    )

    stale = declared - actual
    assert not stale, f"Declared field(s) that no longer exist on the spec: {sorted(stale)}"

  @pytest.mark.xfail(
    strict=True,
    reason=(
      "~30 verdicts (mostly host/IO knobs: cache_path, split, max_workers, ...) still cite "
      "nothing -- they assert 'host-side cache location' rather than pointing at a line. The "
      "PREDICATE is correct and deliberately not weakened to make this pass; weakening it is "
      "what let the carry_specs fabrication through in the first place. The TABLE is what owes "
      "work. Cite them and this XPASSes -> red -> delete this marker. Tracked in sprint S5."
    ),
  )
  def test_every_reason_cites_resolvable_evidence(self) -> None:
    """A verdict whose reason asserts a conclusion without evidence is not a verdict.

    An earlier version of this test checked `len(verdict.reason) < 20` -- it tested reason
    LENGTH, not truth. Under it I declared carry_specs/dedup_specs as "streaming/sink
    bookkeeping (host/streaming.py); never touches the model", which is fabricated (they feed
    the planner at host/plan.py:79-80). 62 > 20, so it passed. An external review caught it,
    not this test.

    A predicate cheap to satisfy by writing more prose is not enforcement. This one demands a
    citation that actually RESOLVES against the source tree, so a reason cannot be satisfied by
    inventing a plausible-sounding file.
    """
    src = Path(__file__).resolve().parents[2] / "src" / "aminx"
    offenders: list[str] = []

    for name, verdict in OBSERVATIONS.items():
      # A citation is a source path (foo/bar.py[:123]) or a resolvable symbol reference.
      paths = re.findall(r"\b([\w/]+\.py)\b", verdict.reason)
      symbols = re.findall(r"\b(_?[a-z][\w]*(?:_[\w]+)+)\(\)", verdict.reason)
      if not paths and not symbols:
        offenders.append(f"{name}: no file.py or symbol() citation in reason")
        continue
      for rel in paths:
        # Reasons cite paths at varying depth -- "host/plan.py", a bare "_sampling_helper.py",
        # sometimes "aminx/run/specs.py". Resolve by basename anywhere under src/ (or tests/,
        # for reasons that cite a test as evidence). The point is that the cited file EXISTS,
        # not that the author wrote its full path.
        stem = rel.rsplit("/", 1)[-1]
        if not (any(src.rglob(stem)) or any((src.parents[1] / "tests").rglob(stem))):
          offenders.append(f"{name}: cites {rel!r}, which resolves to no file under src/")

    assert not offenders, (
      "Verdict reasons must cite evidence that resolves, not assert a conclusion:\n  "
      + "\n  ".join(offenders)
    )

  def test_known_broken_entries_are_declared(self) -> None:
    undeclared = (set(KNOWN_BROKEN_TIER_A) | set(KNOWN_BROKEN_TIER_B)) - set(OBSERVATIONS)
    assert not undeclared, f"known-broken names not in OBSERVATIONS: {sorted(undeclared)}"


class TestTierAManifestSurvival:
  """Does a caller's value survive plan -> manifest JSON -> reconstructed spec?

  No model, no structure, milliseconds. This is where 6 of the 10 known occurrences live.
  """

  @pytest.mark.parametrize("field", _tier_a_fields())
  def test_value_survives_the_real_round_trip(
    self, field: str, request: pytest.FixtureRequest,
  ) -> None:
    _maybe_xfail(request, field, KNOWN_BROKEN_TIER_A)
    value_a, value_b = DIFFERENTIAL_PAIRS[field]

    got_a = _round_trip(field, value_a)
    got_b = _round_trip(field, value_b)

    assert not _equal(got_a, got_b), (
      f"{field!r} is a SILENT NO-OP: caller set {value_a!r} vs {value_b!r}, but both "
      f"reconstruct to {got_a!r}. The value never survives the manifest, so the campaign "
      f"runs with the dataclass default regardless of what the caller asked for. "
      f"Declared verdict: {OBSERVATIONS[field]}"
    )


class TestOverriddenKnobsAreOverridden:
  """Campaign deliberately overrides these. Assert the override, not the caller's input.

  Without this, a naive differential reports all four as "not wired" -- false alarms that
  would discredit the harness immediately.
  """

  @pytest.mark.parametrize(
    "field",
    sorted(n for n, v in OBSERVATIONS.items() if isinstance(v, Overridden)),
  )
  def test_campaign_overrides_the_caller(self, field: str) -> None:
    verdict = OBSERVATIONS[field]
    assert isinstance(verdict, Overridden)

    payload = _plan_one_row()
    if field not in payload:
      pytest.skip(f"{field} is not carried by the manifest literal at all")

    spec = _reconstruct(payload)
    got = getattr(spec, field)

    if verdict.value is None:
      return  # per-row planner value; asserted by the planner's own tests
    assert got == verdict.value, (
      f"{field!r} should be overridden to {verdict.value!r} by the campaign planner "
      f"({verdict.reason}), got {got!r}. Either the override changed or the declaration is stale."
    )


class TestInertKnobsStayInert:
  """NOT_APPLICABLE knobs must remain absent from the manifest.

  If one starts being carried, the declaration is stale and must be re-adjudicated. This is
  the guard against a fix silently invalidating a triage verdict.
  """

  @pytest.mark.parametrize(
    "field",
    sorted(
      n for n, v in OBSERVATIONS.items()
      if isinstance(v, NotApplicable) and n in DIFFERENTIAL_PAIRS
    ),
  )
  def test_declared_inert_knob_is_still_inert(self, field: str) -> None:
    value_a, value_b = DIFFERENTIAL_PAIRS[field]
    got_a = _round_trip(field, value_a)
    got_b = _round_trip(field, value_b)

    assert _equal(got_a, got_b), (
      f"{field!r} is declared NOT_APPLICABLE in campaign mode "
      f"({OBSERVATIONS[field].reason}) but its value now SURVIVES the round trip "
      f"({got_a!r} vs {got_b!r}). The declaration is stale -- re-adjudicate it."
    )


class TestArrayKnobFidelity:
  """Array knobs must reach the worker as arrays, not as the JSON lists they travel as.

  These fields were DROPPED entirely by the old hand-written literal, so no campaign worker had
  ever received them. The field-driven write is the first time they arrive at all -- and they
  arrive as plain lists, because the worker's constructor deliberately bypasses spec_json's
  decoder (to keep strict unknown-key rejection).

  I originally reported this round-trip as "87/88 fields survive". That number came from a
  comparison containing `if not hasattr(a, "shape") else True` -- it auto-passed every array
  field, i.e. exactly the ones that change. An independent pass caught it. These tests exist so
  the claim is checked rather than asserted.
  """

  # ar_mask is deliberately absent: it is a DEAD spec field -- no consumer anywhere reads it
  # (build_inference_bundle derives its own from mode/schedule), per this table's own verdict.
  # Asserting fidelity on a value nothing consumes would be theatre.
  ARRAY_KNOBS = ("fixed_mask", "fixed_tokens", "bias", "state_weights", "tie_group_map",
                 "structure_mapping", "state_position_map")

  def _worker_spec(self, **spec_kwargs: Any) -> SamplingSpecification:
    """Reconstruct via the REAL worker path, including its coercion step."""
    from aminx.run.spec_json import _coerce_field_value

    payload = migrate_removed_spec_keys(_plan_one_row(**spec_kwargs))
    coerced = {k: _coerce_field_value(SamplingSpecification, k, v) for k, v in payload.items()}
    return SamplingSpecification(**coerced)

  @pytest.mark.parametrize("field", ARRAY_KNOBS)
  def test_array_knob_arrives_as_an_array(self, field: str) -> None:
    value_a, value_b = DIFFERENTIAL_PAIRS[field]
    spec = self._worker_spec(**{field: value_b})
    got = getattr(spec, field)

    assert hasattr(got, "shape"), (
      f"{field!r} reached the worker as {type(got).__name__}, not an array. It travels as a "
      f"JSON list and nothing restored the type, so any consumer doing numpy-style indexing "
      f"(x[:, None], x.shape, take_along_axis) breaks or misbehaves on it."
    )
    assert np.array_equal(np.asarray(got), np.asarray(value_b)), (
      f"{field!r} arrived as an array but with the wrong value"
    )

  def test_temperature_arrives_as_a_tuple(self) -> None:
    """temperature is Sequence[float]; __post_init__ coerces a float but never a list."""
    spec = self._worker_spec(temperature=0.9)
    assert isinstance(spec.temperature, tuple), (
      f"temperature reached the worker as {type(spec.temperature).__name__}, not tuple"
    )

  def test_tie_group_map_survives_kernel_dispatchs_raw_use(self) -> None:
    """The one consumer that does NOT wrap defensively -- found by an independent audit pass.

    `kernel_dispatch.py:143-147` does `jnp.atleast_2d(spec.tie_group_map)` and
    `spec.tie_group_map.shape[0]` on the RAW value. `jnp.atleast_2d` refuses a plain list
    outright (stricter than numpy's), so a single-structure campaign row setting tie_group_map
    hard-crashes without coercion. The PoE path (multistate_poe.py:310) wraps correctly; this
    one does not.

    Note the behavior history: before the field-driven write, such a row ran SILENTLY WITHOUT
    tying (the field was dropped). After it, the row crashed. After coercion, it works. Tier A
    could never have caught this -- it only checks the reconstructed spec's value, never drives
    it through kernel_dispatch.
    """
    import jax.numpy as jnp

    spec = self._worker_spec(tie_group_map=np.array([0, 0, 1, 1], dtype=np.int32))
    # The exact expression from kernel_dispatch.py:143-147.
    result = jnp.broadcast_to(
      jnp.atleast_2d(spec.tie_group_map), (2, spec.tie_group_map.shape[0]),
    )
    assert result.shape == (2, 4)

  def test_coercion_does_not_cost_strict_unknown_key_rejection(self) -> None:
    """The reason the worker keeps the plain constructor rather than spec_json's decoder.

    Coercion is a value transform, so an unknown key still reaches the constructor and raises.
    If this ever regresses, the manifest silently accepts typo'd keys -- trading one
    silent-failure class for another, which would be an absurd outcome for this audit.
    """
    from aminx.run.spec_json import _coerce_field_value

    payload = migrate_removed_spec_keys(_plan_one_row())
    payload["not_a_real_field"] = 123
    coerced = {k: _coerce_field_value(SamplingSpecification, k, v) for k, v in payload.items()}

    with pytest.raises(TypeError, match="not_a_real_field"):
      SamplingSpecification(**coerced)


class TestTierBModelBoundary:
  """Does the value reach the MODEL, not just the manifest?

  Tier A cannot see this hop, and that gap is dangerous rather than merely incomplete: the
  three knobs below are broken in TWO places -- dropped by the manifest AND never forwarded
  from prep.py to load_model. Fix only the manifest (4a) and all three XPASS Tier A while
  remaining broken, so their xfail markers get deleted and three real bugs are certified
  working. Tier B is what keeps that honest.

  Drives the real path: plan -> manifest JSON -> reconstructed spec -> prep_protein_stream_and_model
  -> load_model. Spies on load_model rather than loading real weights, so no checkpoint is needed.
  """

  # spec field -> the load_model kwarg it must arrive as
  FORWARDING_KNOBS = {
    "ligand_mpnn_use_side_chain_context": "use_side_chain_context",
    "use_electrostatics": "use_electrostatics",
    "use_vdw": "use_vdw",
  }

  @pytest.mark.parametrize("field", sorted(FORWARDING_KNOBS))
  def test_value_reaches_load_model(
    self, field: str, minimal_model: Any, monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
  ) -> None:
    _maybe_xfail(request, field, KNOWN_BROKEN_TIER_B)
    kwarg = self.FORWARDING_KNOBS[field]
    captured: list[dict[str, Any]] = []

    def _spy_load_model(**kwargs: Any) -> Any:
      captured.append(kwargs)
      return minimal_model

    monkeypatch.setattr("aminx.host.prep.load_model", _spy_load_model)

    from aminx.host.prep import prep_protein_stream_and_model

    def _kwarg_for(value: Any) -> Any:
      captured.clear()
      payload = _plan_one_row(**{field: value})
      payload["inputs"] = ["tests/data/1ubq.pdb"]
      spec = _reconstruct(payload)
      prep_protein_stream_and_model(spec)
      assert captured, "load_model was never called -- the spy is misplaced, not the knob broken"
      return captured[-1].get(kwarg, "<ABSENT>")

    got_false = _kwarg_for(False)
    got_true = _kwarg_for(True)

    assert got_false != got_true, (
      f"{field!r} never reaches the model: caller set False vs True, but load_model received "
      f"{kwarg}={got_false!r} both times. The model is built identically either way, so the knob "
      f"is inert at the model boundary even if it survives the manifest. "
      f"Declared verdict: {OBSERVATIONS[field]}"
    )


class TestNonSerializableKnobsRejected:
  """Callables must not silently vanish into the manifest.

  UPDATED for the field-driven write. Under the hand-written literal these fields were simply
  absent, so the test asserted absence. The dump is exhaustive over `fields()`, so a None-valued
  non-serializable field now appears AS None -- which round-trips correctly and carries no lie.

  The property that actually matters was never "is it absent"; it is **"can it silently carry a
  wrong value"**. So: None is fine, and a real value must RAISE rather than be dropped. That
  distinction is the whole audit.
  """

  @pytest.mark.parametrize(
    "field",
    sorted(n for n, v in OBSERVATIONS.items() if isinstance(v, (NonSerializable, Internal))),
  )
  def test_never_carries_a_value(self, field: str) -> None:
    payload = _plan_one_row()
    assert payload.get(field) is None, (
      f"{field!r} is declared non-serializable/internal but the manifest carries "
      f"{payload.get(field)!r}. A non-serializable field may appear as None (harmless, "
      f"round-trips); carrying a VALUE means it is serializable after all -- update the "
      f"declaration -- or the payload is lying."
    )

  def test_a_callable_knob_raises_rather_than_dropping_silently(self) -> None:
    """Occurrence #13, created by the fix for the first twelve, and caught by an audit agent.

    spec_json drops callables silently (`if callable(val): continue`), so `decoding_order_fn` --
    a real, supported knob -- still vanished from the manifest after the field-driven write.
    Worse, spec_partition's own documentation asserted this was impossible. A campaign that
    silently substitutes the default decoding order produces designs that differ from what was
    asked for, with no error: precisely the bug class this audit exists to end.
    """
    from aminx.host.spec_partition import SpecPartitionError

    def custom_order(*_args: Any, **_kwargs: Any) -> None:
      return None

    with pytest.raises(SpecPartitionError, match="decoding_order_fn"):
      _plan_one_row(decoding_order_fn=custom_order)

  def test_setting_a_non_serializable_field_raises_rather_than_drops(self) -> None:
    """The behavior change that is the point: silent drop -> loud error.

    Under the literal, setting carry_specs on a campaign base_spec was silently ignored. It now
    raises. No live caller sets it (cli.py's campaign sites set only inputs/return_logits/
    checkpoint_id/chain_id/ligand_context_path), so this breaks nobody and tells the truth.
    """
    from aminx.run.spec_json import SpecJSONEncodeError
    from xtrax.tiling import CarrySpec

    spec = CarrySpec(axis_name="n_samples", init={"x": 0.0}, transition=lambda c, x: (c, x))
    with pytest.raises(SpecJSONEncodeError, match="CarrySpec"):
      _plan_one_row(carry_specs=[spec])
