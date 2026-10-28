"""Per-arm PRNG-key discipline for wave schedules.

Regression tests for the defect where `schedule_key` was never wired: `build_inference_bundle`
was the only caller of `build_wave_schedule` and substituted `jax.random.PRNGKey(0)` whenever
the caller passed nothing -- which no caller ever did. Every "randomized" schedule in the
package was therefore built from one constant key, and the measurable consequence was that
`random_ar` and `frozen_random_sigma` produced BYTE-IDENTICAL schedules.

That is not a cosmetic bug. Those two arms exist precisely to separate the
permutation-noise floor from the categorical floor (seed protocol M6), so a variance battery
comparing them measured exactly zero and would have reported a null caused by plumbing
rather than by the model.

The tests below pin the three properties that make the fix real:

1. The key discipline is DATA (`SCHEDULE_KEY_POLICY`), not a docstring instruction, and it
   covers every declared arm.
2. `build_wave_schedule_per_sample` applies it mechanically -- `random_ar` genuinely varies
   across the sample axis while `frozen_random_sigma` genuinely does not.
3. A missing key for a key-consuming arm RAISES instead of silently becoming PRNGKey(0).
"""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.schedule_selector import (
  SCHEDULE_KEY_POLICY,
  DecodingSchedule,
  build_wave_schedule,
  build_wave_schedule_per_sample,
  schedule_consumes_key,
  schedule_key_policy,
)

SEQ_LEN = 12
NUM_SAMPLES = 6

# Six tie groups of two positions each: exercises real tie-group grouping rather than the
# degenerate one-group-per-position case.
TIE_GROUP_MAP = jnp.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=jnp.int32)

_BUNDLE_FIELDS = ("group_ids", "group_positions", "group_valid", "position_valid")


def _fields(bundle) -> dict[str, np.ndarray]:
  """All four schedule arrays as numpy, for comparison."""
  return {name: np.asarray(getattr(bundle, name)) for name in _BUNDLE_FIELDS}


def _sample_slice(bundle, index: int) -> dict[str, np.ndarray]:
  """One sample's schedule out of a stacked bundle."""
  return {name: np.asarray(getattr(bundle, name))[index] for name in _BUNDLE_FIELDS}


def _identical(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
  """Whether two schedules agree on every field."""
  return all(np.array_equal(left[name], right[name]) for name in _BUNDLE_FIELDS)


class TestPolicyTable:
  """The key discipline is data, and it is complete."""

  def test_policy_covers_every_declared_arm(self):
    """Every arm in the DecodingSchedule literal has a policy, and vice versa.

    Guards the failure mode where a sixth arm is added and silently inherits no policy.
    """
    declared = set(typing.get_args(DecodingSchedule))
    assert declared == set(SCHEDULE_KEY_POLICY), (
      f"policy table and DecodingSchedule disagree: "
      f"only in literal={declared - set(SCHEDULE_KEY_POLICY)}, "
      f"only in table={set(SCHEDULE_KEY_POLICY) - declared}"
    )

  @pytest.mark.parametrize(
    ("mode", "expected"),
    [
      ("fixed_n_to_c", "unused"),
      ("random_ar", "per_sample"),
      ("frozen_random_sigma", "run_level"),
      ("chromatic", "unused"),
      ("improper_coloring", "run_level"),
    ],
  )
  def test_policy_values(self, mode, expected):
    """The per-arm policy matches seed protocol M6."""
    assert schedule_key_policy(mode) == expected

  def test_fixed_n_to_c_and_chromatic_ignore_the_key(self):
    """fixed_n_to_c and chromatic do not read a key."""
    assert not schedule_consumes_key("fixed_n_to_c")
    assert not schedule_consumes_key("chromatic")
    for mode in set(SCHEDULE_KEY_POLICY) - {"fixed_n_to_c", "chromatic"}:
      assert schedule_consumes_key(mode), mode

  def test_unknown_mode_raises(self):
    """An unknown arm is an error, not a default."""
    with pytest.raises(ValueError, match="Unknown schedule mode"):
      schedule_key_policy("not_a_schedule")  # type: ignore[arg-type]


class TestPerSampleStacking:
  """build_wave_schedule_per_sample applies the policy mechanically."""

  def test_random_ar_varies_across_samples(self):
    """THE REGRESSION TEST: random_ar must produce a different schedule per sample.

    Before the fix every sample shared PRNGKey(0), so all NUM_SAMPLES schedules were
    identical and the arm measured no permutation noise at all.
    """
    stacked = build_wave_schedule_per_sample(
      "random_ar",
      key=jax.random.PRNGKey(0),
      num_samples=NUM_SAMPLES,
      tie_group_map=TIE_GROUP_MAP,
    )
    first = _sample_slice(stacked, 0)
    differing = [i for i in range(1, NUM_SAMPLES) if not _identical(first, _sample_slice(stacked, i))]
    assert differing, (
      "random_ar produced identical schedules for every sample -- the per-sample key is "
      "not being split (this is the original defect)."
    )
    # Not merely "some differ": with 6 groups there are 720 orderings, so all 6 samples
    # differing from sample 0 is the expected outcome, and a weaker assertion would let a
    # partially-wired split through.
    assert len(differing) == NUM_SAMPLES - 1

  def test_frozen_random_sigma_is_constant_across_samples(self):
    """frozen_random_sigma must reuse one permutation -- that IS the arm."""
    stacked = build_wave_schedule_per_sample(
      "frozen_random_sigma",
      key=jax.random.PRNGKey(0),
      num_samples=NUM_SAMPLES,
      tie_group_map=TIE_GROUP_MAP,
    )
    first = _sample_slice(stacked, 0)
    for i in range(1, NUM_SAMPLES):
      assert _identical(first, _sample_slice(stacked, i)), f"sample {i} drifted"

  def test_random_ar_and_frozen_are_no_longer_the_same_arm(self):
    """The two arms must be distinguishable from the SAME run-level key.

    This is the property the original defect destroyed: given one key, both arms used to
    yield the same single schedule. Now the policy makes random_ar fan out while
    frozen_random_sigma stays put, so their sample-axis behaviour differs even though the
    run-level key is identical.
    """
    key = jax.random.PRNGKey(0)
    kwargs = {"num_samples": NUM_SAMPLES, "tie_group_map": TIE_GROUP_MAP}
    random_ar = build_wave_schedule_per_sample("random_ar", key=key, **kwargs)
    frozen = build_wave_schedule_per_sample("frozen_random_sigma", key=key, **kwargs)

    random_ar_unique = {
      _sample_slice(random_ar, i)["group_ids"].tobytes() for i in range(NUM_SAMPLES)
    }
    frozen_unique = {_sample_slice(frozen, i)["group_ids"].tobytes() for i in range(NUM_SAMPLES)}

    assert len(random_ar_unique) > 1, "random_ar collapsed to a single schedule"
    assert len(frozen_unique) == 1, "frozen_random_sigma should have exactly one schedule"

  def test_fixed_n_to_c_is_identity_and_constant(self):
    """fixed_n_to_c ignores the key and repeats the identity-ordered schedule."""
    stacked = build_wave_schedule_per_sample(
      "fixed_n_to_c",
      key=jax.random.PRNGKey(0),
      num_samples=NUM_SAMPLES,
      tie_group_map=TIE_GROUP_MAP,
    )
    first = _sample_slice(stacked, 0)
    for i in range(1, NUM_SAMPLES):
      assert _identical(first, _sample_slice(stacked, i))

    # Groups appear in ascending order, since the decoding order is arange(L).
    np.testing.assert_array_equal(first["group_ids"].ravel(), np.arange(6))

  def test_fixed_n_to_c_ignores_the_key_entirely(self):
    """Two different keys give the same fixed_n_to_c schedule."""
    kwargs = {"num_samples": 2, "tie_group_map": TIE_GROUP_MAP}
    a = build_wave_schedule_per_sample("fixed_n_to_c", key=jax.random.PRNGKey(0), **kwargs)
    b = build_wave_schedule_per_sample("fixed_n_to_c", key=jax.random.PRNGKey(99), **kwargs)
    assert _identical(_sample_slice(a, 0), _sample_slice(b, 0))

  @pytest.mark.parametrize("mode", ["random_ar", "fixed_n_to_c", "frozen_random_sigma"])
  def test_leading_axis_is_the_sample_axis(self, mode):
    """Every field gains a leading axis of length num_samples, for any arm.

    Uniformity across arms is what lets a caller vmap with in_axes=(0, 0) without
    branching on the schedule -- the arm's semantics live in the builder, not the caller.
    """
    stacked = build_wave_schedule_per_sample(
      mode,
      key=jax.random.PRNGKey(1),
      num_samples=NUM_SAMPLES,
      tie_group_map=TIE_GROUP_MAP,
    )
    single = build_wave_schedule(mode, key=jax.random.PRNGKey(1), tie_group_map=TIE_GROUP_MAP)
    for name in _BUNDLE_FIELDS:
      stacked_shape = np.asarray(getattr(stacked, name)).shape
      single_shape = np.asarray(getattr(single, name)).shape
      assert stacked_shape == (NUM_SAMPLES, *single_shape), name

  def test_shapes_are_permutation_invariant(self):
    """Stackability precondition: a permutation reorders groups, it does not add or remove.

    If this ever fails, per-sample schedules can no longer be stacked into one array and
    build_wave_schedule_per_sample's contract breaks -- so it is asserted directly rather
    than left implicit.
    """
    shapes = set()
    for seed in range(8):
      bundle = build_wave_schedule(
        "random_ar", key=jax.random.PRNGKey(seed), tie_group_map=TIE_GROUP_MAP
      )
      shapes.add(tuple(np.asarray(bundle.group_ids).shape))
    assert len(shapes) == 1, f"num_waves varied with the permutation: {shapes}"

  def test_num_samples_one_still_has_a_sample_axis(self):
    """num_samples=1 produces a leading axis of 1, not a squeezed bundle."""
    stacked = build_wave_schedule_per_sample(
      "random_ar", key=jax.random.PRNGKey(0), num_samples=1, tie_group_map=TIE_GROUP_MAP
    )
    assert np.asarray(stacked.group_ids).shape[0] == 1

  @pytest.mark.parametrize("num_samples", [0, -1])
  def test_non_positive_num_samples_raises(self, num_samples):
    """A zero or negative sample count is an error."""
    with pytest.raises(ValueError, match="num_samples must be positive"):
      build_wave_schedule_per_sample(
        "random_ar",
        key=jax.random.PRNGKey(0),
        num_samples=num_samples,
        tie_group_map=TIE_GROUP_MAP,
      )


class TestBundleBuilderRefusesToInventAKey:
  """A missing key for a key-consuming arm raises instead of defaulting to PRNGKey(0)."""

  @staticmethod
  def _minimal_structure_kwargs() -> dict:
    """The smallest valid structure inputs for build_inference_bundle."""
    return {
      "coords": jnp.zeros((SEQ_LEN, 4, 3), dtype=jnp.float32),
      "mask": jnp.ones((SEQ_LEN,), dtype=jnp.float32),
      "residue_index": jnp.arange(SEQ_LEN, dtype=jnp.int32),
      "chain_index": jnp.zeros((SEQ_LEN,), dtype=jnp.int32),
      "tie_group_map": TIE_GROUP_MAP,
    }

  @pytest.mark.parametrize(
    "mode", ["random_ar", "frozen_random_sigma", "improper_coloring"]
  )
  def test_missing_key_raises_for_key_consuming_arms(self, mode):
    """Every arm that reads a key refuses to run without one."""
    with pytest.raises(ValueError, match="consumes a PRNG key"):
      build_inference_bundle(schedule=mode, schedule_key=None, **self._minimal_structure_kwargs())

  def test_error_names_the_policy_and_the_host_side_escape_hatch(self):
    """The message has to be actionable, since this is the wiring bug's tripwire."""
    with pytest.raises(ValueError) as excinfo:
      build_inference_bundle(
        schedule="random_ar", schedule_key=None, **self._minimal_structure_kwargs()
      )
    message = str(excinfo.value)
    assert "per_sample" in message
    assert "build_wave_schedule_per_sample" in message

  def test_fixed_n_to_c_needs_no_key(self):
    """The default arm is unaffected -- no key, no error, no behaviour change."""
    bundle, _config = build_inference_bundle(
      schedule="fixed_n_to_c", schedule_key=None, **self._minimal_structure_kwargs()
    )
    assert bundle is not None

  def test_chromatic_needs_no_key(self):
    """Chromatic schedule is deterministic given adjacency and needs no key."""
    bundle, _config = build_inference_bundle(
      schedule="chromatic", schedule_key=None, **self._minimal_structure_kwargs()
    )
    assert bundle is not None

  def test_explicit_key_is_accepted(self):
    """Supplying the key is the supported path and does not raise."""
    bundle, _config = build_inference_bundle(
      schedule="random_ar",
      schedule_key=jax.random.PRNGKey(0),
      **self._minimal_structure_kwargs(),
    )
    assert bundle is not None

  def test_prebuilt_wave_bypasses_the_key_requirement(self):
    """Passing `wave` directly skips schedule construction, so no key is needed.

    This is the path a caller uses after building a per-sample stack host-side.
    """
    wave = build_wave_schedule(
      "random_ar", key=jax.random.PRNGKey(0), tie_group_map=TIE_GROUP_MAP
    )
    bundle, _config = build_inference_bundle(
      schedule="random_ar",
      schedule_key=None,
      wave=wave,
      **self._minimal_structure_kwargs(),
    )
    assert bundle is not None

  def test_build_wave_schedule_fixed_n_to_c_accepts_key_none(self):
    """build_wave_schedule('fixed_n_to_c') executes without key and generates 0..seq_len-1."""
    # Degenerate case: one tie group per position (identity map)
    identity_map = jnp.arange(SEQ_LEN, dtype=jnp.int32)
    bundle_id = build_wave_schedule("fixed_n_to_c", tie_group_map=identity_map, key=None)
    assert bundle_id is not None
    np.testing.assert_array_equal(np.asarray(bundle_id.group_ids).ravel(), np.arange(SEQ_LEN))

    # Real tie groups case: 6 tie groups of 2 positions each
    bundle_tied = build_wave_schedule("fixed_n_to_c", tie_group_map=TIE_GROUP_MAP, key=None)
    assert bundle_tied is not None
    np.testing.assert_array_equal(np.asarray(bundle_tied.group_ids).ravel(), np.arange(6))

  def test_build_wave_schedule_random_ar_requires_key(self):
    """build_wave_schedule('random_ar') raises ValueError when key is None."""
    with pytest.raises(ValueError, match="schedule mode 'random_ar' requires a PRNG `key`"):
      build_wave_schedule("random_ar", tie_group_map=TIE_GROUP_MAP, key=None)

