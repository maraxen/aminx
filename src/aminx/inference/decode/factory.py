"""Factory for resolving decode modes into typed mode classes (Task 11).

Implements make_decode_fn: converts (mode, strategy) pairs into concrete mode-class
instances (ConditionalDecode, UnconditionalDecode, AutoregressiveDecode, STEDecode),
dispatching each mode's state-axis strategy via tiling/dispatch.make_axis_dispatch_via_xtrax
(EPIC #1541 T2.GATE-passed flip, 2026-07-06).

Risk D-12 mitigation: wraps DispatchRejected from make_axis_dispatch_via_xtrax with
mode-name context to help users understand which mode failed the dispatch check.

Note: Stage-set projection for STE is handled internally by STEDecode (see ste.py).
"""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
from xtrax.tiling import CarryShape, JaxScanIterator

from aminx.inference.decode._base import _ConditionalDecodeBase
from aminx.inference.decode.conditional import ConditionalDecode
from aminx.inference.decode.mode import (
  AutoregressiveConfig,
  AutoregressiveMode,
  ConditionalMode,
  DecodeMode,
  STEMode,
  UnconditionalMode,
)
from aminx.inference.decode.ste import STEDecode
from aminx.inference.decode.unconditional import UnconditionalDecode
from aminx.tiling.dispatch import DispatchRejected, make_axis_dispatch_via_xtrax
from aminx.tiling.strategy import AxisStrategy
from aminx.utils.decoding_order import DecodingOrderFn, random_decoding_order

_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)


def make_decode_fn(
  model: Any,
  mode: DecodeMode,
  strategy: AxisStrategy,
  decoding_order_fn: DecodingOrderFn | None = None,
  autoregressive_config: AutoregressiveConfig | None = None,
) -> ConditionalDecode | UnconditionalDecode | AutoregressiveDecode | STEDecode:
  """Factory: resolve a decode mode and strategy into a typed mode-class instance.

  Dispatches by mode type, resolving the state-axis strategy via
  tiling/dispatch.make_axis_dispatch_via_xtrax. Wraps DispatchRejected with mode-name
  context (Risk D-12) to help users understand which mode caused rejection.

  Parameters
  ----------
  model : Any
      Model instance (passed through to mode-class constructor).
  mode : DecodeMode
      One of ConditionalMode, UnconditionalMode, AutoregressiveMode, STEMode.
  strategy : AxisStrategy
      State-axis strategy (Vmap, SafeMap, or Scan). Scan on state is rejected.
  decoding_order_fn : DecodingOrderFn, optional
      Consumed ONLY by ``STEMode`` (default: random_decoding_order). It is deliberately not
      used for ``AutoregressiveMode``: an AR decode reads the order from the bundle it is
      given (``bundle.wave`` / ``bundle.conditioning.ar_mask``), which is where a caller's
      function is applied -- see :func:`aminx.inference.bundle_builder.with_decoding_order`.
      Ignored for the conditional and unconditional modes.
  autoregressive_config : AutoregressiveConfig, optional
      Only consulted when ``mode`` is ``AutoregressiveMode``. Controls
      ``AutoregressiveDecode.use_while_loop`` via
      ``autoregressive_config.inference_only`` (default: False, the safe
      lax.scan/differentiable path). Set ``inference_only=True`` for
      inference-only sampling to get lax.while_loop's much faster compile
      (single XLA WhileOp vs. an unrolled Scan) -- never set True on any
      path that needs gradients through the AR loop. Ignored for non-AR
      modes.

  Returns
  -------
  ConditionalDecode | UnconditionalDecode | AutoregressiveDecode | STEDecode
      Concrete mode-class instance, ready to call.

  Raises
  ------
  DispatchRejected
      If strategy is invalid for the axis (e.g., Scan on state).
      Message includes mode-name context for clarity.

  Notes
  -----
  For AutoregressiveMode, the wave-axis iterator is always xtrax.tiling.JaxScanIterator
  (a structural invariant); only the state-axis strategy is user-configurable.
  The wave_carry is initialized with a default shape of (1024,) (typical L);
  the actual shape is materialized inside AutoregressiveDecode.__call__ from
  the encoder output.

  For STEMode, the factory recursively creates an inner mode-class via make_decode_fn
  with the STEMode.inner_mode, then wraps it in STEDecode.

  """
  if decoding_order_fn is None:
    decoding_order_fn = _DEFAULT_DECODING_ORDER_FN

  try:
    if isinstance(mode, ConditionalMode):
      state_iter = make_axis_dispatch_via_xtrax(strategy, axis="state")
      return ConditionalDecode(model=model, state_iterator=state_iter)

    if isinstance(mode, UnconditionalMode):
      state_iter = make_axis_dispatch_via_xtrax(strategy, axis="state")
      return UnconditionalDecode(model=model, state_iterator=state_iter)

    if isinstance(mode, AutoregressiveMode):
      state_iter = make_axis_dispatch_via_xtrax(strategy, axis="state")
      wave_iter = JaxScanIterator()
      # Default wave_carry shape (L=1024); actual shape materialized in __call__
      wave_carry = CarryShape(name="sequence", shape=(1024,), dtype=jnp.int32)
      ar_config = autoregressive_config if autoregressive_config is not None else AutoregressiveConfig()

      # Import here to avoid circular dependency
      from aminx.inference.decode.autoregressive import AutoregressiveDecode

      return AutoregressiveDecode(
        model=model,
        state_iterator=state_iter,
        wave_iterator=wave_iter,
        wave_carry=wave_carry,
        # inference_only defaults to False (safe for training); pass
        # autoregressive_config=AutoregressiveConfig(inference_only=True) for
        # inference-only sampling to get lax.while_loop's much faster compile.
        use_while_loop=ar_config.inference_only,
      )

    if isinstance(mode, STEMode):
      # Recursively build the inner mode
      inner = make_decode_fn(model, mode.inner_mode, strategy)
      # STE requires the inner to be a _ConditionalDecodeBase (ConditionalDecode)
      if not isinstance(inner, _ConditionalDecodeBase):
        raise TypeError(
          f"STEMode inner must be _ConditionalDecodeBase instance, got {type(inner)}",
        )
      # The order function is consumed by STEDecode itself (it draws the order it optimises
      # against); it used to be handed only to the inner mode, which ignores it, so the
      # caller's function was silently replaced by STEDecode's default.
      return STEDecode(
        inner=inner,
        iterations=mode.iterations,
        decoding_order_fn=decoding_order_fn,
      )

    # Exhaustiveness check: should never reach here if DecodeMode is sealed
    raise TypeError(f"Unknown mode type: {type(mode)}")

  except DispatchRejected as e:
    # Wrap with mode-name context (Risk D-12)
    mode_name = type(mode).__name__
    raise DispatchRejected(f"{mode_name}: {e}") from e


__all__ = ["make_decode_fn"]
