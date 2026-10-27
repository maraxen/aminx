"""HLO export and static memory comparison helpers.

Vendored from jaxbeans ``core/profiling.py`` for Phase 0 CI (no jaxbeans package
required on isolated checkouts). Upstream path:
``jaxbeans/src/jaxbeans/core/profiling.py`` (jaxbeans 0.1.0 tree).

Upstream sync: unknown (path-only vendored excerpt).

Behavior matches upstream; only docstrings/headers differ.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

import jax
from jax import profiler


def trace_kernel(name: str) -> Callable[[Callable], Callable]:
  """Decorator to add a named trace event for XLA profiling.

  Parameters
  ----------
  name : str
    Trace event label.

  Returns
  -------
  Callable[[Callable], Callable]
    Decorator wrapping the function.
  """

  def decorator(fn: Callable) -> Callable:
    """Wrap function with named trace annotation.

    Parameters
    ----------
    fn : Callable
      Function to wrap.

    Returns
    -------
    Callable
      Wrapped function.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
      """Trace execution wrapper.

      Parameters
      ----------
      args : Any
        Positional arguments.
      kwargs : Any
        Keyword arguments.

      Returns
      -------
      Any
        Function output.
      """
      with profiler.TraceAnnotation(name):
        return fn(*args, **kwargs)

    return wrapper

  return decorator


def start_trace(path: str = "/tmp/jax_trace") -> None:
  """Start JAX profiler trace (useful for capturing chrome://tracing files).

  Parameters
  ----------
  path : str, default="/tmp/jax_trace"
    Output trace directory path.

  Returns
  -------
  None
  """
  profiler.start_trace(path)


def stop_trace(log_message: str | None = None) -> None:
  """Stop active JAX profiler trace.

  Parameters
  ----------
  log_message : str | None, optional
    Optional status message logged when stopping trace.

  Returns
  -------
  None
  """
  del log_message
  profiler.stop_trace()


def export_hlo(func: Callable, *args: Any, **kwargs: Any) -> str:
  """Lower a JAX function and export its HLO as a string.

  Parameters
  ----------
  func : Callable
    JAX-traceable callable.
  args : Any
    Positional sample inputs for tracing.
  kwargs : Any
    Keyword sample inputs for tracing.

  Returns
  -------
  str
    Textual XLA HLO module string.
  """
  lowered = jax.jit(func).lower(*args, **kwargs)
  if lowered is None:
    raise ValueError("Failed to lower function to HLO.")
  compiler_ir = lowered.compiler_ir("hlo")
  if compiler_ir is None:
    raise ValueError("Failed to get compiler IR from lowered function.")
  return compiler_ir.as_hlo_text()


def analyze_memory(func: Callable, *args: Any, **kwargs: Any) -> str:
  """Perform static memory analysis on a lowered JAX kernel.

  Parameters
  ----------
  func : Callable
    JAX-traceable callable.
  args : Any
    Positional sample inputs for tracing.
  kwargs : Any
    Keyword sample inputs for tracing.

  Returns
  -------
  str
    Formatted static memory analysis string from compiler.
  """
  lowered = jax.jit(func).lower(*args, **kwargs)
  compiled = lowered.compile()
  analysis = compiled.memory_analysis()
  return str(analysis)


def assert_zero_copy_overhead(
  func_legacy: Callable,
  func_adapter: Callable,
  *args: Any,
  **kwargs: Any,
) -> None:
  """Verify adapter-based kernel has identical memory analysis to legacy.

  Parameters
  ----------
  func_legacy : Callable
    Reference legacy kernel.
  func_adapter : Callable
    Adapter or modernized kernel under test.
  args : Any
    Sample arguments for compiler memory analysis.
  kwargs : Any
    Sample keyword arguments for compiler memory analysis.

  Returns
  -------
  None
  """
  mem_legacy = analyze_memory(func_legacy, *args, **kwargs)
  mem_adapter = analyze_memory(func_adapter, *args, **kwargs)

  if mem_legacy != mem_adapter:
    raise AssertionError(
      "Memory analysis mismatch between legacy and adapter kernels.\n"
      f"Legacy Memory:\n{mem_legacy}\n"
      f"Adapter Memory:\n{mem_adapter}",
    )


_HLO_INSTR_RE = re.compile(r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*\S+\s+([a-zA-Z][\w.\-]*)\(")
_CUSTOM_CALL_TARGET_RE = re.compile(r'custom_call_target="([^"]+)"')
_COLLECTIVE_OPS: frozenset[str] = frozenset({
  "all-reduce",
  "all-to-all",
  "all-gather",
  "collective-permute",
  "reduce-scatter",
})
_DYNAMIC_SHAPE_OPS: frozenset[str] = frozenset({
  "set-dimension-size",
  "get-dimension-size",
  "dynamic-reshape",
})


def parse_hlo_op_histogram(hlo_text: str) -> dict[str, int]:
  """Count HLO leaf-instruction primitive names (e.g. "add", "dot", "fusion") in ``hlo_text``.

  Parameters
  ----------
  hlo_text : str
    Exported HLO representation.

  Returns
  -------
  dict[str, int]
    Histogram mapping primitive opcode name to count.
  """
  histogram: dict[str, int] = {}
  for line in hlo_text.splitlines():
    m = _HLO_INSTR_RE.match(line)
    if m:
      op = m.group(1)
      histogram[op] = histogram.get(op, 0) + 1
  return histogram


def check_hlo_smells(
  hlo_text: str,
  *,
  allow_lapack_ffi: bool = True,
  disallow_python_callbacks: bool = True,
  disallow_collectives: bool = True,
  disallow_dynamic_shapes: bool = True,
  disallow_unvetted_custom_calls: bool = False,
  allowed_custom_call_targets: Sequence[str] | None = None,
) -> list[str]:
  """Inspect compiled HLO text for performance, portability, and host-leak smells.

  Parameters
  ----------
  hlo_text : str
    Exported HLO representation of a lowered JAX program.
  allow_lapack_ffi : bool, default=True
    Whether to permit LAPACK FFI custom calls (e.g. lapack_sgesdd_ffi for matrix decomposition).
  disallow_python_callbacks : bool, default=True
    Whether to flag Python host callbacks (e.g. xla_ffi_python_cpu_callback).
  disallow_collectives : bool, default=True
    Whether to flag distributed collective instructions (e.g. all-reduce, all-to-all)
    on expected single-device programs.
  disallow_dynamic_shapes : bool, default=True
    Whether to flag dynamic shape manipulation instructions or bounded dynamic shapes.
  disallow_unvetted_custom_calls : bool, default=False
    Whether to flag any custom call not explicitly listed in ``allowed_custom_call_targets``.
  allowed_custom_call_targets : Sequence[str] | None, optional
    Set of allowed custom call target strings when ``disallow_unvetted_custom_calls`` is True.

  Returns
  -------
  list[str]
    List of detected smell descriptions. Empty if the HLO is free of configured smells.
  """
  smells: list[str] = []
  allowed_targets = set(allowed_custom_call_targets or ())
  if allow_lapack_ffi:
    allowed_targets.update({"lapack_sgesdd_ffi", "lapack_dgesdd_ffi"})

  for line in hlo_text.splitlines():
    if "custom-call" in line:
      match = _CUSTOM_CALL_TARGET_RE.search(line)
      target = match.group(1) if match else "unknown"
      if disallow_python_callbacks and (
        "python_cpu_callback" in target or "python_callback" in target or "HostCallback" in target
      ):
        smells.append(f"Host Python callback detected: {target!r} in {line.strip()[:100]}")
      elif disallow_unvetted_custom_calls and target not in allowed_targets:
        smells.append(f"Unvetted custom call target: {target!r} in {line.strip()[:100]}")

    m = _HLO_INSTR_RE.match(line)
    if m:
      op = m.group(1)
      if disallow_collectives and op in _COLLECTIVE_OPS:
        smells.append(f"Distributed collective instruction detected: {op!r} in {line.strip()[:100]}")
      if disallow_dynamic_shapes and op in _DYNAMIC_SHAPE_OPS:
        smells.append(f"Dynamic shape instruction detected: {op!r} in {line.strip()[:100]}")

  if disallow_dynamic_shapes and "<=" in hlo_text:
    for line in hlo_text.splitlines():
      if "<=" in line and ("f32[" in line or "s32[" in line or "u32[" in line):
        smells.append(f"Bounded dynamic shape detected: {line.strip()[:100]}")
        break

  return smells


def assert_no_hlo_smells(hlo_text: str, **kwargs: Any) -> None:
  """Assert that ``hlo_text`` contains no HLO smells, raising AssertionError with details.

  Parameters
  ----------
  hlo_text : str
    Exported HLO string to inspect.
  kwargs : Any
    Forwarded to :func:`check_hlo_smells`.

  Returns
  -------
  None
  """
  smells = check_hlo_smells(hlo_text, **kwargs)
  if smells:
    formatted = "\n  - ".join(smells)
    raise AssertionError(f"HLO smell check failed with {len(smells)} issue(s):\n  - {formatted}")

