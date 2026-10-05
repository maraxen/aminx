"""ONNX-graph RNG walker (T2 step 3): the ONNX-side half of the RNG audit.

`aminx.export.rng_audit.find_rng_primitives` proves a traced JAX function is RNG-free
at the jaxpr level; this module proves the SAME thing about a converted ONNX graph,
which needs its own walk because ONNX's control-flow/reuse constructs are shaped
differently from jaxpr's (`If`/`Loop`/`Scan` GraphProto attributes instead of nested
Jaxprs, and `FunctionProto` call-by-(domain, op_type) instead of a jaxpr's inlined
`pjit`/`scan` params).

`find_onnx_rng_ops(model_proto)` walks the main graph, every subgraph attribute
(`If`/`Loop`/`Scan`, at any nesting depth), and every `FunctionProto` in
`model_proto.functions`. A node whose `(domain, op_type)` matches a local function's
`(domain, name)` is RESOLVED: the walker descends into that function body (recursively,
guarding against cycles with a visited-function-keys set) *before* any domain check is
applied to the call node itself -- a custom-domain call into a clean function must not
be misreported as an unknown-domain op, and a custom-domain call into a POISONED
function must still surface the RNG op inside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from onnx import AttributeProto

if TYPE_CHECKING:
  from onnx import FunctionProto, GraphProto, ModelProto, NodeProto

#: ONNX ops that draw randomness or sample from a distribution.
RNG_OPS: frozenset[str] = frozenset(
  {
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    "Multinomial",
    "Bernoulli",
  },
)

#: Domains an ONNX node can carry without being flagged as unusual. Anything else
#: (including the empty string standing for the default domain) that is not resolved
#: as a local-function call is reported.
_STANDARD_DOMAINS: frozenset[str] = frozenset({"", "ai.onnx", "ai.onnx.ml"})


def _subgraphs(node: NodeProto) -> list[tuple[str, GraphProto]]:
  """Every `(label, GraphProto)` directly attached to `node` as a GRAPH/GRAPHS attribute.

  Covers `If`'s `then_branch`/`else_branch`, `Loop`/`Scan`'s `body`, and any other
  op that carries a graph- or graph-list-typed attribute.
  """
  found: list[tuple[str, GraphProto]] = []
  for attr in node.attribute:
    if attr.type == AttributeProto.GRAPH:
      found.append((attr.name, attr.g))
    elif attr.type == AttributeProto.GRAPHS:
      found.extend((f"{attr.name}[{i}]", g) for i, g in enumerate(attr.graphs))
  return found


def find_onnx_rng_ops(model_proto: ModelProto) -> list[str]:
  """Recursively collect every RNG/unusual-domain finding reachable from `model_proto`.

  Args:
    model_proto: A parsed `onnx.ModelProto` (e.g. from `onnx.load`).

  Returns:
    A list of findings, one entry per occurrence (duplicates included), in traversal
    order. Each entry is one of:
      - `"rng:<path>/<OpType>"` -- an RNG op (`RNG_OPS`), wherever found;
      - `"ms-domain:<OpType>"` -- an unresolved node in domain `"com.microsoft"`;
      - `"unknown-domain:<domain>/<OpType>"` -- an unresolved node in any other
        non-standard domain.
    Empty means no RNG primitive and no unusual-domain node anywhere in the graph.
  """
  functions_by_key: dict[tuple[str, str], FunctionProto] = {
    (fn.domain, fn.name): fn for fn in model_proto.functions
  }
  found: list[str] = []

  def walk_nodes(
    nodes: Any,  # noqa: ANN401 -- RepeatedCompositeContainer[NodeProto], not import-stable
    path: str,
    visited_functions: frozenset[tuple[str, str]],
  ) -> None:
    for node in nodes:
      label = node.name or node.op_type
      node_path = f"{path}/{label}"
      key = (node.domain, node.op_type)

      if key in functions_by_key:
        # Resolved: descend into the function body BEFORE any domain check on the
        # call node itself (module docstring). Guard against a function that
        # (directly or transitively) calls itself.
        if key not in visited_functions:
          fn = functions_by_key[key]
          walk_nodes(fn.node, f"{node_path}::{node.op_type}", visited_functions | {key})
        continue

      if node.op_type in RNG_OPS:
        found.append(f"rng:{node_path}/{node.op_type}")
      elif node.domain == "com.microsoft":
        found.append(f"ms-domain:{node.op_type}")
      elif node.domain not in _STANDARD_DOMAINS:
        found.append(f"unknown-domain:{node.domain}/{node.op_type}")

      for sub_label, subgraph in _subgraphs(node):
        walk_nodes(subgraph.node, f"{node_path}/{sub_label}", visited_functions)

  walk_nodes(model_proto.graph.node, "graph", frozenset())
  return found
