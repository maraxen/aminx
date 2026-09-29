"""Issue 2051: ``Aminx.__call__`` forwards ``inference`` into the encoder.

(a) ``inference=True`` is bit-identical across PRNG keys (encoder features and logits).
(b) ``inference=False`` is not: negative control that dropout still depends on the key.
(c) the training encode (``inference=False``) still applies dropout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aminx.model.mpnn import Aminx

_L = 6
_D = 32


def _model() -> Aminx:
  return Aminx(
    node_features=_D,
    edge_features=_D,
    hidden_features=_D,
    num_encoder_layers=2,
    num_decoder_layers=1,
    k_neighbors=_L,
    dropout_rate=0.1,
    key=jax.random.PRNGKey(0),
  )


def _structure() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  coords = jax.random.normal(jax.random.PRNGKey(7), (_L, 4, 3))
  mask = jnp.ones((_L,))
  residue_index = jnp.arange(_L)
  chain_index = jnp.zeros((_L,), dtype=jnp.int32)
  return coords, mask, residue_index, chain_index


def _encode_and_logits(
  model: Aminx,
  key: jax.Array,
  *,
  inference: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  coords, mask, residue_index, chain_index = _structure()
  nodes, edges, neighbors = model(
    coords,
    mask,
    residue_index,
    chain_index,
    prng_key=key,
    backbone_noise=0.0,
    inference=inference,
  )
  decoded = model.decoder(
    nodes,
    edges,
    neighbors,
    mask,
    inference=inference,
    key=key,
  )
  logits = jax.vmap(model.w_out)(decoded)
  return nodes, edges, logits


def test_inference_true_is_key_invariant() -> None:
  """Fresh Aminx, inference=True: encoder features and logits ignore the PRNG key."""
  model = _model()
  nodes_1, edges_1, logits_1 = _encode_and_logits(model, jax.random.PRNGKey(1), inference=True)
  nodes_2, edges_2, logits_2 = _encode_and_logits(model, jax.random.PRNGKey(2), inference=True)
  assert jnp.array_equal(nodes_1, nodes_2)
  assert jnp.array_equal(edges_1, edges_2)
  assert jnp.array_equal(logits_1, logits_2)


def test_inference_false_depends_on_key() -> None:
  """Negative control: inference=False keeps encoder dropout, so keys differ."""
  model = _model()
  nodes_1, _, _ = _encode_and_logits(model, jax.random.PRNGKey(1), inference=False)
  nodes_2, _, _ = _encode_and_logits(model, jax.random.PRNGKey(2), inference=False)
  assert not jnp.array_equal(nodes_1, nodes_2)


def test_training_path_applies_dropout() -> None:
  """Training encode passes ``inference=False``, so encoder features still depend on the key.

  Same arguments as ``train_step``'s ``single_forward`` encode: zero backbone noise and no
  structure mapping. At zero noise the features are deterministic, so a difference is dropout.
  """
  model = _model()
  coords, mask, residue_index, chain_index = _structure()

  def encode(key: jax.Array) -> jax.Array:
    nodes, _, _ = model(
      coords,
      mask,
      residue_index,
      chain_index,
      backbone_noise=jnp.array(0.0),
      structure_mapping=None,
      initial_node_features=None,
      prng_key=key,
      inference=False,
    )
    return nodes

  nodes_1 = encode(jax.random.PRNGKey(1))
  nodes_2 = encode(jax.random.PRNGKey(2))
  assert not jnp.array_equal(nodes_1, nodes_2)
