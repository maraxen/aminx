"""Unconditional scoring kernel using InferenceBundle."""

from jaxtyping import PRNGKeyArray

from aminx.inference.decode.factory import make_decode_fn
from aminx.inference.decode.mode import UnconditionalMode
from aminx.inference.encode import make_encode_fn
from aminx.tiling.strategy import Vmap
from aminx.types.arrays import Logits
from aminx.types.bundles import InferenceBundle
from aminx.types.configs import InferenceConfig
from aminx.types.protocols import ModelProtocol
from aminx.types.stages import StageSet


def kernel(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
) -> Logits:
  """Compute unconditional logits."""
  import jax

  k_enc, k_dec = jax.random.split(prng_key)

  # Encode using vmap strategy (parallel over S states)
  encode_fn = make_encode_fn(model, use_rolling_state=False)
  enc = encode_fn(bundle, k_enc, config)

  # Construct UnconditionalDecode with vmap strategy over state axis
  decode_fn = make_decode_fn(model, mode=UnconditionalMode(), strategy=Vmap())

  # Call the decode function to get logits
  return decode_fn(k_dec, enc, bundle, config, stage_set)
