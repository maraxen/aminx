"""Test script for diffusion training loop."""

import logging
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from proxide.ops.dataset import create_protein_dataset

from aminx.model.diffusion_mpnn import DiffusionAminx
from aminx.training.diffusion import NoiseSchedule
from aminx.training.train_diffusion import train_step
from aminx.utils.aa_convert import training_labels

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def test_diffusion_training() -> None:
  """Run a test of the diffusion training loop."""
  logger.info("Initializing model and optimizer...")
  key = jax.random.PRNGKey(0)
  model_key, train_key = jax.random.split(key)

  model = DiffusionAminx(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=30,
    key=model_key,
  )

  optimizer = optax.adam(1e-4)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

  noise_schedule = NoiseSchedule(num_steps=100)
  logger.info("Schedule shape: %s", noise_schedule.sqrt_alphas_cumprod.shape)
  lr_schedule = optax.constant_schedule(1e-4)

  logger.info("Loading data...")
  # Use the same sample data as before
  ds = create_protein_dataset(
    inputs="src/aminx/training/data/pdb_sample.array_record",
    batch_size=2,
    use_preprocessed=True,
    preprocessed_index_path="src/aminx/training/data/pdb_sample.index.json",
  )

  iterator = iter(ds)

  logger.info("Starting training loop...")
  max_steps = 5
  for i, batch in enumerate(iterator):
    if i >= max_steps:
      break

    logger.info("Step %d", i + 1)

    # Prepare inputs
    coordinates = jnp.array(batch.coordinates)
    mask = jnp.array(batch.mask)
    residue_index = jnp.array(batch.residue_index)
    chain_index = jnp.array(batch.chain_index)
    sequence = training_labels(batch.aatype)  # AF -> MPNN (issue #109)
    physics_features = jnp.array(batch.physics_features)

    step_key = jax.random.fold_in(train_key, i)

    model, opt_state, metrics = train_step(
      model,
      opt_state,
      optimizer,
      coordinates,
      mask,
      residue_index,
      chain_index,
      sequence,
      step_key,
      noise_schedule,
      lr_schedule,
      i,
      physics_features,
      physics_noise_scale=0.5,
    )

    logger.info("Loss: %.4f, Acc: %.4f", metrics.loss, metrics.accuracy)

  logger.info("Diffusion training test passed!")


if __name__ == "__main__":
  test_diffusion_training()
