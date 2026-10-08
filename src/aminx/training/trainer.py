"""Main training loop for Aminx."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import tqdm
from proxide.ops.dataset import create_protein_dataset
from xtrax.training.optim import adamw_with_schedule, make_optimizer, no_bias_wd_mask
from xtrax.training.types import ResumableState

from aminx.io.weights import load_model
from aminx.run.resources import proxide_dataset_resource_kwargs
from aminx.training.checkpoint import (
  get_checkpoint_manager,
  load_checkpoint,
  save_checkpoint,
)
from aminx.training.diffusion import NoiseSchedule
from aminx.training.losses import (
  cross_entropy_loss,
  perplexity,
  sequence_recovery_accuracy,
)
from aminx.training.metrics import (
  EvaluationMetrics,
  TrainingMetrics,
  compute_grad_norm,
)
from aminx.utils.aa_convert import training_labels

if TYPE_CHECKING:
  from chex import ArrayTree

  from aminx.model.diffusion_mpnn import DiffusionAminx
  from aminx.model.mpnn import Aminx
  from aminx.training.specs import TrainingSpecification
  from aminx.types.arrays import BackboneCoordinates, Logits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# See `.agents/TECHNICAL_DEBT.md` §1 (mixed-precision checkpoints), §5 (accum_steps validation).
# Precision policy: read ``spec.run_spec.precision.compute`` (composed RunSpec). ``TrainingSpecification.precision``
# remains the user-facing dataclass field and stays in sync when the spec is constructed; the trainer does not
# consult it directly so RunSpec stays the single routing point for this PR.


def _training_precision(spec: TrainingSpecification) -> str:
  """Resolve training compute-precision label from the composed :class:`~aminx.run.spec.RunSpec`."""
  return spec.run_spec.precision.compute


def get_compute_dtype(precision: str) -> jnp.dtype:
  """Get JAX dtype from precision string.

  Args:
      precision: One of "fp32", "fp16", "bf16"

  Returns:
      JAX dtype

  """
  if precision == "bf16":
    return jnp.bfloat16
  if precision == "fp16":
    return jnp.float16
  return jnp.float32


def _decay_all_wd_mask(params: ArrayTree) -> ArrayTree:
  """Decay every inexact parameter. Non-inexact leaves are left alone."""
  return jax.tree.map(eqx.is_inexact_array, params)


def _weight_decay_mask(spec: TrainingSpecification) -> Callable[[ArrayTree], ArrayTree]:
  """Resolve ``spec.weight_decay_mask`` once for both optimizer paths."""
  if spec.weight_decay_mask == "no_bias":
    return no_bias_wd_mask
  return _decay_all_wd_mask


def _optimizer_total_steps(spec: TrainingSpecification) -> int:
  return spec.total_steps or (spec.num_epochs * 1000)


def create_optimizer(
  spec: TrainingSpecification,
) -> optax.GradientTransformation:
  """Create AdamW optimizer with learning rate schedule using xtrax.

  Args:
      spec: Training specification

  Returns:
      An optax GradientTransformation combining warmup-cosine decay schedule,
      AdamW, and optional gradient clipping.

  """
  total_steps = _optimizer_total_steps(spec)
  wd_mask = _weight_decay_mask(spec)

  if spec.warmup_steps > 0:
    optimizer = adamw_with_schedule(
      peak_lr=spec.learning_rate,
      warmup_steps=spec.warmup_steps,
      total_steps=total_steps,
      weight_decay=spec.weight_decay,
      clip_norm=spec.gradient_clip,
      wd_mask=wd_mask,
    )
  else:
    # No warmup: constant learning rate. Same decay mask as the warmup path.
    optimizer = make_optimizer(
      optax.adamw(
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        mask=wd_mask,
      ),
      clip_norm=spec.gradient_clip,
    )

  return optimizer


def _learning_rate_schedule(spec: TrainingSpecification) -> Callable[[int], jax.Array]:
  """Learning-rate value reported in training metrics.

  The warmup path matches ``xtrax.training.optim.adamw_with_schedule``: cosine
  warmup from 0 to ``learning_rate`` and back to 0 over ``total_steps``. With
  ``warmup_steps == 0`` the optimizer is constant ``spec.learning_rate``.
  """
  if spec.warmup_steps > 0:
    return optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=spec.learning_rate,
      warmup_steps=spec.warmup_steps,
      decay_steps=_optimizer_total_steps(spec),
      end_value=0.0,
    )
  return optax.constant_schedule(spec.learning_rate)


@dataclass
class TrainingResult:
  """Container for results returned by :func:`train`.

  Attributes:
    final_model: The trained model instance.
    final_step: The final training step index.
    checkpoint_dir: Path to the checkpoint directory used.

  """

  final_model: Aminx
  final_step: int
  checkpoint_dir: str | Path


def _assemble_resumable_state(
  *,
  model: eqx.Module,
  opt_state: ArrayTree | None,
  random_seed: int,
  restored: ResumableState | None,
) -> ResumableState:
  """Build the state a run starts from.

  Fresh runs use ``PRNGKey(random_seed)``, step 0, and ``extras["epoch"] == 0``.
  Resume keeps the checkpoint step, PRNG key, and extras (including epoch).
  ``model`` and ``opt_state`` are the caller's (possibly cast) copies so the
  restored key is not paired with a pre-cast template.
  """
  if restored is None:
    return ResumableState(
      step=jnp.int32(0),
      key=jax.random.PRNGKey(random_seed),
      model=model,
      opt_state=opt_state,
      extras={"epoch": jnp.int32(0)},
    )
  return ResumableState(
    step=restored.step,
    key=restored.key,
    model=model,
    opt_state=opt_state,
    extras=restored.extras,
  )


def _with_epoch(state: ResumableState, epoch: int) -> ResumableState:
  """Return ``state`` with ``extras["epoch"]`` set to the epoch being trained."""
  extras = dict(state.extras)
  extras["epoch"] = jnp.int32(epoch)
  return ResumableState(
    step=state.step,
    key=state.key,
    model=state.model,
    opt_state=state.opt_state,
    extras=extras,
  )


def _init_checkpoint_and_model(
  spec: TrainingSpecification,
) -> tuple[ResumableState, ocp.CheckpointManager, ocp.CheckpointManager]:
  """Initialize or restore model, optimizer state and checkpoint manager.

  Applies physics encoder surgery if use_physics_features is enabled.

  Returns (resumable_state, checkpoint_manager, permanent_manager).
  """
  checkpoint_dir = Path(spec.checkpoint_dir)
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_manager = get_checkpoint_manager(
    checkpoint_dir,
    max_to_keep=spec.keep_last_n_checkpoints,
  )

  # Permanent checkpoint manager for epoch-pinned saves (keeps all)
  permanent_checkpoint_dir = checkpoint_dir / "kept"
  permanent_checkpoint_dir.mkdir(parents=True, exist_ok=True)
  permanent_manager = get_checkpoint_manager(
    permanent_checkpoint_dir,
    max_to_keep=None,
  )

  opt_state: ArrayTree | None = None
  compute_dtype = get_compute_dtype(_training_precision(spec))
  restored_state: ResumableState | None = None

  if spec.resume_from_checkpoint:
    model_template = load_model(
      spec.model_version,  # type: ignore[invalid-argument-type]
      spec.model_weights,  # type: ignore[invalid-argument-type]
      use_electrostatics=spec.use_electrostatics,
      use_vdw=spec.use_vdw,
    )

    # Compute abstract opt state for restoration
    optimizer_obj = create_optimizer(spec)
    abstract_opt_state = jax.eval_shape(
      optimizer_obj.init,
      eqx.filter(model_template, eqx.is_inexact_array),
    )

    # Template must match what ``_assemble_resumable_state`` / ``_with_epoch`` save:
    # extras carries an int32 epoch leaf. Checkpoints written with extras={} (before
    # that leaf existed) do not restore.
    state_template = ResumableState(
      step=jnp.int32(0),
      key=jax.random.PRNGKey(0),
      model=model_template,
      opt_state=abstract_opt_state,
      extras={"epoch": jnp.int32(0)},
    )
    restored_state = load_checkpoint(checkpoint_manager, state_template, step=None)
    model = restored_state.model
    opt_state = restored_state.opt_state
    logger.info("Resumed from checkpoint at step %d", int(restored_state.step))
  else:
    model = load_model(
      spec.model_version,  # type: ignore[invalid-argument-type]
      spec.model_weights,  # type: ignore[invalid-argument-type]
      use_electrostatics=spec.use_electrostatics,
      use_vdw=spec.use_vdw,
    )
    optimizer_obj = create_optimizer(spec)
    opt_state = optimizer_obj.init(eqx.filter(model, eqx.is_inexact_array))  # type: ignore[invalid-assignment]

  # Cast model to target precision
  if compute_dtype != jnp.float32:
    logger.info("Casting model parameters to %s", compute_dtype)

    def _cast_fn(x: jax.Array) -> jax.Array:
      return x.astype(compute_dtype) if eqx.is_inexact_array(x) else x

    model = jax.tree_util.tree_map(_cast_fn, model)
    opt_state = jax.tree_util.tree_map(_cast_fn, opt_state)

  resumable_state = _assemble_resumable_state(
    model=model,
    opt_state=opt_state,
    random_seed=spec.random_seed,
    restored=restored_state,
  )

  return resumable_state, checkpoint_manager, permanent_manager


def _create_dataloaders(spec: TrainingSpecification) -> tuple[Any, Any]:
  """Create training and validation data loaders based on the spec."""
  ram_mb, workers, buf = proxide_dataset_resource_kwargs(spec, context="training")
  train_loader = create_protein_dataset(
    cast("Any", spec.inputs),
    batch_size=spec.batch_size,
    foldcomp_database=spec.foldcomp_database if not spec.use_preprocessed else None,
    use_electrostatics=spec.use_electrostatics,
    estat_noise=spec.estat_noise,
    estat_noise_mode=spec.estat_noise_mode,
    use_vdw=spec.use_vdw,
    vdw_noise=spec.vdw_noise,
    vdw_noise_mode=spec.vdw_noise_mode,
    use_preprocessed=spec.use_preprocessed,
    preprocessed_index_path=spec.preprocessed_index_path,
    split="train",
    max_length=spec.max_length,
    truncation_strategy=spec.truncation_strategy,
    ram_budget_mb=ram_mb,
    max_workers=workers,
    max_buffer_size=buf,
  )

  val_loader = None
  if spec.validation_data:
    val_use_preprocessed = spec.use_preprocessed
    val_inputs = spec.validation_data
    val_index_path = spec.validation_preprocessed_index_path

    if spec.validation_preprocessed_path is not None:
      val_use_preprocessed = True
      val_inputs = spec.validation_preprocessed_path
      if val_index_path is None:
        val_index_path = Path(val_inputs).with_suffix(".index.json")

    val_loader = create_protein_dataset(
      cast("Any", val_inputs),
      batch_size=spec.batch_size,
      foldcomp_database=spec.foldcomp_database if not val_use_preprocessed else None,
      use_preprocessed=val_use_preprocessed,
      use_electrostatics=spec.use_electrostatics,
      estat_noise=spec.estat_noise,
      estat_noise_mode=spec.estat_noise_mode,
      use_vdw=spec.use_vdw,
      vdw_noise=spec.vdw_noise,
      vdw_noise_mode=spec.vdw_noise_mode,
      preprocessed_index_path=val_index_path,
      split="valid",
      max_length=spec.max_length,
      truncation_strategy=spec.truncation_strategy,
      ram_budget_mb=ram_mb,
      max_workers=workers,
      max_buffer_size=buf,
    )

  return train_loader, val_loader


def setup_mixed_precision(precision: str) -> None:
  """Configure JAX mixed precision policy.

  Args:
      precision: One of "fp32", "fp16", "bf16"

  """
  if precision == "fp16":
    logger.info("Using FP16 mixed precision (manual casting required)")
  elif precision == "bf16":
    logger.info("Using BF16 mixed precision")
  else:
    logger.info("Using FP32 (full precision)")


def _accumulate_value_and_grad(
  model: eqx.Module,
  micro_loss_fn: Callable[[eqx.Module, ArrayTree], tuple[jax.Array, jax.Array]],
  micro_batches: ArrayTree,
  *,
  batch_size: int,
  accum_steps: int,
) -> tuple[jax.Array, jax.Array, ArrayTree]:
  """Average a loss and its gradients over ``accum_steps`` micro-batches.

  The scan carry is ``(loss_sum, grads_sum)`` with ``grads_sum`` zeros of the
  filtered inexact parameters. Per-micro-batch logits are scan outputs, stacked
  and reshaped to the full batch. The returned loss and gradients are the sums
  divided by ``accum_steps``, which matches one step on the full batch when
  every micro-batch has the same size.

  ``accum_steps == 1`` is not this helper's job; callers keep that path as a
  single ``filter_value_and_grad``.
  """
  if batch_size % accum_steps != 0:
    msg = f"batch_size ({batch_size}) must be divisible by accum_steps ({accum_steps})"
    raise ValueError(msg)

  params = eqx.filter(model, eqx.is_inexact_array)
  grads_sum = jax.tree.map(jnp.zeros_like, params)
  loss_sum = jnp.zeros(())

  def _body(
    carry: tuple[jax.Array, ArrayTree],
    micro: ArrayTree,
  ) -> tuple[tuple[jax.Array, ArrayTree], jax.Array]:
    running_loss, running_grads = carry

    def _loss_for_grad(m: eqx.Module) -> tuple[jax.Array, jax.Array]:
      return micro_loss_fn(m, micro)

    (loss, logits), grads = eqx.filter_value_and_grad(_loss_for_grad, has_aux=True)(model)
    running_grads = jax.tree.map(lambda a, b: a + b, running_grads, grads)
    return (running_loss + loss, running_grads), logits

  (loss_sum, grads_sum), logits_stacked = jax.lax.scan(
    _body,
    (loss_sum, grads_sum),
    micro_batches,
  )
  loss = loss_sum / accum_steps
  grads = jax.tree.map(lambda g: g / accum_steps, grads_sum)
  logits = jnp.reshape(
    logits_stacked,
    (logits_stacked.shape[0] * logits_stacked.shape[1], *logits_stacked.shape[2:]),
  )
  return loss, logits, grads


def train_step(  # noqa: PLR0915
  model: Aminx,
  opt_state: optax.OptState,
  optimizer: optax.GradientTransformation,
  coordinates: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  sequence: jax.Array,
  prng_key: jax.Array,
  label_smoothing: float,
  current_step: int,
  physics_features: jax.Array | None = None,
  backbone_noise_std: float = 0.0,
  mask_strategy: str = "random_order",
  mask_prob: float = 0.15,
  training_mode: str = "autoregressive",
  noise_schedule: NoiseSchedule | None = None,
  accum_steps: int = 1,
  compute_dtype: jnp.dtype = jnp.float32,  # noqa: ARG001
  rbf_features: jax.Array | None = None,
  neighbor_indices: jax.Array | None = None,
  lr_schedule: Callable[[int], jax.Array] | None = None,
) -> tuple[Aminx, optax.OptState, TrainingMetrics]:
  """Single training step.

  Args:
      model: Aminx model
      opt_state: Optimizer state
      optimizer: Optax optimizer with integrated learning rate schedule (from xtrax)
      coordinates: Backbone coordinates
      mask: Valid residue mask
      residue_index: Residue indices
      chain_index: Chain indices
      sequence: Target sequence (integer labels), **MPNN**-ordered (``ACDEFGHIKLMNPQRSTVWYX``) --
          the model's token space. A loader batch's ``aatype`` is AF-ordered and must be passed
          through ``training_labels`` first (issue #109); no range check can detect the mix-up.
      prng_key: PRNG key
      label_smoothing: Label smoothing factor
      current_step: Current training step (used for learning rate scheduling)
      physics_features: Optional physics features (if used)
      backbone_noise_std: Standard deviation of Gaussian noise added to backbone coordinates
      mask_strategy: Strategy for autoregressive masking ("random_order" or "bert")
      mask_prob: Probability of masking a token if using "bert" strategy
      training_mode: "autoregressive" or "diffusion"
      noise_schedule: Noise schedule for diffusion training
      accum_steps: Number of gradient accumulation steps (effective batch_size = batch_size).
      compute_dtype: JAX dtype for computation (e.g., jnp.bfloat16).
      rbf_features: Optional precomputed RBF features from proxide (N, K, 400).
          When provided, RBF computation is skipped in the feature module.
      neighbor_indices: Optional precomputed neighbor indices from proxide (N, K).
          Must be provided if rbf_features is provided.
      lr_schedule: Maps the training step to the optimizer learning rate. When
          omitted, the learning-rate metric is left unset rather than filled with
          a placeholder.

  Returns:
      Tuple of (updated_model, updated_opt_state, metrics)

  """

  def single_forward(
    m: Aminx,
    coords: BackboneCoordinates,
    mask: jax.Array,
    res_idx: jax.Array,
    chain_idx: jax.Array,
    seq: jax.Array,
    key: jax.Array,
    phys_feat: jax.Array | None = None,
    rbf_feats: jax.Array | None = None,  # noqa: ARG001
    neighbor_idx: jax.Array | None = None,  # noqa: ARG001
  ) -> Logits:
    """Forward pass for a single protein."""
    key, subkey = jax.random.split(key)

    n_nodes = mask.shape[0]
    if mask_strategy == "random_order":
      decoding_order = jax.random.permutation(subkey, jnp.arange(n_nodes))
      ranks = jnp.argsort(decoding_order)
      ar_mask = ranks[None, :] < ranks[:, None]
    elif mask_strategy == "bert":
      mask_prob_mask = jax.random.bernoulli(subkey, mask_prob, shape=(n_nodes,))
      can_see = 1.0 - mask_prob_mask
      ar_mask = jnp.tile(can_see[None, :], (n_nodes, 1))
    else:
      ar_mask = jnp.ones((n_nodes, n_nodes))

    one_hot_seq = jax.nn.one_hot(seq, 21)

    # 1. Encode
    coords = jnp.asarray(coords)
    node_features, edge_features, edge_indices = m(
      coords,
      mask,
      res_idx,
      chain_idx,
      backbone_noise=jnp.array(backbone_noise_std),
      structure_mapping=None,  # training is usually single-state
      initial_node_features=phys_feat,
      prng_key=key,
      inference=False,
    )

    if training_mode == "diffusion":
      if noise_schedule is None:
        msg = "noise_schedule required for diffusion training"
        raise ValueError(msg)

      t = jax.random.randint(subkey, (), 0, noise_schedule.num_steps)
      noise = jax.random.normal(subkey, one_hot_seq.shape)
      noisy_seq, _ = noise_schedule.sample_forward(one_hot_seq, t, noise)

      diff_model = cast("DiffusionAminx", m)
      # Inject timestep embedding
      t_embed = diff_model.t_embed_sin(t)
      t_embed = diff_model.t_embed_mlp(t_embed)
      t_embed = t_embed[None, :]  # [1, C]
      node_features = node_features + t_embed

      # Decode
      decoded = diff_model.decoder.call_conditional(
        node_features,
        edge_features,
        edge_indices,
        mask,
        ar_mask,
        noisy_seq,
        diff_model.w_s_embed.weight,
        key=key,
        inference=False,
      )
      return jax.vmap(diff_model.w_out)(decoded)

    # 2. Decode (Conditional MPNN)
    decoded = m.decoder.call_conditional(
      node_features,
      edge_features,
      edge_indices,
      mask,
      ar_mask,
      one_hot_seq,
      m.w_s_embed.weight,
      key=key,
      inference=False,
    )
    return jax.vmap(m.w_out)(decoded)

  def batch_loss(logits: Logits, seq: jax.Array, msk: jax.Array) -> jax.Array:
    return cross_entropy_loss(logits, seq, msk, label_smoothing)

  batch_size = coordinates.shape[0]

  def loss_fn(model: Aminx) -> tuple[jax.Array, jax.Array]:
    """Compute loss for current batch."""
    batch_keys = jax.random.split(prng_key, batch_size)

    logits_batch = jax.vmap(partial(single_forward, model))(
      coordinates,
      mask,
      residue_index,
      chain_index,
      sequence,
      batch_keys,
      physics_features,
      rbf_features,
      neighbor_indices,
    )

    losses = jax.vmap(batch_loss)(logits_batch, sequence, mask)
    loss = jnp.mean(losses)

    return loss, logits_batch

  # Gradient accumulation support. accum_steps == 1 stays a single value_and_grad.
  if accum_steps > 1:
    if batch_size % accum_steps != 0:
      msg = f"batch_size ({batch_size}) must be divisible by accum_steps ({accum_steps})"
      raise ValueError(msg)
    # Reshape features to [accum_steps, batch_size // accum_steps, ...]
    micro_batch_size = batch_size // accum_steps

    def _reshape(x: jax.Array) -> jax.Array:
      return x.reshape((accum_steps, micro_batch_size, *x.shape[1:]))

    def _reshape_opt(x: jax.Array | None) -> jax.Array | None:
      return _reshape(x) if x is not None else None

    # Reshape all inputs
    coords_reshaped = _reshape(coordinates)
    mask_reshaped = _reshape(mask)
    res_idx_reshaped = _reshape(residue_index)
    chain_idx_reshaped = _reshape(chain_index)
    seq_reshaped = _reshape(sequence)
    phys_reshaped = _reshape_opt(physics_features)
    rbf_reshaped = _reshape_opt(rbf_features)
    neighbor_reshaped = _reshape_opt(neighbor_indices)

    # Split PRNG key for each micro-batch
    accum_keys = jax.random.split(prng_key, accum_steps)

    def micro_loss_fn(
      m_model: Aminx,
      inputs: tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array | None,
        jax.Array | None,
        jax.Array | None,
      ],
    ) -> tuple[jax.Array, Logits]:
      (c, m, ri, ci, s, k, p, rbf, nb) = inputs
      micro_keys = jax.random.split(k, micro_batch_size)
      micro_logits = jax.vmap(partial(single_forward, m_model))(
        c,
        m,
        ri,
        ci,
        s,
        micro_keys,
        p,
        rbf,
        nb,
      )
      micro_loss_val = jnp.mean(jax.vmap(batch_loss)(micro_logits, s, m))
      return micro_loss_val, micro_logits

    loss, logits_batch, grads = _accumulate_value_and_grad(
      model,
      micro_loss_fn,
      (
        coords_reshaped,
        mask_reshaped,
        res_idx_reshaped,
        chain_idx_reshaped,
        seq_reshaped,
        accum_keys,
        phys_reshaped,
        rbf_reshaped,
        neighbor_reshaped,
      ),
      batch_size=batch_size,
      accum_steps=accum_steps,
    )
  else:
    (loss, logits_batch), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

  def batch_metrics(logits: Logits, seq: jax.Array, msk: jax.Array) -> tuple[jax.Array, jax.Array]:
    acc = sequence_recovery_accuracy(logits, seq, msk)
    ppl = perplexity(logits, seq, msk)
    return acc, ppl

  accuracies, perplexities = jax.vmap(batch_metrics)(logits_batch, sequence, mask)
  accuracy = jnp.mean(accuracies)
  ppl = jnp.mean(perplexities)
  grad_norm = compute_grad_norm(grads)
  # Same schedule the optimizer applies (constant, or xtrax warmup-cosine). Absent
  # schedule: omit the metric. Do not log a stand-in value.
  current_lr = None if lr_schedule is None else lr_schedule(current_step)

  params = eqx.filter(model, eqx.is_inexact_array)
  updates, new_opt_state = optimizer.update(grads, opt_state, params)
  new_model = eqx.apply_updates(model, updates)

  metrics = TrainingMetrics(
    loss=loss,
    accuracy=accuracy,
    perplexity=ppl,
    learning_rate=current_lr,
    grad_norm=grad_norm,
  )

  return new_model, new_opt_state, metrics


def eval_step(
  model: Aminx,
  coordinates: jax.Array,  # (batch_size, seq_len, 4, 3)
  mask: jax.Array,  # (batch_size, seq_len)
  residue_index: jax.Array,  # (batch_size, seq_len)
  chain_index: jax.Array,  # (batch_size, seq_len)
  sequence: jax.Array,  # (batch_size, seq_len)
  prng_key: jax.Array,
  physics_features: jax.Array | None = None,
  training_mode: str = "autoregressive",
  noise_schedule: NoiseSchedule | None = None,
) -> EvaluationMetrics:
  """Single evaluation step with batching.

  Args:
      model: Aminx model
      coordinates: Backbone coordinates (batched)
      mask: Valid residue mask (batched)
      residue_index: Residue indices (batched)
      chain_index: Chain indices (batched)
      sequence: Target sequence (batched), **MPNN**-ordered (``ACDEFGHIKLMNPQRSTVWYX``). A
          loader batch's ``aatype`` is AF-ordered: pass it through ``training_labels`` first
          (issue #109).
      prng_key: PRNG key
      physics_features: Optional physics features (if used)
      training_mode: "autoregressive" or "diffusion"
      noise_schedule: Noise schedule for diffusion training

  Returns:
      Evaluation metrics

  """
  batch_size = coordinates.shape[0]
  batch_keys = jax.random.split(prng_key, batch_size)

  def single_forward(
    coords: jax.Array,
    msk: jax.Array,
    res_idx: jax.Array,
    chain_idx: jax.Array,
    seq: jax.Array,
    key: jax.Array,
    phys_feat: jax.Array | None = None,
  ) -> Logits:
    """Forward pass for a single protein."""
    inference_model = eqx.nn.inference_mode(model)

    # 1. Encode
    coords = jnp.asarray(coords)
    node_features, edge_features, edge_indices = inference_model(
      coords,
      msk,
      res_idx,
      chain_idx,
      backbone_noise=jnp.array(0.0),
      structure_mapping=None,
      initial_node_features=phys_feat,
      prng_key=key,
    )

    if training_mode == "diffusion":
      if noise_schedule is None:
        msg = "noise_schedule required for diffusion evaluation"
        raise ValueError(msg)

      t = jax.random.randint(key, (), 0, noise_schedule.num_steps)
      one_hot_seq = jax.nn.one_hot(seq, 21)
      noise = jax.random.normal(key, one_hot_seq.shape)
      noisy_seq, _ = noise_schedule.sample_forward(one_hot_seq, t, noise)

      diff_model = cast("DiffusionAminx", inference_model)
      # Inject timestep embedding
      t_embed = diff_model.t_embed_sin(t)
      t_embed = diff_model.t_embed_mlp(t_embed)
      t_embed = t_embed[None, :]
      node_features = node_features + t_embed

      # Decode
      decoded = diff_model.decoder.call_conditional(
        node_features,
        edge_features,
        edge_indices,
        msk,
        jnp.ones((msk.shape[0], msk.shape[0])),
        noisy_seq,
        diff_model.w_s_embed.weight,
        key=key,
        inference=True,
      )
      return jax.vmap(diff_model.w_out)(decoded)

    # 2. Decode (Unconditional/Conditional)
    decoded = inference_model.decoder.call_conditional(
      node_features,
      edge_features,
      edge_indices,
      msk,
      jnp.ones((msk.shape[0], msk.shape[0])),
      jax.nn.one_hot(seq, 21),
      inference_model.w_s_embed.weight,
      key=key,
      inference=True,
    )
    return jax.vmap(inference_model.w_out)(decoded)

  logits_batch = jax.vmap(single_forward)(
    coordinates,
    mask,
    residue_index,
    chain_index,
    sequence,
    batch_keys,
    physics_features,
  )

  def batch_metrics(
    logits: jax.Array,
    seq: jax.Array,
    msk: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    val_loss = cross_entropy_loss(logits, seq, msk, label_smoothing=0.0)
    val_accuracy = sequence_recovery_accuracy(logits, seq, msk)
    val_ppl = perplexity(logits, seq, msk)
    return val_loss, val_accuracy, val_ppl

  losses, accuracies, perplexities = jax.vmap(batch_metrics)(
    logits_batch,
    sequence,
    mask,
  )

  return EvaluationMetrics(
    val_loss=jnp.mean(losses),
    val_accuracy=jnp.mean(accuracies),
    val_perplexity=jnp.mean(perplexities),
  )


def train(spec: TrainingSpecification) -> TrainingResult:  # noqa: PLR0915
  """Train Aminx model.

  Args:
      spec: Training specification

  Returns:
      Dictionary with training results and final model

  Example:
      >>> spec = TrainingSpecification(
      ...     inputs="data/train/",
      ...     validation_data="data/val/",
      ...     batch_size=8,
      ...     num_epochs=10,
      ...     learning_rate=1e-4,
      ... )
      >>> results = train(spec)
      >>> print(f"Final validation accuracy: {results['final_val_accuracy']}")

  """
  setup_mixed_precision(_training_precision(spec))
  logger.info("Starting training with spec: %s", spec)

  optimizer = create_optimizer(spec)
  lr_schedule = _learning_rate_schedule(spec)

  resumable_state, checkpoint_manager, permanent_manager = _init_checkpoint_and_model(
    spec,
  )

  train_loader, val_loader = _create_dataloaders(spec)

  step = int(resumable_state.step)
  best_val_metric = float("inf")
  patience_counter = 0

  prng_key = resumable_state.key
  noise_schedule = None
  if spec.training_mode == "diffusion":
    noise_schedule = NoiseSchedule(
      num_steps=spec.diffusion_num_steps,
      beta_start=spec.diffusion_beta_start,
      beta_end=spec.diffusion_beta_end,
      schedule_type=spec.diffusion_schedule_type,
    )

  logger.info("Starting training loop...")

  # Epoch is persisted in extras["epoch"] instead of being derived from step.
  # The loader has no fixed steps-per-epoch (length can vary between epochs), so
  # step // steps_per_epoch would not land on the same epoch the loop was in.
  # The stored value is the epoch in progress; resume repeats that epoch from the
  # start of the loader and continues the step and PRNG key.
  start_epoch = int(jax.device_get(resumable_state.extras["epoch"]))
  early_stop = False

  for epoch in range(start_epoch, spec.num_epochs):
    resumable_state = _with_epoch(resumable_state, epoch)
    logger.info("Epoch %d/%d", epoch + 1, spec.num_epochs)
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{spec.num_epochs}")

    # JIT the step functions
    compute_dtype = get_compute_dtype(_training_precision(spec))
    filter_jitted_train_step = eqx.filter_jit(train_step)
    filter_jitted_eval_step = eqx.filter_jit(eval_step)

    for batch in pbar:
      prng_key, subkey = jax.random.split(prng_key)

      if isinstance(spec.backbone_noise, (float, int)):
        backbone_noise_std = float(spec.backbone_noise)
      else:
        backbone_noise_std = float(spec.backbone_noise[0])

      updated_model, updated_opt_state, train_metrics = filter_jitted_train_step(
        resumable_state.model,
        resumable_state.opt_state,
        optimizer,
        batch.coordinates,
        batch.mask,
        batch.residue_index,
        batch.chain_index,
        training_labels(batch.aatype),
        subkey,
        spec.label_smoothing,
        step,
        batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
        backbone_noise_std,
        spec.mask_strategy,
        spec.mask_prob,
        spec.training_mode,
        noise_schedule,
        spec.accum_steps,
        compute_dtype,
        lr_schedule=lr_schedule,
      )

      # Update resumable state
      resumable_state = ResumableState(
        step=jnp.int32(step + 1),
        key=prng_key,
        model=updated_model,
        opt_state=updated_opt_state,
        extras=resumable_state.extras,
      )

      step += 1
      # NOTE(io_callback): Logging-only scalar device_get; optional io_callback metrics sink later.
      loss_float = jax.device_get(train_metrics.loss).item()
      pbar.set_postfix({"loss": loss_float})

      if val_loader and step % spec.eval_every == 0:
        val_metrics_list = []
        for val_batch in val_loader:
          prng_key, subkey = jax.random.split(prng_key)

          val_metrics = filter_jitted_eval_step(
            cast("Aminx", resumable_state.model),
            val_batch.coordinates,
            val_batch.mask,
            val_batch.residue_index,
            val_batch.chain_index,
            training_labels(val_batch.aatype),
            subkey,
            val_batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
            spec.training_mode,
            noise_schedule,
          )
          val_metrics_list.append(val_metrics)

        avg_val_loss = jnp.mean(jnp.array([m.val_loss for m in val_metrics_list]))
        avg_val_acc = jnp.mean(jnp.array([m.val_accuracy for m in val_metrics_list]))

        val_loss_float = jax.device_get(avg_val_loss).item()
        val_acc_float = jax.device_get(avg_val_acc).item()

        logger.info(
          "Validation at step %d: val_loss=%.4f, val_acc=%.4f",
          step,
          val_loss_float,
          val_acc_float,
        )

        if spec.early_stopping_patience:
          current_metric = avg_val_loss  # Can switch based on spec.early_stopping_metric
          if current_metric < best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
          else:
            patience_counter += 1

          if patience_counter >= spec.early_stopping_patience:
            logger.info("Early stopping triggered at step %d", step)
            early_stop = True
            break

      if step % spec.checkpoint_every == 0:
        save_checkpoint(checkpoint_manager, resumable_state)

      # Persistent checkpointing
      if spec.save_at_epochs and (epoch + 1) in spec.save_at_epochs:
        logger.info("Saving persistent checkpoint for epoch %d", epoch + 1)
        save_checkpoint(permanent_manager, resumable_state)

    pbar.close()
    if early_stop:
      break

  # The in-loop save only runs when step % checkpoint_every == 0. Always persist
  # the final state, including after early stop. Skip when that step was just written.
  if checkpoint_manager.latest_step() != step:
    save_checkpoint(checkpoint_manager, resumable_state)

  logger.info("Training complete!")

  # Final Test Loop
  logger.info("Starting final test evaluation...")
  test_loader = None

  # Determine test data source
  test_inputs = spec.validation_data  # Default to validation data if no separate test set
  test_use_preprocessed = spec.use_preprocessed
  test_index_path = spec.validation_preprocessed_index_path

  # If we are using preprocessed data, we try to load the 'test' split from the same file
  # or a specific test file if one were added to spec
  if spec.use_preprocessed and spec.preprocessed_index_path:
    # If validation path is set, use that, otherwise fall back to training path
    test_inputs = spec.validation_preprocessed_path or spec.inputs

    test_index_path = spec.validation_preprocessed_index_path or spec.preprocessed_index_path

  try:
    t_ram, t_workers, t_buf = proxide_dataset_resource_kwargs(spec, context="training")
    test_loader = create_protein_dataset(
      cast("Any", test_inputs),
      batch_size=spec.batch_size,
      foldcomp_database=spec.foldcomp_database if not test_use_preprocessed else None,
      use_preprocessed=test_use_preprocessed,
      use_electrostatics=spec.use_electrostatics,
      estat_noise=spec.estat_noise,
      estat_noise_mode=spec.estat_noise_mode,
      use_vdw=spec.use_vdw,
      vdw_noise=spec.vdw_noise,
      vdw_noise_mode=spec.vdw_noise_mode,
      preprocessed_index_path=test_index_path,
      split="test",
      ram_budget_mb=t_ram,
      max_workers=t_workers,
      max_buffer_size=t_buf,
    )

    test_metrics_list = []
    for test_batch in tqdm.tqdm(test_loader, desc="Testing"):
      prng_key, subkey = jax.random.split(prng_key)

      test_metrics = eqx.filter_jit(eval_step)(
        cast("Aminx", resumable_state.model),
        test_batch.coordinates,
        test_batch.mask,
        test_batch.residue_index,
        test_batch.chain_index,
        training_labels(test_batch.aatype),
        subkey,
        test_batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
        spec.training_mode,
        noise_schedule,
      )
      test_metrics_list.append(test_metrics)

    if test_metrics_list:
      avg_test_loss = jnp.mean(jnp.array([m.val_loss for m in test_metrics_list]))
      avg_test_acc = jnp.mean(jnp.array([m.val_accuracy for m in test_metrics_list]))
      avg_test_ppl = jnp.mean(jnp.array([m.val_perplexity for m in test_metrics_list]))

      logger.info("=" * 40)
      logger.info("Final Test Results:")
      logger.info("  Loss: %.4f", jax.device_get(avg_test_loss).item())
      logger.info("  Accuracy: %.4f", jax.device_get(avg_test_acc).item())
      logger.info("  Perplexity: %.4f", jax.device_get(avg_test_ppl).item())
      logger.info("=" * 40)
    else:
      logger.warning("Test loader was empty. No test metrics computed.")

  except Exception:  # noqa: BLE001
    logger.warning(
      "Could not create test loader or run testing (possibly no 'test' split found).",
    )

  checkpoint_manager.close()
  permanent_manager.close()

  return TrainingResult(final_model=cast("Aminx", resumable_state.model), final_step=step, checkpoint_dir=spec.checkpoint_dir)
