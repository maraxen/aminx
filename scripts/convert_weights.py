"""Weight conversion utility: PyTorch LigandMPNN -> JAX Aminx.

This script converts weights from PyTorch LigandMPNN format to JAX Equinox format.

Usage:
    python convert_weights.py --input model.pt --output model.eqx
"""

import argparse
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.model.ligand_features import ProteinFeaturesLigand
from aminx.model.packer import Packer

# Must match dauparas/LigandMPNN `model_utils.ProteinMPNN` (`ligand_mpnn` branch).
NUM_LIGAND_CONTEXT_LAYERS = 2


def convert_linear_layer(
    pt_weight: np.ndarray,
    pt_bias: np.ndarray | None,
    jax_layer: eqx.nn.Linear,
) -> eqx.nn.Linear:
    """Convert a PyTorch Linear layer to JAX.
    
    PyTorch: weight shape [out, in], computes x @ W.T + b
    JAX eqx.nn.Linear: weight shape [in, out], computes x @ W + b
    """
    # Use same weight shape: [out, in]
    # A bias present on exactly one side is a skeleton/checkpoint disagreement, and both
    # directions used to be silent: a checkpoint bias with no slot was DROPPED (#162), and a
    # skeleton bias with no checkpoint value KEPT ITS RANDOM INIT and shipped it (#163).
    # equinox defaults to use_bias=True while several reference layers are bias=False.
    if (pt_bias is None) != (jax_layer.bias is None):
        side = "checkpoint has a bias the skeleton has no slot for (trained values would be dropped)" \
            if pt_bias is not None else \
            "skeleton has a bias the checkpoint does not (random init would ship as if trained)"
        raise ValueError(f"bias mismatch converting Linear{tuple(pt_weight.shape)}: {side}")

    jax_weight = jnp.array(pt_weight)

    # Create new layer with converted weights using eqx.tree_at
    new_layer = eqx.tree_at(lambda l: l.weight, jax_layer, jax_weight)
    if pt_bias is not None:
        new_layer = eqx.tree_at(lambda l: l.bias, new_layer, jnp.array(pt_bias))

    return new_layer


def convert_mlp(
    pt_layers: list[tuple[np.ndarray, np.ndarray]],
    jax_mlp: eqx.nn.MLP,
) -> eqx.nn.MLP:
    """Convert PyTorch MLP layers to JAX MLP.
    
    Args:
        pt_layers: List of (weight, bias) tuples for each linear layer.
        jax_mlp: Target JAX MLP to populate.
    
    Returns:
        New MLP with converted weights.
    """
    new_layers = list(jax_mlp.layers)

    pt_idx = 0
    for i, layer in enumerate(new_layers):
        if isinstance(layer, eqx.nn.Linear):
            pt_weight, pt_bias = pt_layers[pt_idx]
            new_layers[i] = convert_linear_layer(pt_weight, pt_bias, layer)
            pt_idx += 1

    return eqx.tree_at(lambda m: m.layers, jax_mlp, tuple(new_layers))


def convert_layer_norm(
    pt_weight: np.ndarray,
    pt_bias: np.ndarray,
    jax_norm: eqx.nn.LayerNorm,
) -> eqx.nn.LayerNorm:
    """Convert PyTorch LayerNorm to JAX."""
    new_norm = eqx.tree_at(lambda n: n.weight, jax_norm, jnp.array(pt_weight))
    new_norm = eqx.tree_at(lambda n: n.bias, new_norm, jnp.array(pt_bias))
    return new_norm


def convert_embedding(
    pt_weight: np.ndarray,
    jax_embed: eqx.nn.Embedding,
) -> eqx.nn.Embedding:
    """Convert PyTorch Embedding to JAX."""
    return eqx.tree_at(lambda e: e.weight, jax_embed, jnp.array(pt_weight))


def resolve_ligand_side_chain_context(
    mode: str,
    *,
    checkpoint_payload: dict[str, Any] | None = None,
    input_path: str,
) -> bool:
    """Resolve side-chain context usage for LigandMPNN conversion."""
    if mode == "on":
        return True
    if mode == "off":
        return False

    if checkpoint_payload is not None and "ligand_mpnn_use_side_chain_context" in checkpoint_payload:
        return bool(checkpoint_payload["ligand_mpnn_use_side_chain_context"])

    filename = Path(input_path).name.lower()
    return "side_chain_context" in filename or "ligandmpnn_sc" in filename


def get_pytorch_encoder_layer_weights(state_dict: dict, layer_idx: int) -> dict:
    """Extract weights for a single encoder layer."""
    prefix = f"encoder_layers.{layer_idx}."
    return {
        "W1": (state_dict[prefix + "W1.weight"], state_dict[prefix + "W1.bias"]),
        "W2": (state_dict[prefix + "W2.weight"], state_dict[prefix + "W2.bias"]),
        "W3": (state_dict[prefix + "W3.weight"], state_dict[prefix + "W3.bias"]),
        "W11": (state_dict[prefix + "W11.weight"], state_dict[prefix + "W11.bias"]),
        "W12": (state_dict[prefix + "W12.weight"], state_dict[prefix + "W12.bias"]),
        "W13": (state_dict[prefix + "W13.weight"], state_dict[prefix + "W13.bias"]),
        "norm1": (state_dict[prefix + "norm1.weight"], state_dict[prefix + "norm1.bias"]),
        "norm2": (state_dict[prefix + "norm2.weight"], state_dict[prefix + "norm2.bias"]),
        "norm3": (state_dict[prefix + "norm3.weight"], state_dict[prefix + "norm3.bias"]),
        "dense.W_in": (state_dict[prefix + "dense.W_in.weight"], state_dict[prefix + "dense.W_in.bias"]),
        "dense.W_out": (state_dict[prefix + "dense.W_out.weight"], state_dict[prefix + "dense.W_out.bias"]),
    }


def get_pytorch_decoder_layer_weights(state_dict: dict, layer_idx: int, prefix_base: str = "decoder_layers") -> dict:
    """Extract weights for a single decoder layer."""
    prefix = f"{prefix_base}.{layer_idx}."
    return {
        "W1": (state_dict[prefix + "W1.weight"], state_dict[prefix + "W1.bias"]),
        "W2": (state_dict[prefix + "W2.weight"], state_dict[prefix + "W2.bias"]),
        "W3": (state_dict[prefix + "W3.weight"], state_dict[prefix + "W3.bias"]),
        "norm1": (state_dict[prefix + "norm1.weight"], state_dict[prefix + "norm1.bias"]),
        "norm2": (state_dict[prefix + "norm2.weight"], state_dict[prefix + "norm2.bias"]),
        "dense.W_in": (state_dict[prefix + "dense.W_in.weight"], state_dict[prefix + "dense.W_in.bias"]),
        "dense.W_out": (state_dict[prefix + "dense.W_out.weight"], state_dict[prefix + "dense.W_out.bias"]),
    }


def convert_encoder_layer(
    pt_weights: dict,
    jax_layer,  # EncoderLayer
):
    """Convert PyTorch EncLayer weights to JAX EncoderLayer."""
    # edge_message_mlp: W1 -> W2 -> W3
    jax_layer = eqx.tree_at(
        lambda l: l.edge_message_mlp,
        jax_layer,
        convert_mlp(
            [pt_weights["W1"], pt_weights["W2"], pt_weights["W3"]],
            jax_layer.edge_message_mlp,
        ),
    )

    # edge_update_mlp: W11 -> W12 -> W13
    jax_layer = eqx.tree_at(
        lambda l: l.edge_update_mlp,
        jax_layer,
        convert_mlp(
            [pt_weights["W11"], pt_weights["W12"], pt_weights["W13"]],
            jax_layer.edge_update_mlp,
        ),
    )

    # dense: W_in -> W_out
    jax_layer = eqx.tree_at(
        lambda l: l.dense,
        jax_layer,
        convert_mlp(
            [pt_weights["dense.W_in"], pt_weights["dense.W_out"]],
            jax_layer.dense,
        ),
    )

    # LayerNorms
    jax_layer = eqx.tree_at(
        lambda l: l.norm1,
        jax_layer,
        convert_layer_norm(*pt_weights["norm1"], jax_layer.norm1),
    )
    jax_layer = eqx.tree_at(
        lambda l: l.norm2,
        jax_layer,
        convert_layer_norm(*pt_weights["norm2"], jax_layer.norm2),
    )
    jax_layer = eqx.tree_at(
        lambda l: l.norm3,
        jax_layer,
        convert_layer_norm(*pt_weights["norm3"], jax_layer.norm3),
    )

    return jax_layer


def convert_decoder_layer(
    pt_weights: dict,
    jax_layer,  # DecoderLayer
):
    """Convert PyTorch DecLayer weights to JAX DecoderLayer."""
    # message_mlp: W1 -> W2 -> W3
    jax_layer = eqx.tree_at(
        lambda l: l.message_mlp,
        jax_layer,
        convert_mlp(
            [pt_weights["W1"], pt_weights["W2"], pt_weights["W3"]],
            jax_layer.message_mlp,
        ),
    )

    # dense: W_in -> W_out
    jax_layer = eqx.tree_at(
        lambda l: l.dense,
        jax_layer,
        convert_mlp(
            [pt_weights["dense.W_in"], pt_weights["dense.W_out"]],
            jax_layer.dense,
        ),
    )

    # LayerNorms
    jax_layer = eqx.tree_at(
        lambda l: l.norm1,
        jax_layer,
        convert_layer_norm(*pt_weights["norm1"], jax_layer.norm1),
    )
    jax_layer = eqx.tree_at(
        lambda l: l.norm2,
        jax_layer,
        convert_layer_norm(*pt_weights["norm2"], jax_layer.norm2),
    )

    return jax_layer


def convert_decoder_layer_j(
    pt_weights: dict,
    jax_layer,  # DecoderLayerJ
):
    """Convert PyTorch DecLayerJ weights to JAX DecoderLayerJ."""
    # w1, w2, w3
    jax_layer = eqx.tree_at(
        lambda l: l.w1,
        jax_layer,
        convert_linear_layer(*pt_weights["W1"], jax_layer.w1),
    )
    jax_layer = eqx.tree_at(
        lambda l: l.w2,
        jax_layer,
        convert_linear_layer(*pt_weights["W2"], jax_layer.w2),
    )
    jax_layer = eqx.tree_at(
        lambda l: l.w3,
        jax_layer,
        convert_linear_layer(*pt_weights["W3"], jax_layer.w3),
    )

    # dense: W_in -> W_out
    jax_layer = eqx.tree_at(
        lambda l: l.dense,
        jax_layer,
        convert_mlp(
            [pt_weights["dense.W_in"], pt_weights["dense.W_out"]],
            jax_layer.dense,
        ),
    )

    # LayerNorms
    jax_layer = eqx.tree_at(
        lambda l: l.norm1,
        jax_layer,
        convert_layer_norm(*pt_weights["norm1"], jax_layer.norm1),
    )
    jax_layer = eqx.tree_at(
        lambda l: l.norm2,
        jax_layer,
        convert_layer_norm(*pt_weights["norm2"], jax_layer.norm2),
    )

    return jax_layer


def convert_features(pt_state_dict: dict, jax_features: Any) -> Any:
    """Convert ProteinFeatures or ProteinFeaturesLigand weights."""
    if isinstance(jax_features, ProteinFeaturesLigand):
        # Handle aliases: positional_embeddings vs embeddings
        pos_prefix = "features.embeddings." if "features.embeddings.linear.weight" in pt_state_dict else "features.positional_embeddings."

        # features.embeddings.linear -> features.embeddings.w_pos
        jax_features = eqx.tree_at(
            lambda f: f.embeddings.w_pos,
            jax_features,
            convert_linear_layer(
                pt_state_dict[pos_prefix + "linear.weight"],
                pt_state_dict.get(pos_prefix + "linear.bias"),
                jax_features.embeddings.w_pos,
            ),
        )

        # Handle enc_edge_embedding alias
        edge_weight_key = "features.edge_embedding.weight"
        if edge_weight_key not in pt_state_dict:
            edge_weight_key = "features.enc_edge_embedding.weight"

        jax_features = eqx.tree_at(
            lambda f: f.edge_embedding,
            jax_features,
            convert_linear_layer(
                pt_state_dict[edge_weight_key],
                None,
                jax_features.edge_embedding,
            ),
        )

        # Handle norm_edges alias
        norm_edges_prefix = "features.norm_edges."
        if (norm_edges_prefix + "weight") not in pt_state_dict:
            norm_edges_prefix = "features.enc_norm_edges."

        jax_features = eqx.tree_at(
            lambda f: f.norm_edges,
            jax_features,
            convert_layer_norm(
                pt_state_dict[norm_edges_prefix + "weight"],
                pt_state_dict[norm_edges_prefix + "bias"],
                jax_features.norm_edges,
            ),
        )

        # features.node_project_down
        jax_features = eqx.tree_at(
            lambda f: f.node_project_down,
            jax_features,
            convert_linear_layer(
                pt_state_dict["features.node_project_down.weight"],
                pt_state_dict["features.node_project_down.bias"],
                jax_features.node_project_down,
            ),
        )

        # Handle norm_nodes alias
        norm_nodes_prefix = "features.norm_nodes."
        if (norm_nodes_prefix + "weight") not in pt_state_dict:
            norm_nodes_prefix = "features.enc_norm_nodes."

        jax_features = eqx.tree_at(
            lambda f: f.norm_nodes,
            jax_features,
            convert_layer_norm(
                pt_state_dict[norm_nodes_prefix + "weight"],
                pt_state_dict[norm_nodes_prefix + "bias"],
                jax_features.norm_nodes,
            ),
        )
        # features.type_linear
        jax_features = eqx.tree_at(
            lambda f: f.type_linear,
            jax_features,
            convert_linear_layer(
                pt_state_dict["features.type_linear.weight"],
                pt_state_dict["features.type_linear.bias"],
                jax_features.type_linear,
            ),
        )
        # features.y_nodes, features.y_edges
        jax_features = eqx.tree_at(
            lambda f: f.y_nodes,
            jax_features,
            convert_linear_layer(
                pt_state_dict["features.y_nodes.weight"],
                None,
                jax_features.y_nodes,
            ),
        )
        jax_features = eqx.tree_at(
            lambda f: f.y_edges,
            jax_features,
            convert_linear_layer(
                pt_state_dict["features.y_edges.weight"],
                None,
                jax_features.y_edges,
            ),
        )
        # norm_y_edges, norm_y_nodes
        jax_features = eqx.tree_at(
            lambda f: f.norm_y_edges,
            jax_features,
            convert_layer_norm(
                pt_state_dict["features.norm_y_edges.weight"],
                pt_state_dict["features.norm_y_edges.bias"],
                jax_features.norm_y_edges,
            ),
        )
        jax_features = eqx.tree_at(
            lambda f: f.norm_y_nodes,
            jax_features,
            convert_layer_norm(
                pt_state_dict["features.norm_y_nodes.weight"],
                pt_state_dict["features.norm_y_nodes.bias"],
                jax_features.norm_y_nodes,
            ),
        )
    else:
        # Standard ProteinFeatures
        jax_features = eqx.tree_at(
            lambda f: f.w_pos,
            jax_features,
            convert_linear_layer(
                pt_state_dict["features.embeddings.linear.weight"],
                pt_state_dict["features.embeddings.linear.bias"],
                jax_features.w_pos,
            ),
        )
        jax_features = eqx.tree_at(
            lambda f: f.w_e,
            jax_features,
            convert_linear_layer(
                pt_state_dict["features.edge_embedding.weight"],
                None,
                jax_features.w_e,
            ),
        )
        jax_features = eqx.tree_at(
            lambda f: f.norm_edges,
            jax_features,
            convert_layer_norm(
                pt_state_dict["features.norm_edges.weight"],
                pt_state_dict["features.norm_edges.bias"],
                jax_features.norm_edges,
            ),
        )
        jax_features = eqx.tree_at(
            lambda f: f.w_e_proj,
            jax_features,
            convert_linear_layer(
                pt_state_dict["W_e.weight"],
                pt_state_dict["W_e.bias"],
                jax_features.w_e_proj,
            ),
        )
    return jax_features


def convert_packer_model(
    pt_state_dict: dict[str, np.ndarray],
    jax_model: Packer,
):
    """Convert full PyTorch Packer to JAX Packer."""
    print(f"Converting packer model from state dict with {len(pt_state_dict)} keys")

    # Feature extraction
    print("Converting features...")
    # Packer features are embedded in the model differently
    pt_feat_dict = {k[len("features."):]: v for k, v in pt_state_dict.items() if k.startswith("features.")}

    # enc_edge_embedding, enc_node_embedding, enc_norm_edges, enc_norm_nodes
    jax_model = eqx.tree_at(
        lambda m: m.features.enc_edge_embedding,
        jax_model,
        convert_linear_layer(pt_feat_dict["enc_edge_embedding.weight"], None, jax_model.features.enc_edge_embedding),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.enc_node_embedding,
        jax_model,
        convert_linear_layer(pt_feat_dict["enc_node_embedding.weight"], None, jax_model.features.enc_node_embedding),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.enc_norm_edges,
        jax_model,
        convert_layer_norm(pt_feat_dict["enc_norm_edges.weight"], pt_feat_dict["enc_norm_edges.bias"], jax_model.features.enc_norm_edges),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.enc_norm_nodes,
        jax_model,
        convert_layer_norm(pt_feat_dict["enc_norm_nodes.weight"], pt_feat_dict["enc_norm_nodes.bias"], jax_model.features.enc_norm_nodes),
    )

    # decode features
    jax_model = eqx.tree_at(
        lambda m: m.features.w_xy_project_down1,
        jax_model,
        convert_linear_layer(pt_feat_dict["W_XY_project_down1.weight"], pt_feat_dict["W_XY_project_down1.bias"], jax_model.features.w_xy_project_down1),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.dec_edge_embedding1,
        jax_model,
        convert_linear_layer(pt_feat_dict["dec_edge_embedding1.weight"], None, jax_model.features.dec_edge_embedding1),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.dec_norm_edges1,
        jax_model,
        convert_layer_norm(pt_feat_dict["dec_norm_edges1.weight"], pt_feat_dict["dec_norm_edges1.bias"], jax_model.features.dec_norm_edges1),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.dec_node_embedding1,
        jax_model,
        convert_linear_layer(pt_feat_dict["dec_node_embedding1.weight"], None, jax_model.features.dec_node_embedding1),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.dec_norm_nodes1,
        jax_model,
        convert_layer_norm(pt_feat_dict["dec_norm_nodes1.weight"], pt_feat_dict["dec_norm_nodes1.bias"], jax_model.features.dec_norm_nodes1),
    )

    # other feature weights
    jax_model = eqx.tree_at(
        lambda m: m.features.node_project_down,
        jax_model,
        convert_linear_layer(pt_feat_dict["node_project_down.weight"], pt_feat_dict["node_project_down.bias"], jax_model.features.node_project_down),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.norm_nodes,
        jax_model,
        convert_layer_norm(pt_feat_dict["norm_nodes.weight"], pt_feat_dict["norm_nodes.bias"], jax_model.features.norm_nodes),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.type_linear,
        jax_model,
        convert_linear_layer(pt_feat_dict["type_linear.weight"], pt_feat_dict["type_linear.bias"], jax_model.features.type_linear),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.y_nodes,
        jax_model,
        convert_linear_layer(pt_feat_dict["y_nodes.weight"], None, jax_model.features.y_nodes),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.y_edges,
        jax_model,
        convert_linear_layer(pt_feat_dict["y_edges.weight"], None, jax_model.features.y_edges),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.norm_y_edges,
        jax_model,
        convert_layer_norm(pt_feat_dict["norm_y_edges.weight"], pt_feat_dict["norm_y_edges.bias"], jax_model.features.norm_y_edges),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.norm_y_nodes,
        jax_model,
        convert_layer_norm(pt_feat_dict["norm_y_nodes.weight"], pt_feat_dict["norm_y_nodes.bias"], jax_model.features.norm_y_nodes),
    )
    jax_model = eqx.tree_at(
        lambda m: m.features.positional_embeddings.linear,
        jax_model,
        convert_linear_layer(pt_feat_dict["positional_embeddings.linear.weight"], pt_feat_dict["positional_embeddings.linear.bias"], jax_model.features.positional_embeddings.linear),
    )

    # Base model weights
    print("Converting base model weights...")
    for key in ["W_e", "W_v", "W_f", "W_v_sc", "linear_down", "W_torsions", "W_c", "W_e_context", "W_nodes_y", "W_edges_y", "V_C"]:
        pt_w = pt_state_dict[f"{key}.weight"]
        pt_b = pt_state_dict.get(f"{key}.bias")
        jax_layer = getattr(jax_model, key.lower())
        jax_model = eqx.tree_at(lambda m, k=key.lower(): getattr(m, k), jax_model, convert_linear_layer(pt_w, pt_b, jax_layer))

    jax_model = eqx.tree_at(
        lambda m: m.v_c_norm,
        jax_model,
        convert_layer_norm(pt_state_dict["V_C_norm.weight"], pt_state_dict["V_C_norm.bias"], jax_model.v_c_norm),
    )

    # Layers
    print("Converting layers...")
    new_encoder_layers = []
    for i, jax_layer in enumerate(jax_model.encoder_layers):
        pt_weights = get_pytorch_encoder_layer_weights(pt_state_dict, i)
        new_encoder_layers.append(convert_encoder_layer(pt_weights, jax_layer))
    jax_model = eqx.tree_at(lambda m: m.encoder_layers, jax_model, tuple(new_encoder_layers))

    new_context = []
    for i, layer in enumerate(jax_model.context_encoder_layers):
        pt_w = get_pytorch_decoder_layer_weights(pt_state_dict, i, prefix_base="context_encoder_layers")
        new_context.append(convert_decoder_layer(pt_w, layer))
    jax_model = eqx.tree_at(lambda m: m.context_encoder_layers, jax_model, tuple(new_context))

    new_y_context = []
    for i, layer in enumerate(jax_model.y_context_encoder_layers):
        pt_w = get_pytorch_decoder_layer_weights(pt_state_dict, i, prefix_base="y_context_encoder_layers")
        new_y_context.append(convert_decoder_layer_j(pt_w, layer))
    jax_model = eqx.tree_at(lambda m: m.y_context_encoder_layers, jax_model, tuple(new_y_context))

    new_decoder_layers = []
    for i, layer in enumerate(jax_model.decoder_layers):
        pt_w = get_pytorch_decoder_layer_weights(pt_state_dict, i, prefix_base="decoder_layers")
        new_decoder_layers.append(convert_decoder_layer(pt_w, layer))
    jax_model = eqx.tree_at(lambda m: m.decoder_layers, jax_model, tuple(new_decoder_layers))

    return jax_model


def convert_full_model(
    pt_state_dict: dict[str, np.ndarray],
    jax_model,  # Aminx or PrxteinLigandMPNN
):
    """Convert full PyTorch ProteinMPNN to JAX Aminx or PrxteinLigandMPNN."""
    print(f"Converting model from state dict with {len(pt_state_dict)} keys")
    from aminx.model.ligand_mpnn import PrxteinLigandMPNN

    # Convert feature extraction layers
    print("Converting features...")
    jax_model = eqx.tree_at(
        lambda m: m.features,
        jax_model,
        convert_features(pt_state_dict, jax_model.features),
    )

    # Convert encoder layers
    print("Converting encoder layers...")
    new_encoder_layers = []
    for i, jax_layer in enumerate(jax_model.encoder.layers):
        print(f"  Encoder layer {i}")
        pt_weights = get_pytorch_encoder_layer_weights(pt_state_dict, i)
        new_encoder_layers.append(convert_encoder_layer(pt_weights, jax_layer))

    jax_model = eqx.tree_at(
        lambda m: m.encoder.layers,
        jax_model,
        tuple(new_encoder_layers),
    )

    # Convert decoder layers
    print("Converting decoder layers...")
    new_decoder_layers = []
    for i, jax_layer in enumerate(jax_model.decoder.layers):
        print(f"  Decoder layer {i}")
        pt_weights = get_pytorch_decoder_layer_weights(pt_state_dict, i)
        new_decoder_layers.append(convert_decoder_layer(pt_weights, jax_layer))

    jax_model = eqx.tree_at(
        lambda m: m.decoder.layers,
        jax_model,
        tuple(new_decoder_layers),
    )

    if isinstance(jax_model, PrxteinLigandMPNN):
        print("Converting context encoders...")
        # context_encoder_layers are DecLayer in PT
        new_context = []
        for i, layer in enumerate(jax_model.context_encoder):
            prefix = f"context_encoder_layers.{i}."
            pt_w = {
                "W1": (pt_state_dict[prefix + "W1.weight"], pt_state_dict[prefix + "W1.bias"]),
                "W2": (pt_state_dict[prefix + "W2.weight"], pt_state_dict[prefix + "W2.bias"]),
                "W3": (pt_state_dict[prefix + "W3.weight"], pt_state_dict[prefix + "W3.bias"]),
                "norm1": (pt_state_dict[prefix + "norm1.weight"], pt_state_dict[prefix + "norm1.bias"]),
                "norm2": (pt_state_dict[prefix + "norm2.weight"], pt_state_dict[prefix + "norm2.bias"]),
                "dense.W_in": (pt_state_dict[prefix + "dense.W_in.weight"], pt_state_dict[prefix + "dense.W_in.bias"]),
                "dense.W_out": (pt_state_dict[prefix + "dense.W_out.weight"], pt_state_dict[prefix + "dense.W_out.bias"]),
            }
            new_context.append(convert_decoder_layer(pt_w, layer))
        jax_model = eqx.tree_at(lambda m: m.context_encoder, jax_model, tuple(new_context))

        new_y_context = []
        for i, layer in enumerate(jax_model.y_context_encoder):
            prefix = f"y_context_encoder_layers.{i}."
            pt_w = {
                "W1": (pt_state_dict[prefix + "W1.weight"], pt_state_dict[prefix + "W1.bias"]),
                "W2": (pt_state_dict[prefix + "W2.weight"], pt_state_dict[prefix + "W2.bias"]),
                "W3": (pt_state_dict[prefix + "W3.weight"], pt_state_dict[prefix + "W3.bias"]),
                "norm1": (pt_state_dict[prefix + "norm1.weight"], pt_state_dict[prefix + "norm1.bias"]),
                "norm2": (pt_state_dict[prefix + "norm2.weight"], pt_state_dict[prefix + "norm2.bias"]),
                "dense.W_in": (pt_state_dict[prefix + "dense.W_in.weight"], pt_state_dict[prefix + "dense.W_in.bias"]),
                "dense.W_out": (pt_state_dict[prefix + "dense.W_out.weight"], pt_state_dict[prefix + "dense.W_out.bias"]),
            }
            new_y_context.append(convert_decoder_layer(pt_w, layer))
        jax_model = eqx.tree_at(lambda m: m.y_context_encoder, jax_model, tuple(new_y_context))

        # Additional projections for LigandMPNN
        print("Converting ligand-specific projections...")
        for key in ["W_v", "W_c", "W_nodes_y", "W_edges_y", "V_C"]:
            jax_model = eqx.tree_at(
                lambda m, k=key.lower(): getattr(m, k),
                jax_model,
                convert_linear_layer(
                    pt_state_dict[key + ".weight"],
                    pt_state_dict.get(key + ".bias"),
                    getattr(jax_model, key.lower()),
                ),
            )

        jax_model = eqx.tree_at(
            lambda m: m.v_c_norm,
            jax_model,
            convert_layer_norm(
                pt_state_dict["V_C_norm.weight"],
                pt_state_dict["V_C_norm.bias"],
                jax_model.v_c_norm,
            ),
        )

        # W_e mapping for Ligand features
        jax_model = eqx.tree_at(
            lambda m: m.features.w_e_proj,
            jax_model,
            convert_linear_layer(
                pt_state_dict["W_e.weight"],
                pt_state_dict["W_e.bias"],
                jax_model.features.w_e_proj,
            ),
        )

    # Convert W_s (sequence embedding)
    print("Converting sequence embedding...")
    jax_model = eqx.tree_at(
        lambda m: m.w_s_embed,
        jax_model,
        convert_embedding(pt_state_dict["W_s.weight"], jax_model.w_s_embed),
    )

    # Convert W_out
    print("Converting output projection...")
    jax_model = eqx.tree_at(
        lambda m: m.w_out,
        jax_model,
        convert_linear_layer(
            pt_state_dict["W_out.weight"],
            pt_state_dict["W_out.bias"],
            jax_model.w_out,
        ),
    )

    return jax_model


def convert_physics_encoder(
    pt_state_dict: dict[str, Any],
    jax_model: Any,
) -> Any:
    """Convert membrane PhysicsEncoder weights: physics_projection, physics_norm, physics_w_v.

    Reads PT keys:
      features.node_embedding.{weight,bias} → encoder.physics_projection
      features.norm_nodes.{weight,bias}      → encoder.physics_norm
      W_v.{weight,bias}                      → encoder.physics_w_v
    """
    jax_model = eqx.tree_at(
        lambda m: m.encoder.physics_projection,
        jax_model,
        convert_linear_layer(
            pt_state_dict["features.node_embedding.weight"],
            pt_state_dict.get("features.node_embedding.bias"),
            jax_model.encoder.physics_projection,
        ),
    )
    jax_model = eqx.tree_at(
        lambda m: (m.encoder.physics_norm.weight, m.encoder.physics_norm.bias),
        jax_model,
        (
            jnp.array(pt_state_dict["features.norm_nodes.weight"]),
            jnp.array(pt_state_dict["features.norm_nodes.bias"]),
        ),
    )
    jax_model = eqx.tree_at(
        lambda m: m.encoder.physics_w_v,
        jax_model,
        convert_linear_layer(
            pt_state_dict["W_v.weight"],
            pt_state_dict.get("W_v.bias"),
            jax_model.encoder.physics_w_v,
        ),
    )
    return jax_model


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to JAX")
    parser.add_argument("--input", type=str, required=True, help="Input PyTorch .pt file")
    parser.add_argument("--output", type=str, required=True, help="Output JAX .eqx file")
    parser.add_argument(
        "--ligand-side-chain-context",
        type=str,
        choices=["auto", "on", "off"],
        default="auto",
        help="Side-chain context mode for LigandMPNN conversion (default: auto).",
    )
    args = parser.parse_args()

    print(f"Loading PyTorch weights from {args.input}")
    checkpoint_payload: dict[str, Any] | None = None

    try:
        import torch
        checkpoint = torch.load(args.input, map_location="cpu")
        checkpoint_payload = checkpoint if isinstance(checkpoint, dict) else None
        if "model_state_dict" in checkpoint:
            pt_state_dict = checkpoint["model_state_dict"]
        else:
            pt_state_dict = checkpoint

        pt_state_dict = {
            k: v.numpy() if hasattr(v, "numpy") else v
            for k, v in pt_state_dict.items()
        }
    except ImportError:
        print("PyTorch not available - loading from numpy")
        pt_state_dict = np.load(args.input, allow_pickle=True).item()

    # Initialize JAX model with matching architecture
    from aminx.model.ligand_mpnn import PrxteinLigandMPNN
    from aminx.model.mpnn import Aminx

    key = jax.random.PRNGKey(0)

    # Detect architecture type
    is_packer = "W_torsions.weight" in pt_state_dict
    is_ligand = "features.type_linear.weight" in pt_state_dict or "features.node_project_down.weight" in pt_state_dict
    is_membrane = "features.node_embedding.weight" in pt_state_dict

    # Detect positional embedding size
    pos_weight = pt_state_dict.get("features.embeddings.linear.weight")
    if pos_weight is None:
        pos_weight = pt_state_dict.get("features.positional_embeddings.linear.weight")

    if pos_weight is not None:
        num_pos = (pos_weight.shape[1] - 2) // 2
        print(f"Detected num_positional_embeddings: {num_pos}")
    else:
        num_pos = 16

    if is_packer:
        print("Detected Packer architecture")
        jax_model = Packer(
            edge_features=128,
            node_features=128,
            num_positional_embeddings=num_pos,
            num_rbf=16,
            top_k=30,
            atom_context_num=16,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            key=key,
        )
        jax_model = convert_packer_model(pt_state_dict, jax_model)
    elif is_ligand:
        print("Detected LigandMPNN architecture")
        use_side_chain_context = resolve_ligand_side_chain_context(
            args.ligand_side_chain_context,
            checkpoint_payload=checkpoint_payload,
            input_path=args.input,
        )
        print(f"  ligand_mpnn_use_side_chain_context={use_side_chain_context}")
        # Base config for LigandMPNN weights
        jax_model = PrxteinLigandMPNN(
            node_features=128,
            edge_features=128,
            hidden_features=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=32 if "32" in args.input else 48,
            # A property of the checkpoint (reference: run.py reads checkpoint["atom_context_num"]).
            atom_context_num=int((checkpoint_payload or {}).get("atom_context_num", 16)),
            num_positional_embeddings=num_pos,
            num_context_layers=NUM_LIGAND_CONTEXT_LAYERS,
            ligand_mpnn_use_side_chain_context=use_side_chain_context,
            dropout_rate=0.0,
            key=key,
        )
        jax_model = convert_full_model(pt_state_dict, jax_model)
    else:
        print("Detected standard ProteinMPNN architecture")
        if is_membrane:
            print("  Special case: Membrane model detected (using PhysicsEncoder for node_embedding)")

        jax_model = Aminx(
            node_features=128,
            edge_features=128,
            hidden_features=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=48,
            num_positional_embeddings=num_pos,
            physics_feature_dim=3 if is_membrane else None,
            dropout_rate=0.0,
            key=key,
        )
        jax_model = convert_full_model(pt_state_dict, jax_model)

        if is_membrane:
            jax_model = convert_physics_encoder(pt_state_dict, jax_model)

    # Save
    print(f"Saving JAX weights to {args.output}")
    eqx.tree_serialise_leaves(args.output, jax_model)
    print("Done!")


if __name__ == "__main__":
    main()
