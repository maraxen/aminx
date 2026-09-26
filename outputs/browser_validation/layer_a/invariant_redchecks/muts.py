MUTS = {
 1: ("src/aminx/model/decoder.py", "  attention_mask = jnp.take_along_axis(ar_mask, neighbor_indices, axis=1)", "  attention_mask = 1 - jnp.take_along_axis(ar_mask, neighbor_indices, axis=1)"),
 2: ("src/aminx/inference/decode/unconditional.py", "      node_features, edge_features, neighbor_indices, mask = inputs\n", "      node_features, edge_features, neighbor_indices, mask = inputs\n      node_features = node_features + jnp.atleast_2d(cond.sequence_oh) @ self.model.w_s_embed.weight\n"),
 3: ("src/aminx/inference/decode/autoregressive.py", "        final_token = jnp.where(is_group_fixed, group_fixed_token, sampled).astype(jnp.int32)  # (G,)", "        final_token = sampled.astype(jnp.int32)  # (G,)"),
 4: ("src/aminx/inference/decode/autoregressive.py", "          )(logits, cond_bias)  # (L, 21)", "          )(logits, zeros_bias)  # (L, 21)"),
 5: ("src/aminx/inference/decode/autoregressive.py", "jax.random.categorical(k, sampling_logits_g / cond.temperature)", "jax.random.categorical(k, sampling_logits_g * cond.temperature)"),
 6: ("src/aminx/inference/decode/autoregressive.py", "      mask_group = (cond.tie_group_map[0][None, :] == group_id[:, None]) & is_first_occurrence[:, None]  # (G, L)", "      mask_group = (jnp.arange(L)[None, :] == pos0[:, None]) & is_first_occurrence[:, None]  # (G, L)"),
 7: ("src/aminx/model/encoder.py", "      if initial_node_features is None\n      else jax.vmap(self.physics_w_v)(", "      if True\n      else jax.vmap(self.physics_w_v)("),
 8: ("src/aminx/model/features.py", "      _, neighbor_indices = top_k(-distances_masked, k)", "      _, neighbor_indices = top_k(distances_masked, k)"),
}
import sys
n=int(sys.argv[1]); f,old,new=MUTS[n]
t=open(f).read(); c=t.count(old)
if c!=1: sys.exit(f"MUT{n}: old string count={c} in {f}")
open(f,"w").write(t.replace(old,new)); print(f"MUT{n} applied to {f}")
