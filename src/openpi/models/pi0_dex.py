import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0_dex(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs): #rngs is a generator of random numbers
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        hand_expert_config = _gemma.get_config(config.hand_expert_variant) 
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        # llm_hand only needs the 300M expert - the 2B base comes from frozen llm's KV cache
        # Use only hand_expert_config since we pass [hand_suffix_tokens] (single expert, no 2B part)
        llm_hand = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[hand_expert_config],  # Only 300M expert, no 2B base needed
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm_hand.lazy_init(rngs=rngs, method="init", use_adarms=[True] if config.pi05 else [False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, llm_hand=llm_hand, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.hand_in_proj = nnx.Linear(config.action_dim, hand_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.hand_time_mlp_in = nnx.Linear(2 * hand_expert_config.width, hand_expert_config.width, rngs=rngs)
            self.hand_time_mlp_out = nnx.Linear(hand_expert_config.width, hand_expert_config.width, rngs=rngs)
        self.hand_out_proj = nnx.Linear(hand_expert_config.width, config.action_dim, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        
        # MLP structure for B, C matrices: transform KV cache from frozen llm to llm_hand
        # MEMORY OPTIMIZATION: Currently commented out
        # Instead, using identity transformation and letting hand expert learn query projections
        # that work with frozen KV cache.
        # 
        # If you want to re-enable KV transformation, uncomment below and uncomment MLP code in transform_kv_cache
        # head_dim = hand_expert_config.head_dim
        # self.kv_transform_k_mlp_in = nnx.Linear(head_dim, head_dim, rngs=rngs)
        # self.kv_transform_k_mlp_out = nnx.Linear(head_dim, head_dim, rngs=rngs)
        # self.kv_transform_v_mlp_in = nnx.Linear(head_dim, head_dim, rngs=rngs)
        # self.kv_transform_v_mlp_out = nnx.Linear(head_dim, head_dim, rngs=rngs)
        # self._zero_init_kv_transform_mlps()
        
        # mlp structure for D, E matrices (for future use if needed)
        # TODO
        
        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True
    
    # MEMORY OPTIMIZATION: Commented out - KV transformation MLPs not needed with identity transformation
    # def _zero_init_kv_transform_mlps(self):
    #     """Zero-initialize the kernels of KV transformation MLPs for residual connection."""
    #     # Directly set kernel values to zero so the transformation starts as identity
    #     for name in ['kv_transform_k_mlp_in', 'kv_transform_k_mlp_out', 
    #                  'kv_transform_v_mlp_in', 'kv_transform_v_mlp_out']:
    #         mlp = getattr(self, name)
    #         mlp_state = nnx.state(mlp)
    #         if 'kernel' in mlp_state:
    #             # Create a new Param with zero values
    #             zero_kernel = nnx.Param(jnp.zeros_like(mlp_state['kernel'].value))
    #             # Update the module's kernel directly
    #             mlp.kernel = zero_kernel
    
    def transform_kv_cache(self, kv_cache):
        """Transform KV cache from frozen llm to be used by llm_hand.
        
        MEMORY OPTIMIZATION: Currently using identity transformation (no MLP).
        The hand expert learns query projections that work with frozen KV cache.
        This saves ~9.65 GB of activation memory from KV transformation MLPs.
        
        Args:
            kv_cache: Tuple of (K, V) caches, each of shape [l, b, s, k, h]
                     where l=num_layers, b=batch, s=seq_len, k=num_kv_heads, h=head_dim
        
        Returns:
            KV cache unchanged (identity transformation). The hand expert's learnable
            query projections adapt to work with the frozen keys/values.
        """
        if kv_cache is None:
            return None
        
        # Identity transformation: just return the KV cache as-is
        # The hand expert will learn query projections that align with frozen K/V
        # This saves ~9.65 GB of activation memory from KV transformation MLPs
        return kv_cache
        
        # OPTION: Uncomment below to use MLP transformation (more memory, potentially better adaptation)
        # k_cache, v_cache = kv_cache
        # original_dtype = k_cache.dtype
        # l, b, s, k, h = k_cache.shape
        # k_flat = k_cache.reshape(-1, h)
        # k_transformed = self.kv_transform_k_mlp_in(k_flat)
        # k_transformed = nnx.swish(k_transformed)
        # k_transformed = self.kv_transform_k_mlp_out(k_transformed)
        # k_transformed = k_flat + k_transformed
        # k_transformed = k_transformed.astype(original_dtype)
        # k_cache_transformed = k_transformed.reshape(l, b, s, k, h)
        # v_flat = v_cache.reshape(-1, h)
        # v_transformed = self.kv_transform_v_mlp_in(v_flat)
        # v_transformed = nnx.swish(v_transformed)
        # v_transformed = self.kv_transform_v_mlp_out(v_transformed)
        # v_transformed = v_flat + v_transformed
        # v_transformed = v_transformed.astype(original_dtype)
        # v_cache_transformed = v_transformed.reshape(l, b, s, k, h)
        # return (k_cache_transformed, v_cache_transformed)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
        at.Float[at.Array, "b emb"] | None,
    ]:
        # Process action tokens
        action_tokens = self.action_in_proj(noisy_actions)
        action_time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        action_input_mask = []
        action_ar_mask = []
        action_tokens_list = []
        if not self.pi05:
            # add a single state token to action expert
            state_token = self.state_proj(obs.state)[:, None, :]
            action_tokens_list.append(state_token)
            action_input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            action_ar_mask += [True]
        
        if self.pi05:
            # time MLP (for adaRMS)
            action_time_emb = self.time_mlp_in(action_time_emb)
            action_time_emb = nnx.swish(action_time_emb)
            action_time_emb = self.time_mlp_out(action_time_emb)
            action_time_emb = nnx.swish(action_time_emb)
            action_expert_tokens = action_tokens
            action_adarms_cond = action_time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(action_time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            action_adarms_cond = None
        action_tokens_list.append(action_expert_tokens)
        action_input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        action_ar_mask += [True] + ([False] * (self.action_horizon - 1))
        action_suffix_tokens = jnp.concatenate(action_tokens_list, axis=1)
        action_suffix_mask = jnp.concatenate(action_input_mask, axis=1)
        action_suffix_ar_mask = jnp.array(action_ar_mask)

        # Process hand tokens (same as action tokens)
        hand_tokens = self.hand_in_proj(noisy_actions)
        hand_time_emb = posemb_sincos(timestep, self.hand_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        hand_input_mask = []
        hand_ar_mask = []
        hand_tokens_list = []
        if not self.pi05:
            # add a single state token to hand expert (same as action expert)
            state_token = self.state_proj(obs.state)[:, None, :]
            hand_tokens_list.append(state_token)
            hand_input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            hand_ar_mask += [True]
        
        if self.pi05:
            # time MLP (for adaRMS) - share the same time MLP as action since they're the same
            hand_time_emb = self.time_mlp_in(hand_time_emb)
            hand_time_emb = nnx.swish(hand_time_emb)
            hand_time_emb = self.time_mlp_out(hand_time_emb)
            hand_time_emb = nnx.swish(hand_time_emb)
            hand_expert_tokens = hand_tokens
            hand_adarms_cond = hand_time_emb
        else:
            # mix timestep + hand information using an MLP (no adaRMS)
            hand_time_tokens = einops.repeat(hand_time_emb, "b emb -> b s emb", s=self.action_horizon)
            hand_time_tokens_concat = jnp.concatenate([hand_tokens, hand_time_tokens], axis=-1)
            hand_time_tokens_concat = self.hand_time_mlp_in(hand_time_tokens_concat)
            hand_time_tokens_concat = nnx.swish(hand_time_tokens_concat)
            hand_time_tokens_concat = self.hand_time_mlp_out(hand_time_tokens_concat)
            hand_expert_tokens = hand_time_tokens_concat
            hand_adarms_cond = None
        hand_tokens_list.append(hand_expert_tokens)
        hand_input_mask.append(jnp.ones(hand_expert_tokens.shape[:2], dtype=jnp.bool_))
        hand_ar_mask += [True] + ([False] * (self.action_horizon - 1))
        hand_suffix_tokens = jnp.concatenate(hand_tokens_list, axis=1)
        hand_suffix_mask = jnp.concatenate(hand_input_mask, axis=1)
        hand_suffix_ar_mask = jnp.array(hand_ar_mask)

        return action_suffix_tokens, hand_suffix_tokens, action_suffix_mask, hand_suffix_mask, action_suffix_ar_mask, hand_suffix_ar_mask, action_adarms_cond, hand_adarms_cond

    def embed_hand_suffix_only(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Embed only hand suffix tokens (memory optimization for compute_loss).
        
        This avoids computing action_suffix_tokens which are not needed in compute_loss,
        reducing JIT compilation memory by ~50% (from ~30GB to ~15GB).
        """
        # Process hand tokens (same logic as embed_suffix but only for hand)
        hand_tokens = self.hand_in_proj(noisy_actions)
        hand_time_emb = posemb_sincos(timestep, self.hand_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        hand_input_mask = []
        hand_ar_mask = []
        hand_tokens_list = []
        if not self.pi05:
            # add a single state token to hand expert
            state_token = self.state_proj(obs.state)[:, None, :]
            hand_tokens_list.append(state_token)
            hand_input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            hand_ar_mask += [True]
        
        if self.pi05:
            # time MLP (for adaRMS)
            hand_time_emb = self.time_mlp_in(hand_time_emb)
            hand_time_emb = nnx.swish(hand_time_emb)
            hand_time_emb = self.time_mlp_out(hand_time_emb)
            hand_time_emb = nnx.swish(hand_time_emb)
            hand_expert_tokens = hand_tokens
            hand_adarms_cond = hand_time_emb
        else:
            # mix timestep + hand information using an MLP (no adaRMS)
            hand_time_tokens = einops.repeat(hand_time_emb, "b emb -> b s emb", s=self.action_horizon)
            hand_time_tokens_concat = jnp.concatenate([hand_tokens, hand_time_tokens], axis=-1)
            hand_time_tokens_concat = self.hand_time_mlp_in(hand_time_tokens_concat)
            hand_time_tokens_concat = nnx.swish(hand_time_tokens_concat)
            hand_time_tokens_concat = self.hand_time_mlp_out(hand_time_tokens_concat)
            hand_expert_tokens = hand_time_tokens_concat
            hand_adarms_cond = None
        hand_tokens_list.append(hand_expert_tokens)
        hand_input_mask.append(jnp.ones(hand_expert_tokens.shape[:2], dtype=jnp.bool_))
        hand_ar_mask += [True] + ([False] * (self.action_horizon - 1))
        hand_suffix_tokens = jnp.concatenate(hand_tokens_list, axis=1)
        hand_suffix_mask = jnp.concatenate(hand_input_mask, axis=1)
        hand_suffix_ar_mask = jnp.array(hand_ar_mask)

        return hand_suffix_tokens, hand_suffix_mask, hand_suffix_ar_mask, hand_adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001 # beta-distributed time sampling in FM
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix and suffix tokens
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # MEMORY OPTIMIZATION: Only compute hand_suffix_tokens for compute_loss
        hand_suffix_tokens, hand_suffix_mask, hand_suffix_ar_mask, hand_adarms_cond = self.embed_hand_suffix_only(observation, x_t, time)
        
        # Forward pass through frozen llm to get KV cache
        # We need to process prefix through frozen llm to get KV cache
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # Forward pass through frozen llm with aggressive checkpointing to save memory
        # Wrap the forward pass with jax.checkpoint to prevent storing intermediate activations
        # This is critical: without checkpointing, JAX materializes the entire 2.3B model
        # computation graph during JIT compilation, causing OOM
        def frozen_llm_forward(prefix_tokens, prefix_attn_mask, prefix_positions):
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None], 
                mask=prefix_attn_mask, 
                positions=prefix_positions
            )
            return kv_cache
        
        # Use checkpointing: recompute activations instead of storing them
        # prevent_cse=True prevents common subexpression elimination that could increase memory
        # policy=None uses default (save only what's necessary for gradients, but we'll stop_gradient anyway)
        kv_cache_frozen = jax.checkpoint(
            frozen_llm_forward,
            prevent_cse=True,
        )(prefix_tokens, prefix_attn_mask, prefix_positions)
        
        # Stop gradient on KV cache to ensure no gradients flow back
        # This is critical: tells JAX/XLA we don't need the computation graph for gradients
        kv_cache_frozen = jax.tree.map(jax.lax.stop_gradient, kv_cache_frozen)
        
        # Transform KV cache from frozen llm to be used by llm_hand
        # This transformation needs gradients (MLPs are trainable)
        kv_cache_transformed = self.transform_kv_cache(kv_cache_frozen)
        
        # When using KV cache, construct attention mask correctly:
        # - suffix_attn_mask: how suffix tokens attend to each other (b, suffix_len, suffix_len)
        hand_suffix_attn_mask = make_attn_mask(hand_suffix_mask, hand_suffix_ar_mask)
        # - prefix_attn_mask: how suffix tokens attend to prefix tokens (b, suffix_len, prefix_len)
        prefix_attn_mask_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=hand_suffix_tokens.shape[1])
        # - full_attn_mask: how suffix tokens attend to full sequence (b, suffix_len, prefix_len + suffix_len)
        full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, hand_suffix_attn_mask], axis=-1)
        batch_size = observation.state.shape[0]
        assert full_attn_mask.shape == (
            batch_size,
            hand_suffix_tokens.shape[1],
            prefix_tokens.shape[1] + hand_suffix_tokens.shape[1],
        )
        # Positions for suffix tokens (relative to start of sequence)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(hand_suffix_mask, axis=-1) - 1
        
        # Use llm_hand with transformed KV cache for hand action inference
        # llm_hand only has the 300M expert (no 2B base), so pass just hand_suffix_tokens
        (hand_suffix_out,), _ = self.PaliGemma.llm_hand(
            [hand_suffix_tokens], 
            mask=full_attn_mask, 
            positions=positions, 
            kv_cache=kv_cache_transformed,
            adarms_cond=[hand_adarms_cond]
        )
        
        # Project hand expert outputs to action space
        hand_v_t = self.hand_out_proj(hand_suffix_out[:, -self.action_horizon :])
        # No need to mask arm dimensions, compute loss on all dimensions
        loss = jnp.mean(jnp.square(hand_v_t - u_t), axis=-1)
        
        return loss

    def sample_arm(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, tuple]:
        """Sample arm actions using frozen llm and return actions + KV cache.
        
        Returns:
            arm_actions: Arm actions (first 3 dimensions) of shape [b, ah, 3]
            kv_cache: KV cache from frozen llm that can be used for hand sampling
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # First fill KV cache with a forward pass of the prefix through frozen llm
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache_frozen = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)

        def step(carry):
            x_t, time = carry
            # Embed suffix tokens for arm actions
            action_suffix_tokens, _, action_suffix_mask, _, action_suffix_ar_mask, _, action_adarms_cond, _ = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each other
            suffix_attn_mask = make_attn_mask(action_suffix_mask, action_suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=action_suffix_tokens.shape[1])
            # `full_attn_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                action_suffix_tokens.shape[1],
                prefix_tokens.shape[1] + action_suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(action_suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, action_suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache_frozen,  # Reuse cached prefix, don't update
                adarms_cond=[None, action_adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        
        # Extract only arm dimensions (first 3)
        arm_actions = x_0[..., :3]
        return arm_actions, kv_cache_frozen

    def sample_hand(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        kv_cache_frozen: tuple,
        arm_actions: _model.Actions,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample hand actions using llm_hand with transformed KV cache from frozen llm.
        
        Args:
            rng: Random key
            observation: Observation
            kv_cache_frozen: KV cache from frozen llm (from sample_arm)
            arm_actions: Arm actions from sample_arm, used to construct full actions for hand sampling
            num_steps: Number of diffusion steps
            noise: Optional noise tensor for hand actions
        
        Returns:
            hand_actions: Hand actions (remaining dimensions) of shape [b, ah, action_dim-3]
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        
        # Create full action tensor with arm actions and noise for hand dimensions
        # For hand sampling, we fix arm actions and only sample hand dimensions
        if noise is None:
            hand_noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim - 3))
            full_noise = jnp.concatenate([arm_actions, hand_noise], axis=-1)
        else:
            # Use provided noise but fix arm dimensions to arm_actions
            full_noise = noise.at[..., :3].set(arm_actions)
        
        # Transform KV cache from frozen llm to be used by llm_hand
        kv_cache_transformed = self.transform_kv_cache(kv_cache_frozen)
        
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        def step(carry):
            x_t, time = carry
            # Embed suffix tokens for hand actions
            _, hand_suffix_tokens, _, hand_suffix_mask, _, hand_suffix_ar_mask, _, hand_adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each other
            suffix_attn_mask = make_attn_mask(hand_suffix_mask, hand_suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=hand_suffix_tokens.shape[1])
            # `full_attn_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                hand_suffix_tokens.shape[1],
                prefix_tokens.shape[1] + hand_suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(hand_suffix_mask, axis=-1) - 1

            (suffix_out,), _ = self.PaliGemma.llm_hand(
                [hand_suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache_transformed,  # Reuse cached prefix, don't update
                adarms_cond=[hand_adarms_cond],
            )
            
            # Project to action space and get hand actions
            hand_v_t = self.hand_out_proj(suffix_out[:, -self.action_horizon :])
            
            # Only update hand dimensions, keep arm actions fixed
            # Extract hand dimensions from v_t and update only those
            hand_v_t_only = hand_v_t[..., 3:]  # Hand dimensions only
            x_t_hand = x_t[..., 3:]  # Current hand state
            x_t_hand_updated = x_t_hand + dt * hand_v_t_only
            
            # Combine fixed arm actions with updated hand actions
            x_t_updated = jnp.concatenate([arm_actions, x_t_hand_updated], axis=-1)

            return x_t_updated, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (full_noise, 1.0))
        
        # Extract only hand dimensions (remaining dimensions after first 3)
        hand_actions = x_0[..., 3:]
        return hand_actions

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions by first sampling arm actions, then hand actions.
        
        Pipeline:
        1. Sample arm actions using frozen llm -> get arm_actions + kv_cache
        2. Transform kv_cache and sample hand actions using llm_hand
        3. Combine arm and hand actions
        """
        rng_arm, rng_hand = jax.random.split(rng, 2)
        
        # Step 1: Sample arm actions first
        arm_actions, kv_cache_frozen = self.sample_arm(
            rng_arm,
            observation,
            num_steps=num_steps,
            noise=noise[..., :3] if noise is not None else None,
        )
        
        # Step 2: Sample hand actions using transformed KV cache
        hand_actions = self.sample_hand(
            rng_hand,
            observation,
            kv_cache_frozen,
            arm_actions,
            num_steps=num_steps,
            noise=noise[..., 3:] if noise is not None else None,
        )
        
        # Step 3: Combine arm and hand actions
        full_actions = jnp.concatenate([arm_actions, hand_actions], axis=-1)
        return full_actions
