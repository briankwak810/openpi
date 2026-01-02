import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    hand_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 20
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # Pi0_dex is a variant with dual LLMs (llm and llm_hand) for dexterous manipulation
    dex: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        if self.dex:
            from openpi.models.pi0_dex import Pi0_dex
            return Pi0_dex(self, rngs=nnx.Rngs(rng))
        from openpi.models.pi0 import Pi0
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
    
    def get_freeze_filter_dex(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter for Pi0_dex model with LoRA fine-tuning of hand expert.
        
        For Pi0_dex with combined module [PaliGemma, action_expert, hand_expert]:
        - Freeze expert 0 (PaliGemma 2B): PaliGemma/llm/... (no suffix)
        - Freeze expert 2 (hand_expert) base params but allow LoRA: PaliGemma/llm/..._2/... (excluding lora)
        - Freeze img (image encoder): PaliGemma/img/...
        - Allow expert 1 (action_expert) to be trainable: PaliGemma/llm/..._1/...
        - Allow KV transformation MLPs and other trainable components
        
        Note: Expert naming in gemma Module:
        - Expert 0: no suffix (e.g., "attn")
        - Expert 1: suffix "_1" (e.g., "attn_1")
        - Expert 2: suffix "_2" (e.g., "attn_2")
        """
        filters = []
        
        # Freeze expert 0 (PaliGemma 2B) - paths without expert suffix
        # Match PaliGemma/llm/... but exclude expert 1 and expert 2 paths
        frozen_llm_base = nnx_utils.PathRegex(".*PaliGemma/llm.*")
        expert_1_filter = nnx_utils.PathRegex(".*PaliGemma/llm.*_1.*")
        expert_2_filter = nnx_utils.PathRegex(".*PaliGemma/llm.*_2.*")
        # Freeze base (expert 0) but not expert 1 or expert 2
        filters.append(frozen_llm_base)
        filters.append(nnx.Not(expert_1_filter))  # Don't freeze expert 1
        filters.append(nnx.Not(expert_2_filter))  # Don't freeze expert 2 (will handle separately)
        
        # Freeze the image encoder completely
        img_filter = nnx_utils.PathRegex(".*PaliGemma/img.*")
        filters.append(img_filter)
        
        # Freeze expert 2 (hand_expert) base params but allow LoRA params
        # Check hand_expert_variant for "lora" (indicating LoRA fine-tuning of hand_expert)
        if "lora" in self.hand_expert_variant:
            hand_expert_base_filter = nnx_utils.PathRegex(".*PaliGemma/llm.*_2.*")
            filters.append(hand_expert_base_filter)
            
            # Exclude LoRA parameters from being frozen (they should be trainable)
            lora_filter = nnx_utils.PathRegex(".*lora.*")
            filters.append(nnx.Not(lora_filter))
        
        # Also exclude KV transformation MLPs (they should be trainable)
        kv_transform_filter = nnx_utils.PathRegex(".*kv_transform.*")
        filters.append(nnx.Not(kv_transform_filter))
        
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)