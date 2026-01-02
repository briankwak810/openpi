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
        """Returns the freeze filter for Pi0_dex model with LoRA fine-tuning of llm_hand.
        
        For Pi0_dex:
        - Freeze llm (frozen arm model): PaliGemma/llm/...
        - Freeze llm_hand base params but allow LoRA: PaliGemma/llm_hand/... (excluding lora)
        - Freeze img (image encoder): PaliGemma/img/...
        - Allow KV transformation MLPs and other trainable components
        
        Note: This checks hand_expert_variant for "lora" to determine if llm_hand should be frozen with LoRA trainable.
        """
        filters = []
        
        # Freeze the frozen llm (arm model) completely
        frozen_llm_filter = nnx_utils.PathRegex(".*PaliGemma/llm.*")
        filters.append(frozen_llm_filter)
        
        # Freeze the image encoder completely
        img_filter = nnx_utils.PathRegex(".*PaliGemma/img.*")
        filters.append(img_filter)
        
        # Freeze llm_hand base params but allow LoRA params
        # Check hand_expert_variant for "lora" (indicating LoRA fine-tuning of llm_hand)
        if "lora" in self.hand_expert_variant:
            llm_hand_base_filter = nnx_utils.PathRegex(".*PaliGemma/llm_hand.*")
            filters.append(llm_hand_base_filter)
            
            # Exclude LoRA parameters from being frozen (they should be trainable)
            lora_filter = nnx_utils.PathRegex(".*lora.*")
            filters.append(nnx.Not(lora_filter))
        
        # Also exclude KV transformation MLPs (they should be trainable)
        kv_transform_filter = nnx_utils.PathRegex(".*kv_transform.*")
        filters.append(nnx.Not(kv_transform_filter))
        
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)