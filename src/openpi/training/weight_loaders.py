import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class Pi0DexWeightLoader(WeightLoader):
    """Loads weights from a pi0 checkpoint for pi0_dex model with combined module.
    
    This loader handles the combined module architecture where all three experts are in one module:
    - Expert 0: PaliGemma (2B, frozen) - loaded from checkpoint's PaliGemma/llm/... (no suffix)
    - Expert 1: Action expert (300M) - loaded from checkpoint's PaliGemma/llm/..._1/... (same)
    - Expert 2: Hand expert (300M) - copied from checkpoint's PaliGemma/llm/..._1/... → ..._2/... (change suffix)
    
    This loader:
    1. Loads expert 0 (PaliGemma) and expert 1 (action_expert) directly from checkpoint
    2. Copies expert 1 weights to expert 2 (hand_expert) by changing _1 suffix to _2
    3. Handles LoRA parameters (initialized automatically if using LoRA variants)
    4. Handles KV transformation MLPs (initialized to zero, so not loaded)
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load the checkpoint (from pi0 model)
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        
        result = {}
        
        # Step 1: Copy weights that match directly (expert 0, expert 1, img, projections, etc.)
        for k, v in flat_loaded.items():
            if k in flat_ref:
                result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
        
        # Step 2: Map expert 1 weights (with _1 suffix) to expert 2 (with _2 suffix) for hand expert
        # The checkpoint has expert 1 (action_expert) with _1 suffix, we need to copy it to expert 2 (hand_expert) with _2 suffix
        # NOTE: Do NOT map the embedder - expert 2's embedder has width 1024 (300M), while checkpoint's embedder has width 2048 (2B)
        pattern = re.compile(r"^PaliGemma/llm/(.*)$")
        for k, v in flat_loaded.items():
            match = pattern.match(k)
            if match:
                path_suffix = match.group(1)
                # Skip embedder - it has different size (1024 vs 2048) and will be initialized fresh
                if path_suffix.startswith("embedder/"):
                    continue
                # Only map expert weights (those with _1 suffix) to expert 2 (with _2 suffix)
                if "_1" in path_suffix:
                    # Map expert weights: change _1 suffix to _2 for hand expert
                    new_path = path_suffix.replace("_1", "_2")
                    new_key = f"PaliGemma/llm/{new_path}"
                    if new_key in flat_ref:
                        result[new_key] = v.astype(flat_ref[new_key].dtype) if v.dtype != flat_ref[new_key].dtype else v
        
        # Step 3: Add missing LoRA weights (if using LoRA)
        lora_pattern = re.compile(".*lora.*")
        for k in {k for k in flat_ref if lora_pattern.fullmatch(k)}:
            if k not in result:
                result[k] = flat_ref[k]
        
        # Step 4: Add missing KV transformation MLPs (optional - only if model uses them)
        # Note: These may be commented out for memory optimization (identity transformation)
        kv_transform_pattern = re.compile(".*kv_transform.*")
        for k in {k for k in flat_ref if kv_transform_pattern.fullmatch(k)}:
            if k not in result:
                result[k] = flat_ref[k]
        
        # Step 5: Add missing hand projection and time MLP layers (new in pi0_dex, not in pi0 checkpoint)
        hand_layers_pattern = re.compile(".*(hand_in_proj|hand_out_proj|hand_time_mlp_in|hand_time_mlp_out).*")
        for k in {k for k in flat_ref if hand_layers_pattern.fullmatch(k)}:
            if k not in result:
                result[k] = flat_ref[k]
        
        # Step 6: Add expert 2 embedder from reference (not loaded from checkpoint due to size mismatch)
        # The embedder will be initialized fresh with the correct 1024 width (300M expert)
        embedder_pattern = re.compile(r"^PaliGemma/llm/.*_2/embedder/.*$")
        for k in {k for k in flat_ref if embedder_pattern.fullmatch(k)}:
            if k not in result:
                result[k] = flat_ref[k]
        
        return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
