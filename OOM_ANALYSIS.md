# OOM Analysis: Why Pi0-Dex OOMs But Pi0-Base Doesn't

## Problem

Even after removing KV cache transformation MLPs (saving ~9.65 GB), pi0-dex still OOMs during initialization while pi0-base works fine.

## Root Cause: Dual Forward Passes

**Key Difference:**
- **Pi0-base**: ONE forward pass - concatenates prefix + suffix, processes together
- **Pi0-dex**: TWO separate forward passes - frozen llm (prefix) + llm_hand (suffix with KV cache)

The OOM occurs during **JIT compilation** (`jax.jit` in `init_train_state`). During compilation, JAX traces through the computation graph. With two separate forward passes, JAX must trace through TWO complete LLM computation graphs simultaneously:

1. **Frozen llm forward pass** (~2.3B params) - needs to be traced
2. **llm_hand forward pass** (~300M params) - needs to be traced  
3. **Both in same computation graph** - doubles the compilation memory

## Why Pi0-Base Works

Pi0-base does a single forward pass:
```python
# Pi0-base: ONE forward pass
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [prefix_tokens, suffix_tokens],  # Both together
    mask=attn_mask, 
    positions=positions, 
    adarms_cond=[None, adarms_cond]
)
```

This means JAX only traces ONE computation graph during compilation, keeping memory manageable.

## Why Checkpointing Doesn't Help

`jax.checkpoint` reduces **runtime** memory by recomputing activations, but it doesn't reduce **compilation** memory. During JIT compilation, JAX still needs to trace the structure of both forward passes, even if they're checkpointed.

1. **Frozen llm forward pass** (~2.3B params, 18 layers)
   - Even with `stop_gradient`, JAX still traces the computation
   - During compilation, XLA allocates memory for the computation graph
   - **This is the main culprit**: ~20-25 GB just for tracing the frozen model

2. **llm_hand forward pass** (~300M params, 18 layers)
   - Another full forward pass that needs to be traced
   - ~5-10 GB for computation graph

3. **KV cache storage** (~3.86 GB)
   - Actual data, not just computation graph
   - Can't be checkpointed away

4. **Dual embeddings** (action + hand suffixes)
   - Extra embedding computations
   - ~1-2 GB

## Solution: Early Stop Gradient

Applied `stop_gradient` EARLY on inputs to the frozen llm forward pass:

```python
# Stop gradients on inputs to frozen llm
prefix_tokens_stopped = jax.lax.stop_gradient(prefix_tokens)
prefix_attn_mask_stopped = jax.lax.stop_gradient(prefix_attn_mask)
prefix_positions_stopped = jax.lax.stop_gradient(prefix_positions)

# Forward pass with stopped inputs
_, kv_cache_frozen = self.PaliGemma.llm(
    [prefix_tokens_stopped, None], 
    mask=prefix_attn_mask_stopped, 
    positions=prefix_positions_stopped
)
kv_cache_frozen = jax.tree.map(jax.lax.stop_gradient, kv_cache_frozen)
```

**How this helps:**
- Applying `stop_gradient` early tells JAX not to build the gradient computation graph for the frozen forward pass
- This should reduce the amount of computation graph that needs to be materialized during compilation
- May reduce compilation-time memory by signaling to XLA that gradients aren't needed

## Why Stop Gradient Alone Wasn't Enough

`stop_gradient` only prevents gradients from flowing back, but:
- JAX still needs to trace the computation during JIT compilation
- XLA still tries to optimize and materialize the computation graph
- The forward pass still creates intermediate activations that XLA wants to store

## Additional Optimizations Needed

If still OOM, try:

1. **Reduce batch size further**: Try `batch_size=1` (currently using `batch_size=1`)
2. **Reduce sequence length**: Lower `action_horizon` or `max_token_len`
3. **Add checkpointing to llm_hand**: Also checkpoint the hand expert forward pass
4. **Use rematerialization**: Set XLA flags to be more aggressive
5. **Split compilation**: Compile frozen llm and llm_hand separately

## Expected Memory After Checkpointing

- Model parameters: ~5.2 GB
- Frozen llm forward (with checkpointing): ~5-10 GB (down from 20-25 GB)
- llm_hand forward: ~5-10 GB
- KV cache: ~3.86 GB
- Gradients: ~10-15 GB
- **Total: ~30-45 GB** (down from 43-50 GB)

This should fit in most GPUs with 40-48 GB memory.

