# Freeze Filter vs stop_gradient: Why Freeze Doesn't Help with Compilation OOM

## How Freeze Filter Works

The freeze filter operates at the **training infrastructure level**:

1. **Parameter filtering** (line 111 in `train.py`):
   ```python
   opt_state=tx.init(params.filter(config.trainable_filter))
   ```
   - Frozen params are excluded from optimizer state
   - Only trainable params get optimizer momentum/state

2. **Gradient computation filtering** (line 157-158):
   ```python
   diff_state = nnx.DiffState(0, config.trainable_filter)
   loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(...)
   ```
   - `nnx.DiffState` with `trainable_filter` tells JAX to only compute gradients for trainable params
   - Frozen params don't get gradients computed

3. **Memory optimization** (line 104):
   ```python
   params = nnx_utils.state_map(params, config.freeze_filter, 
                                lambda p: p.replace(p.value.astype(jnp.bfloat16)))
   ```
   - Frozen params converted to bfloat16 to save memory

## How stop_gradient Works

`stop_gradient` operates at the **computation graph level**:

```python
x_stopped = jax.lax.stop_gradient(x)
# Tells JAX: "Don't build gradient computation graph for this value"
```

## Key Difference: Compilation vs Runtime

### What Freeze Filter Does:
- ✅ **Runtime**: Prevents gradient computation for frozen params
- ✅ **Runtime**: Excludes frozen params from optimizer
- ✅ **Runtime**: Saves memory by using bfloat16
- ❌ **Compilation**: Does NOT reduce computation graph size
- ❌ **Compilation**: Forward pass still executes fully
- ❌ **Compilation**: JAX still traces the entire computation structure

### What stop_gradient Does:
- ✅ **Runtime**: Prevents gradient computation (redundant with freeze filter)
- ✅ **Compilation**: Can help XLA optimize graph structure
- ⚠️ **Compilation**: May reduce memory slightly, but limited

## Why Pi0-Dex Still OOMs

The OOM happens during **JIT compilation** (`jax.jit(init, ...)` at line 126):

1. **During compilation**, JAX traces through the model structure
2. **For pi0-base**: ONE forward pass → ONE computation graph to trace
3. **For pi0-dex**: TWO forward passes → TWO computation graphs to trace simultaneously
4. **The freeze filter doesn't help** because:
   - The forward pass still executes (frozen params are still used)
   - JAX still needs to trace the computation structure
   - Even without gradients, the computation graph structure must be materialized

## The Real Problem

Even though:
- Frozen params don't get gradients (via freeze filter)
- Frozen params use less memory (bfloat16)
- Forward pass can use stop_gradient

**During compilation**, JAX/XLA still needs to:
1. Trace through both forward passes
2. Materialize the computation graph structure for both
3. Allocate memory for the traced graph

This is why pi0-dex uses ~43 GB during compilation vs pi0-base's ~20 GB - it's tracing TWO complete LLM forward passes instead of one.

## Conclusion

**Freeze filter** = Training-time optimization (gradients, optimizer, memory)
**stop_gradient** = Computation-time hint (may help XLA optimize)

**Neither solves the compilation memory issue** because the problem is structural: having two forward passes means JAX must trace two computation graphs during compilation, regardless of whether gradients are computed.

The solution requires either:
1. Reducing computation graph size (shorter sequences, smaller models)
2. Avoiding dual forward passes (architectural change)
3. Splitting compilation (compile separately and cache)

