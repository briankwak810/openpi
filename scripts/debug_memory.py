"""Debug script to identify memory bottlenecks in pi0_dex training."""

import os
import functools
import jax
import jax.numpy as jnp
import einops
import flax.nnx as nnx
from openpi.models import pi0_config
import openpi.models.model as _model
from openpi.models.pi0_dex import make_attn_mask

def get_memory_usage():
    """Get current memory usage from JAX."""
    try:
        # Get memory stats from all devices
        devices = jax.devices()
        for i, device in enumerate(devices):
            if hasattr(device, 'memory_stats'):
                stats = device.memory_stats()
                allocated = stats.get('bytes_in_use', 0) / 1e9
                peak = stats.get('bytes_peak', 0) / 1e9
                # Try to get more detailed stats
                live_bytes = stats.get('bytes_live', 0) / 1e9
                reserved = stats.get('bytes_reserved', 0) / 1e9
                print(f"Device {i} ({device}):")
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Reserved: {reserved:.2f} GB")
                print(f"  Peak: {peak:.2f} GB")
                if live_bytes > 0:
                    print(f"  Live: {live_bytes:.2f} GB")
                # Print all available stats keys for debugging
                if i == 0:  # Only print once
                    all_keys = list(stats.keys())
                    if len(all_keys) > 5:
                        print(f"  (Available stats keys: {len(all_keys)} total)")
    except Exception as e:
        print(f"Could not get memory stats: {e}")

def debug_compute_loss():
    """Debug memory usage in compute_loss step by step."""
    print("=" * 80)
    print("Memory Debug: pi0_dex compute_loss")
    print("=" * 80)
    
    # Create model using the config.create method (same as training)
    model_config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m_lora",
        dex=True,  # Enable dex mode
    )
    rng = jax.random.PRNGKey(0)
    model = model_config.create(rng)
    
    # Create dummy observation and actions
    batch_size = 2
    action_horizon = 50
    action_dim = 32
    
    observation = _model.Observation(
        state=jnp.ones((batch_size, 32)),
        images={
            'base_0_rgb': jnp.ones((batch_size, 224, 224, 3)),
            'left_wrist_0_rgb': jnp.ones((batch_size, 224, 224, 3)),
            'right_wrist_0_rgb': jnp.ones((batch_size, 224, 224, 3)),
        },
        image_masks={
            'base_0_rgb': jnp.ones((batch_size,), dtype=bool),
            'left_wrist_0_rgb': jnp.ones((batch_size,), dtype=bool),
            'right_wrist_0_rgb': jnp.ones((batch_size,), dtype=bool),
        },
        tokenized_prompt=jnp.ones((batch_size, 48), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, 48), dtype=bool),
    )
    actions = jnp.ones((batch_size, action_horizon, action_dim))
    
    print("\n1. After model creation:")
    get_memory_usage()
    
    # Test embed_prefix
    print("\n2. After embed_prefix:")
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    print(f"  prefix_tokens shape: {prefix_tokens.shape}")
    get_memory_usage()
    
    # Test embed_suffix
    print("\n3. After embed_suffix:")
    x_t = jnp.ones((batch_size, action_horizon, action_dim))
    time = jnp.ones((batch_size,))
    suffix_output = model.embed_suffix(observation, x_t, time)
    print(f"  suffix tokens shapes: {[x.shape if hasattr(x, 'shape') else type(x) for x in suffix_output]}")
    get_memory_usage()
    
    # Test frozen llm forward
    print("\n4. After frozen llm forward (KV cache):")
    from openpi.models.pi0_dex import make_attn_mask
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    
    _, kv_cache_frozen = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=prefix_positions
    )
    print(f"  KV cache structure: {type(kv_cache_frozen)}")
    if isinstance(kv_cache_frozen, tuple):
        k_cache, v_cache = kv_cache_frozen
        print(f"  K cache shape: {k_cache.shape if hasattr(k_cache, 'shape') else 'N/A'}")
        print(f"  V cache shape: {v_cache.shape if hasattr(v_cache, 'shape') else 'N/A'}")
    get_memory_usage()
    
    # Test KV cache transformation
    print("\n5. After KV cache transformation:")
    kv_cache_transformed = model.transform_kv_cache(kv_cache_frozen)
    get_memory_usage()
    
    # Test full compute_loss (but don't compute gradients)
    print("\n6. After full compute_loss (forward only, no gradients):")
    loss = model.compute_loss(rng, observation, actions, train=False)
    print(f"  Loss shape: {loss.shape}")
    print(f"  Loss mean: {float(jnp.mean(loss)):.6f}")
    print(f"  Loss std: {float(jnp.std(loss)):.6f}")
    print(f"  Loss min: {float(jnp.min(loss)):.6f}")
    print(f"  Loss max: {float(jnp.max(loss)):.6f}")
    get_memory_usage()
    
    # Additional check: verify loss computation details
    print("\n7. Loss computation details:")
    # Recompute to get intermediate values
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    obs = _model.preprocess_observation(preprocess_rng, observation, train=False)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    
    # Get hand_v_t to check its shape
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    _, hand_suffix_tokens, _, hand_suffix_mask, _, hand_suffix_ar_mask, _, hand_adarms_cond = model.embed_suffix(obs, x_t, time)
    
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache_frozen = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)
    kv_cache_frozen = jax.tree.map(jax.lax.stop_gradient, kv_cache_frozen)
    kv_cache_transformed = model.transform_kv_cache(kv_cache_frozen)
    
    hand_suffix_attn_mask = make_attn_mask(hand_suffix_mask, hand_suffix_ar_mask)
    prefix_attn_mask_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=hand_suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, hand_suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(hand_suffix_mask, axis=-1) - 1
    
    # llm_hand now only has one expert (300M), so pass [hand_suffix_tokens] instead of [None, hand_suffix_tokens]
    (hand_suffix_out,), _ = model.PaliGemma.llm_hand(
        [hand_suffix_tokens], 
        mask=full_attn_mask, 
        positions=positions, 
        kv_cache=kv_cache_transformed,
        adarms_cond=[hand_adarms_cond]
    )
    hand_v_t = model.hand_out_proj(hand_suffix_out[:, -model.action_horizon :])
    print(f"  hand_v_t shape: {hand_v_t.shape}")
    print(f"  u_t shape: {u_t.shape}")
    print(f"  hand_v_t - u_t shape: {(hand_v_t - u_t).shape}")
    print(f"  Squared error shape: {jnp.square(hand_v_t - u_t).shape}")
    print(f"  Loss per timestep (mean over action_dim): {jnp.mean(jnp.square(hand_v_t - u_t), axis=-1).shape}")
    get_memory_usage()
    
    print("\n" + "=" * 80)
    print("Memory Debug Complete")
    print("=" * 80)


def debug_compute_loss_with_jit():
    """Debug memory usage in compute_loss with JIT compilation.
    
    This demonstrates various options for debugging memory when jax.jit is used:
    1. Memory before/after compilation
    2. Memory during execution (warmup vs subsequent calls)
    3. Using jax.block_until_ready to ensure computation completes
    4. Device memory profiling with jax.profiler
    """
    print("=" * 80)
    print("Memory Debug: pi0_dex compute_loss WITH JIT")
    print("=" * 80)
    
    # Create model using the config.create method (same as training)
    model_config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m_lora",
        dex=True,  # Enable dex mode
    )
    rng = jax.random.PRNGKey(0)
    model = model_config.create(rng)
    
    # Create dummy observation and actions
    batch_size = 2
    action_horizon = 50
    action_dim = 32
    
    observation = _model.Observation(
        state=jnp.ones((batch_size, 32)),
        images={
            'base_0_rgb': jnp.ones((batch_size, 224, 224, 3)),
            'left_wrist_0_rgb': jnp.ones((batch_size, 224, 224, 3)),
            'right_wrist_0_rgb': jnp.ones((batch_size, 224, 224, 3)),
        },
        image_masks={
            'base_0_rgb': jnp.ones((batch_size,), dtype=bool),
            'left_wrist_0_rgb': jnp.ones((batch_size,), dtype=bool),
            'right_wrist_0_rgb': jnp.ones((batch_size,), dtype=bool),
        },
        tokenized_prompt=jnp.ones((batch_size, 48), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, 48), dtype=bool),
    )
    actions = jnp.ones((batch_size, action_horizon, action_dim))
    
    print("\n1. After model creation (before JIT):")
    get_memory_usage()
    
    # Create a JIT-compiled version of compute_loss
    # Option 1: JIT the entire compute_loss method
    @functools.partial(jax.jit, static_argnames=['train'])
    def jitted_compute_loss(model_state, rng, observation, actions, train=False):
        """JIT-compiled compute_loss wrapper."""
        # Reconstruct model from state
        model = nnx.merge(model_state['graphdef'], model_state['params'])
        return model.compute_loss(rng, observation, actions, train=train)
    
    # Prepare model state for JIT
    graphdef = nnx.graphdef(model)
    params = nnx.state(model)
    model_state = {'graphdef': graphdef, 'params': params}
    
    print("\n2. After preparing model state (before compilation):")
    get_memory_usage()
    
    # Option 2: Measure memory during compilation (first call triggers compilation)
    print("\n3. Compiling JIT function (first call - this may take time)...")
    print("   Memory BEFORE compilation:")
    get_memory_usage()
    
    # First call triggers compilation - this is where memory spikes can occur
    try:
        loss_compiled = jitted_compute_loss(model_state, rng, observation, actions, train=False)
        # Block until compilation completes
        loss_compiled.block_until_ready()
        print("   Memory AFTER compilation (first call):")
        get_memory_usage()
        print(f"   Loss shape: {loss_compiled.shape}")
    except Exception as e:
        print(f"   ERROR during compilation: {e}")
        print("   This is where OOM often occurs during JIT compilation!")
        get_memory_usage()
        return
    
    # Option 3: Measure memory during execution (subsequent calls are faster)
    print("\n4. Executing JIT function (second call - should be faster):")
    print("   Memory BEFORE execution:")
    get_memory_usage()
    
    loss_executed = jitted_compute_loss(model_state, rng, observation, actions, train=False)
    loss_executed.block_until_ready()  # Ensure computation completes
    print("   Memory AFTER execution (second call):")
    get_memory_usage()
    print(f"   Loss mean: {float(jnp.mean(loss_executed)):.6f}")
    
    # Option 4: Multiple executions to see if memory stabilizes
    print("\n5. Multiple executions to check memory stability:")
    for i in range(3):
        loss = jitted_compute_loss(model_state, rng, observation, actions, train=False)
        loss.block_until_ready()
        print(f"   After execution {i+1}:")
        get_memory_usage()
    
    print("\n" + "=" * 80)
    print("JIT Memory Debug Complete")
    print("=" * 80)


def debug_with_device_memory_profiler():
    """Use JAX's device memory profiler to get detailed memory analysis.
    
    This requires:
    1. Installing pprof: pip install pprof
    2. Running with profiling enabled
    3. Analyzing the profile with pprof
    
    Usage:
        # Set environment variable before running
        export JAX_PROFILER_PORT=9999
        python debug_memory.py --profile
    """
    print("=" * 80)
    print("Device Memory Profiling Setup")
    print("=" * 80)
    print("\nTo use device memory profiling:")
    print("1. Install pprof: pip install pprof")
    print("2. Run with profiling enabled:")
    print("   export JAX_PROFILER_PORT=9999")
    print("   python debug_memory.py --profile")
    print("3. Save memory profile:")
    print("   jax.profiler.save_device_memory_profile('memory_profile.pb')")
    print("4. Analyze with pprof:")
    print("   pprof --web memory_profile.pb")
    print("\nAlternatively, use XLA memory profiling:")
    print("   export XLA_FLAGS='--xla_dump_to=/tmp/xla_dump'")
    print("   export TF_CPP_MIN_LOG_LEVEL=0")
    
    # Example of saving a memory profile
    try:
        # jax.profiler may not be available in all JAX versions
        try:
            import jax.profiler
            profile_path = "/tmp/jax_memory_profile.pb"
            print(f"\nSaving memory profile to {profile_path}...")
            jax.profiler.save_device_memory_profile(profile_path)
            print(f"Profile saved! Analyze with: pprof --web {profile_path}")
        except ImportError:
            print("\nNote: jax.profiler not available in this JAX version.")
            print("For device memory profiling, use XLA flags or pprof directly.")
    except Exception as e:
        print(f"Could not save profile (this is OK if pprof not available): {e}")


def diagnose_oom_during_training():
    """Diagnose OOM issues during training by simulating the training step compilation."""
    print("=" * 80)
    print("OOM Diagnosis: Training Step Compilation")
    print("=" * 80)
    
    print("\nThis function simulates what happens during train_step JIT compilation.")
    print("The OOM typically occurs during the FIRST call to ptrain_step (line 243 in train.py)")
    print("because JAX must trace BOTH forward passes simultaneously.\n")
    
    dex = True
    
    # Create model
    model_config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        hand_expert_variant="gemma_300m_lora",
        dex=dex,
    )
    rng = jax.random.PRNGKey(0)
    model = model_config.create(rng)
    
    # Create dummy batch (matching training setup)
    batch_size = 1  # Using batch_size=1 as in config
    action_horizon = model_config.action_horizon
    action_dim = 32
    
    observation = _model.Observation(
        state=jnp.ones((batch_size, 32)),
        images={
            'base_0_rgb': jnp.ones((batch_size, 224, 224, 3)),
            'left_wrist_0_rgb': jnp.ones((batch_size, 224, 224, 3)),
            'right_wrist_0_rgb': jnp.ones((batch_size, 224, 224, 3)),
        },
        image_masks={
            'base_0_rgb': jnp.ones((batch_size,), dtype=bool),
            'left_wrist_0_rgb': jnp.ones((batch_size,), dtype=bool),
            'right_wrist_0_rgb': jnp.ones((batch_size,), dtype=bool),
        },
        tokenized_prompt=jnp.ones((batch_size, 48), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, 48), dtype=bool),
    )
    actions = jnp.ones((batch_size, action_horizon, action_dim))
    
    print("1. Memory before creating JIT-compiled train_step:")
    get_memory_usage()
    
    # Get the freeze filter to match actual training setup
    # This ensures frozen 2B model params are NOT trainable (matching train.py line 157)
    freeze_filter = model_config.get_freeze_filter_dex() if dex else model_config.get_freeze_filter()
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    
    print(f"\nFreeze filter: {freeze_filter}")
    print("This will freeze:")
    print("  - Frozen LLM (2.3B params): PaliGemma/llm/...")
    print("  - Image encoder: PaliGemma/img/...")
    print("  - llm_hand base params (but allow LoRA params)")
    print("\nTrainable params:")
    print("  - LoRA parameters in llm_hand")
    print("  - KV transformation MLPs (if enabled)")
    print("  - Action/hand projections and other trainable components")
    print("\n⚠️  IMPORTANT: Even though frozen params don't get gradients,")
    print("   JAX still traces their forward pass during JIT compilation.")
    print("   This is why OOM occurs - both forward passes must be traced simultaneously.")
    
    # Simulate train_step function (simplified version)
    def train_step_simulated(model_state, rng, observation, actions):
        """Simplified train_step that matches the structure in train.py"""
        model_merged = nnx.merge(model_state['graphdef'], model_state['params'])
        model_merged.train()
        
        def loss_fn(model, rng, obs, acts):
            return jnp.mean(model.compute_loss(rng, obs, acts, train=True))
        
        # This is where the dual forward passes happen
        # During JIT compilation, JAX traces:
        # 1. Frozen LLM forward pass (2.3B params) - NO GRADIENTS computed
        # 2. Hand expert LLM forward pass (300M params) - GRADIENTS computed for trainable params
        # Both simultaneously in the computation graph
        
        # Use trainable_filter to match actual training (train.py line 157)
        # This excludes frozen 2B model params from gradient computation
        diff_state = nnx.DiffState(0, trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
            model_merged, rng, observation, actions
        )
        return loss, grads
    
    # Prepare model state
    graphdef = nnx.graphdef(model)
    params = nnx.state(model)
    model_state = {'graphdef': graphdef, 'params': params}
    
    print("\n2. Memory after preparing model state:")
    get_memory_usage()
    
    # Create JIT-compiled function (this is where OOM happens)
    print("\n3. Creating JIT-compiled train_step (this matches train.py line 243):")
    print("   Memory BEFORE JIT compilation setup:")
    get_memory_usage()
    
    try:
        # This simulates: ptrain_step = jax.jit(functools.partial(train_step, config), ...)
        jitted_train_step = jax.jit(train_step_simulated)
        
        print("   Memory AFTER JIT function creation (before first call):")
        get_memory_usage()
        
        print("\n4. First call to JIT function (triggers compilation):")
        print("   ⚠️  THIS IS WHERE OOM TYPICALLY OCCURS ⚠️")
        print("   JAX must trace BOTH forward passes simultaneously:")
        print("   - Frozen LLM forward (2.3B params)")
        print("   - Hand expert forward (300M params)")
        print("   Memory BEFORE first call:")
        get_memory_usage()
        
        # First call triggers compilation
        loss, grads = jitted_train_step(model_state, rng, observation, actions)
        loss.block_until_ready()
        
        print("   Memory AFTER compilation (first call completed):")
        get_memory_usage()
        print(f"   Loss: {float(loss):.6f}")
        
        print("\n5. Second call (execution, should be faster):")
        print("   Memory BEFORE second call:")
        get_memory_usage()
        
        loss2, grads2 = jitted_train_step(model_state, rng, observation, actions)
        loss2.block_until_ready()
        
        print("   Memory AFTER second call:")
        get_memory_usage()
        
        print("\n✅ If you got here, compilation succeeded!")
        print("   If OOM occurred, it happened during step 4 (first call).")
        
    except Exception as e:
        print(f"\n❌ OOM ERROR during compilation: {e}")
        print("\nThis confirms the OOM occurs during JIT compilation.")
        print("The issue is that JAX must trace BOTH forward passes simultaneously.")
        print("\nSolutions:")
        print("1. Reduce action_horizon or max_token_len")
        print("2. Use XLA memory flags: export XLA_FLAGS='--xla_gpu_memory_fraction=0.8'")
        print("3. Split compilation (compile forward passes separately)")
        print("4. Use gradient accumulation with smaller micro-batches")
        get_memory_usage()
    
    print("\n" + "=" * 80)
    print("OOM Diagnosis Complete")
    print("=" * 80)
    print("\nSee OOM_DIAGNOSIS.md for detailed analysis and solutions.")


def debug_jit_options():
    """Demonstrate different JIT options for memory debugging."""
    print("=" * 80)
    print("JIT Memory Debugging Options")
    print("=" * 80)
    
    print("\n=== OPTION 1: Disable JIT Temporarily ===")
    print("Set environment variable: JAX_DISABLE_JIT=True")
    print("Or in code: jax.config.update('jax_disable_jit', True)")
    print("This allows using standard Python debugging tools.")
    
    print("\n=== OPTION 2: Measure Memory During Compilation ===")
    print("First call to jitted function triggers compilation.")
    print("Use get_memory_usage() before and after first call.")
    print("Memory spikes during compilation are common and expected.")
    
    print("\n=== OPTION 3: Use jax.block_until_ready ===")
    print("Always call .block_until_ready() on JIT results before measuring memory.")
    print("This ensures computation completes before measurement.")
    
    print("\n=== OPTION 4: Device Memory Profiling ===")
    print("Use jax.profiler.save_device_memory_profile() to get detailed profiles.")
    print("Analyze with pprof tool for visualization.")
    
    print("\n=== OPTION 5: XLA Memory Flags ===")
    print("Set XLA_FLAGS='--xla_dump_to=/tmp/xla_dump' to dump XLA graphs.")
    print("Set XLA_FLAGS='--xla_gpu_memory_fraction=0.8' to limit GPU memory.")
    
    print("\n=== OPTION 6: JIT Compilation Callbacks ===")
    print("Use jax.jit with compilation callbacks to monitor compilation:")
    print("  def compilation_callback(module_name, compile_time_secs):")
    print("      print(f'Compiled {module_name} in {compile_time_secs}s')")
    print("  jax.config.update('jax_jit_pjit_api_merge', True)")
    
    print("\n=== OPTION 7: Memory Stats During Execution ===")
    print("Use device.memory_stats() to get detailed stats:")
    print("  - bytes_in_use: Currently allocated memory")
    print("  - bytes_peak: Peak memory usage")
    print("  - bytes_reserved: Reserved memory")
    print("  - bytes_live: Live memory (if available)")
    
    print("\n=== OPTION 8: Checkpointing Inside JIT ===")
    print("Use jax.checkpoint inside JIT-compiled functions to reduce memory.")
    print("This trades computation for memory (recomputes instead of storing).")
    
    print("\n=== OPTION 9: Donate Arguments ===")
    print("Use donate_argnums in jax.jit to reuse input buffers:")
    print("  jax.jit(fn, donate_argnums=(0,))  # Donate first argument")
    print("This reduces peak memory by reusing input memory for outputs.")
    
    print("\n=== OPTION 10: Static Arguments ===")
    print("Use static_argnums/static_argnames to avoid recompilation:")
    print("  jax.jit(fn, static_argnames=['train'])")
    print("This can reduce compilation memory if done correctly.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if "--jit" in sys.argv:
            debug_compute_loss_with_jit()
        elif "--profile" in sys.argv:
            debug_with_device_memory_profiler()
        elif "--options" in sys.argv:
            debug_jit_options()
        elif "--diagnose" in sys.argv or "--oom" in sys.argv:
            diagnose_oom_during_training()
        else:
            print("Usage:")
            print("  python debug_memory.py          # Original debug (no JIT)")
            print("  python debug_memory.py --jit     # Debug with JIT")
            print("  python debug_memory.py --diagnose # Diagnose OOM during training")
            print("  python debug_memory.py --profile # Device memory profiling")
            print("  python debug_memory.py --options # Show all JIT debugging options")
    else:
        # Original behavior
        debug_compute_loss()

