"""
FSDP2 (composable fully-sharded data parallel) utilities for the Helios transformer.

Memory lifecycle per transformer block during a forward pass:
  1. FSDP2 pre-hook: all-gather this block's sharded params → each rank has full params
  2. Diffusers CP pre-hook: split hidden_states by cp_size (Ulysses sequence split)
  3. Block forward: full params + local sequence shard; Ulysses all-to-all inside attn
  4. Diffusers CP post-hook: nothing (split_output=False, gather done in transformer.forward)
  5. FSDP2 post-hook: re-shard (free) this block's params → back to 1/N params per rank

FSDP2 all-gather (block-granularity, before block forward) and Ulysses all-to-all
(head-granularity, inside attention) are strictly sequential → no communication conflict.

Usage:
    # Apply BEFORE enable_parallelism() so FSDP2 hooks register first.
    apply_fsdp2_to_transformer(pipe.transformer, param_dtype=weight_dtype)
    pipe.transformer.enable_parallelism(config=cp_config)
"""

import torch
import torch.distributed as dist

try:
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

    _FSDP2_AVAILABLE = True
except ImportError:
    _FSDP2_AVAILABLE = False


def apply_fsdp2_to_transformer(
    transformer,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    reshard_after_forward: bool = True,
) -> None:
    """Apply FSDP2 to the Helios transformer in-place for memory-efficient inference.

    Each HeliosTransformerBlock is wrapped independently so its parameters are
    materialized (all-gathered) one block at a time.  Between block invocations
    each GPU holds only 1/world_size of every block's parameters.

    The root module (patch embeddings, condition embedder, norm_out, proj_out) is
    also wrapped; those params are all-gathered at the start of transformer.forward()
    and freed at the end.

    Args:
        transformer: HeliosTransformer3DModel already moved to the target CUDA device.
        param_dtype: dtype in which sharded params are stored and communicated.
        reduce_dtype: dtype for gradient reduction (unused in inference).
        reshard_after_forward: Whether to free each module's params after its forward.
                               True (default) minimises per-GPU peak param memory.
    """
    if not _FSDP2_AVAILABLE:
        raise RuntimeError(
            "FSDP2 not found.  Requires PyTorch >= 2.2: "
            "torch.distributed._composable.fsdp.fully_shard"
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialised before apply_fsdp2_to_transformer()."
        )

    # FSDP2 requires uniform original parameter dtype across all sharded params.
    # Some params (norms, scale_shift_table, time_embedder) are kept in fp32 by
    # diffusers' _keep_in_fp32_modules.  Cast them to param_dtype here.
    # fp32 *computation* inside those modules is safe: FP32LayerNorm / FP32_RMSNorm
    # explicitly cast inputs via .float() inside their forward, independent of
    # the stored param dtype.
    for p in transformer.parameters():
        if p.dtype != param_dtype:
            p.data = p.data.to(param_dtype)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )

    # --- Inner: shard each transformer block independently ---
    # With reshard_after_forward=True each block's params are freed immediately
    # after its forward, so peak GPU memory holds at most one block's full params
    # at any given moment during the 40-block loop.
    for block in transformer.blocks:
        fully_shard(block, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)

    # --- Outer: shard the root to cover remaining params ---
    # (patch_embedding, patch_short/mid/long, condition_embedder, norm_out, proj_out)
    # These are all-gathered once at the start of transformer.forward() and freed at
    # the end, so they stay available throughout the embedding + block loop + output.
    fully_shard(transformer, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)
