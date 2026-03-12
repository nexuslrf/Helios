# Copyright 2025 The Helios Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HeliosVACETransformer3DModel – VACE-conditioned variant of HeliosTransformer3DModel.

Architecture mirrors WanVACETransformer3DModel but is adapted for Helios:
  - Uses HeliosAttention / HeliosAttnProcessor (not WanAttention).
  - Per-token 4-D timestep embedding [B, seq_len, 6, inner_dim] (Helios) vs.
    3-D [B, 6, inner_dim] (WAN).
  - HeliosRotaryPosEmbed takes explicit frame_indices rather than hidden_states.
  - VACE hints are applied only to the *current-chunk* tokens; history tokens
    (prepended by HeliosTransformer3DModel.forward) are left untouched.
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import apply_lora_scale, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph

from .transformer_helios_diffusers import (
    HeliosAttention,
    HeliosAttnProcessor,
    HeliosOutputNorm,
    HeliosRotaryPosEmbed,
    HeliosTimeTextEmbedding,
    HeliosTransformerBlock,
    center_down_sample_3d,
    pad_for_3d_conv,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ---------------------------------------------------------------------------
# Checkpoint key remapping: Wan2.1-VACE → this model's state-dict naming
# ---------------------------------------------------------------------------

def _remap_vace_module_key(key: str) -> str | None:
    """Map a key from the Wan2.1-VACE-module checkpoint to this model's state-dict.

    Checkpoint convention            → Model convention
    ─────────────────────────────────────────────────────
    vace_patch_embedding.*           → vace_patch_embedding.*
    vace_blocks.N.before_proj.*      → vace_blocks.N.proj_in.*
    vace_blocks.N.after_proj.*       → vace_blocks.N.proj_out.*
    vace_blocks.N.self_attn.q.*      → vace_blocks.N.attn1.to_q.*
    vace_blocks.N.self_attn.k.*      → vace_blocks.N.attn1.to_k.*
    vace_blocks.N.self_attn.v.*      → vace_blocks.N.attn1.to_v.*
    vace_blocks.N.self_attn.o.*      → vace_blocks.N.attn1.to_out.0.*
    vace_blocks.N.self_attn.norm_q.* → vace_blocks.N.attn1.norm_q.*
    vace_blocks.N.self_attn.norm_k.* → vace_blocks.N.attn1.norm_k.*
    vace_blocks.N.cross_attn.*       → vace_blocks.N.attn2.*  (same sub-keys)
    vace_blocks.N.ffn.0.*            → vace_blocks.N.ffn.net.0.proj.*
    vace_blocks.N.ffn.2.*            → vace_blocks.N.ffn.net.2.*
    vace_blocks.N.norm3.*            → vace_blocks.N.norm3.*
    vace_blocks.N.modulation         → vace_blocks.N.scale_shift_table
    model_type.*                     → (skip)
    """
    if key.startswith("model_type."):
        return None

    if key.startswith("vace_patch_embedding."):
        return key  # identical naming

    parts = key.split(".", 2)
    if len(parts) != 3 or parts[0] != "vace_blocks":
        return None

    n, rest = parts[1], parts[2]
    p = f"vace_blocks.{n}."

    _PROJ = {"q": "to_q", "k": "to_k", "v": "to_v"}

    if rest.startswith("before_proj."):
        return p + "proj_in." + rest[len("before_proj."):]
    if rest.startswith("after_proj."):
        return p + "proj_out." + rest[len("after_proj."):]

    for attn_src, attn_dst in (("self_attn", "attn1"), ("cross_attn", "attn2")):
        if rest.startswith(attn_src + "."):
            sub = rest[len(attn_src) + 1:]
            first = sub.split(".")[0]
            tail = sub[len(first) + 1:]
            if first in _PROJ:
                return p + f"{attn_dst}.{_PROJ[first]}.{tail}"
            if first == "o":
                return p + f"{attn_dst}.to_out.0.{tail}"
            if first in ("norm_q", "norm_k"):
                return p + f"{attn_dst}.{first}.{tail}"

    if rest.startswith("ffn.0."):
        return p + "ffn.net.0.proj." + rest[len("ffn.0."):]
    if rest.startswith("ffn.2."):
        return p + "ffn.net.2." + rest[len("ffn.2."):]
    if rest.startswith("norm3."):
        return p + rest
    if rest == "modulation":
        return p + "scale_shift_table"

    return None  # unknown key, skip


# ---------------------------------------------------------------------------
# VACE transformer block (Helios-flavour)
# ---------------------------------------------------------------------------

@maybe_allow_in_graph
class HeliosVACETransformerBlock(nn.Module):
    """A single VACE processing block for the Helios transformer.

    Mirrors WanVACETransformerBlock but handles:
      - HeliosAttention (not WanAttention).
      - 4-D per-token temb: [B, seq_len, 6, inner_dim] (chunked on dim 2).
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        apply_input_projection: bool = False,
        apply_output_projection: bool = False,
    ):
        super().__init__()

        # 1. Optional input projection (layer 0 only)
        self.proj_in = nn.Linear(dim, dim) if apply_input_projection else None

        # 2. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = HeliosAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            processor=HeliosAttnProcessor(),
            is_amplify_history=False,
        )

        # 3. Cross-attention
        self.attn2 = HeliosAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=HeliosAttnProcessor(),
            is_cross_attention=True,
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        # elementwise_affine=True: checkpoint stores norm3.weight + norm3.bias
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 5. Optional output projection
        self.proj_out = nn.Linear(dim, dim) if apply_output_projection else None

        # 6. Scale-shift parameters (same layout as HeliosTransformerBlock)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,            # main stream (unused here, kept for API parity)
        encoder_hidden_states: torch.Tensor,    # text embeddings
        control_hidden_states: torch.Tensor,    # VACE control stream
        temb: torch.Tensor,                     # [B, seq_len, 6, inner_dim]  (4-D)
        rotary_emb: torch.Tensor,               # RoPE for current chunk
    ):
        # --- optional input projection (fuses main stream into control) ---
        if self.proj_in is not None:
            control_hidden_states = self.proj_in(control_hidden_states) + hidden_states

        # --- extract scale/shift from 4-D temb ---
        # temb: [B, seq_len, 6, inner_dim]  →  chunk on dim 2
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # Fallback: WAN-style 3-D temb [B, 6, inner_dim]
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention on control stream
        norm_ctrl = (self.norm1(control_hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(
            control_hidden_states
        )
        attn_output = self.attn1(norm_ctrl, None, None, rotary_emb, None)
        control_hidden_states = (control_hidden_states.float() + attn_output * gate_msa).type_as(
            control_hidden_states
        )

        # 2. Cross-attention to text
        norm_ctrl = self.norm2(control_hidden_states.float()).type_as(control_hidden_states)
        attn_output = self.attn2(norm_ctrl, encoder_hidden_states, None, None, None)
        control_hidden_states = control_hidden_states + attn_output

        # 3. Feed-forward
        norm_ctrl = (self.norm3(control_hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            control_hidden_states
        )
        ff_output = self.ffn(norm_ctrl)
        control_hidden_states = (control_hidden_states.float() + ff_output.float() * c_gate_msa).type_as(
            control_hidden_states
        )

        # 4. Optional output projection → conditioning hint
        conditioning_states = None
        if self.proj_out is not None:
            conditioning_states = self.proj_out(control_hidden_states)

        return conditioning_states, control_hidden_states


# ---------------------------------------------------------------------------
# Full VACE transformer model
# ---------------------------------------------------------------------------

class HeliosVACETransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    """HeliosTransformer3DModel augmented with a parallel VACE control stream.

    The VACE stream processes 96-channel conditioning latents
    (inactive_16ch + reactive_16ch + mask_64ch) and injects additive hints
    into the *current-chunk* portion of the main hidden states at selected
    transformer layers.

    History tokens prepended by the normal Helios forward pass are left
    untouched – VACE conditioning is chunk-local.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding",
        "vace_patch_embedding",
        "patch_short",
        "patch_mid",
        "patch_long",
        "condition_embedder",
        "norm",
    ]
    _no_split_modules = ["HeliosTransformerBlock", "HeliosVACETransformerBlock", "HeliosOutputNorm"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "history_key_scale",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["HeliosTransformerBlock", "HeliosVACETransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        rope_dim: tuple[int, ...] = (44, 42, 42),
        rope_theta: float = 10000.0,
        guidance_cross_attn: bool = True,
        zero_history_timestep: bool = True,
        has_multi_term_memory_patch: bool = True,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",
        # VACE-specific
        vace_layers: list[int] = [0, 5, 10, 15, 20, 25, 30, 35],
        vace_in_channels: int = 96,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        if max(vace_layers) >= num_layers:
            raise ValueError(
                f"vace_layers {vace_layers} exceed the number of transformer layers {num_layers}."
            )
        if 0 not in vace_layers:
            raise ValueError("vace_layers must include layer 0.")

        # ------------------------------------------------------------------ #
        # 1. Patch & position embeddings  (identical to HeliosTransformer3DModel)
        # ------------------------------------------------------------------ #
        self.rope = HeliosRotaryPosEmbed(rope_dim=rope_dim, theta=rope_theta)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # VACE conditioning patch embedding: 96ch → inner_dim
        self.vace_patch_embedding = nn.Conv3d(vace_in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # ------------------------------------------------------------------ #
        # 2. Multi-term memory patches  (same as base model)
        # ------------------------------------------------------------------ #
        self.zero_history_timestep = zero_history_timestep
        self.inner_dim = inner_dim
        if has_multi_term_memory_patch:
            self.patch_short = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
            self.patch_mid = nn.Conv3d(
                in_channels,
                inner_dim,
                kernel_size=tuple(2 * p for p in patch_size),
                stride=tuple(2 * p for p in patch_size),
            )
            self.patch_long = nn.Conv3d(
                in_channels,
                inner_dim,
                kernel_size=tuple(4 * p for p in patch_size),
                stride=tuple(4 * p for p in patch_size),
            )

        # ------------------------------------------------------------------ #
        # 3. Condition embeddings
        # ------------------------------------------------------------------ #
        self.condition_embedder = HeliosTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
        )

        # ------------------------------------------------------------------ #
        # 4. Main transformer blocks  (same as base model)
        # ------------------------------------------------------------------ #
        self.blocks = nn.ModuleList(
            [
                HeliosTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    guidance_cross_attn=guidance_cross_attn,
                    is_amplify_history=is_amplify_history,
                    history_scale_mode=history_scale_mode,
                )
                for _ in range(num_layers)
            ]
        )

        # ------------------------------------------------------------------ #
        # 5. VACE blocks (one per vace_layer)
        # ------------------------------------------------------------------ #
        self.vace_blocks = nn.ModuleList(
            [
                HeliosVACETransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    apply_input_projection=(i == 0),   # layer-0 fuses main stream
                    apply_output_projection=True,       # all layers emit a hint
                )
                for i in range(len(vace_layers))
            ]
        )

        # ------------------------------------------------------------------ #
        # 6. Output norm & projection
        # ------------------------------------------------------------------ #
        self.norm_out = HeliosOutputNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))

        self.gradient_checkpointing = False

    # ---------------------------------------------------------------------- #
    # VACE module weight loader
    # ---------------------------------------------------------------------- #

    def load_vace_module(self, path: str) -> "HeliosVACETransformer3DModel":
        """Load VACE-specific weights from a Wan2.1-VACE-module safetensors file.

        The checkpoint contains only vace_patch_embedding and vace_blocks keys
        using Wan naming conventions.  This method remaps them to match this
        model's state-dict and loads them with strict=False (base model weights
        are left untouched).

        Args:
            path: Path to the .safetensors file
                  (e.g. ``BestWishYsh/Wan2_1-VACE_module_14B_bf16.safetensors``).

        Returns:
            self (for chaining).
        """
        from safetensors import safe_open

        state = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                mapped = _remap_vace_module_key(key)
                if mapped is not None:
                    state[mapped] = f.get_tensor(key)

        missing, unexpected = self.load_state_dict(state, strict=False)
        vace_missing = [k for k in missing if k.startswith(("vace_patch_embedding", "vace_blocks"))]
        if vace_missing:
            logger.warning(f"load_vace_module: missing VACE keys: {vace_missing}")
        if unexpected:
            logger.warning(f"load_vace_module: unexpected keys: {unexpected}")
        logger.info(f"Loaded {len(state)} VACE module tensors from {path}")
        return self

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        # ---- history latents (multi-scale memory) ----
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        # ---- VACE control ----
        control_hidden_states: torch.Tensor | None = None,   # [B, 96, T, H, W]
        control_hidden_states_scale: float | torch.Tensor = 1.0,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # ------------------------------------------------------------------ #
        # Replicate the base HeliosTransformer3DModel forward up to block loop
        # ------------------------------------------------------------------ #
        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.config.patch_size

        # 2. Patch-embed current chunk
        hidden_states = self.patch_embedding(hidden_states)
        _, _, post_patch_num_frames, post_patch_height, post_patch_width = hidden_states.shape

        if indices_hidden_states is None:
            indices_hidden_states = (
                torch.arange(0, post_patch_num_frames).unsqueeze(0).expand(batch_size, -1)
            )

        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        rotary_emb = self.rope(
            frame_indices=indices_hidden_states,
            height=post_patch_height,
            width=post_patch_width,
            device=hidden_states.device,
        )
        rotary_emb = rotary_emb.flatten(2).transpose(1, 2)
        original_context_length = hidden_states.shape[1]

        # 3. Short history
        if latents_history_short is not None and indices_latents_history_short is not None:
            latents_history_short = latents_history_short.to(hidden_states)
            latents_history_short = self.patch_short(latents_history_short)
            _, _, _, H1, W1 = latents_history_short.shape
            latents_history_short = latents_history_short.flatten(2).transpose(1, 2)

            rotary_emb_short = self.rope(
                frame_indices=indices_latents_history_short,
                height=H1,
                width=W1,
                device=latents_history_short.device,
            )
            rotary_emb_short = rotary_emb_short.flatten(2).transpose(1, 2)
            hidden_states = torch.cat([latents_history_short, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_short, rotary_emb], dim=1)

        # 4. Mid history
        if latents_history_mid is not None and indices_latents_history_mid is not None:
            latents_history_mid = latents_history_mid.to(hidden_states)
            latents_history_mid = pad_for_3d_conv(latents_history_mid, (2, 4, 4))
            latents_history_mid = self.patch_mid(latents_history_mid)
            latents_history_mid = latents_history_mid.flatten(2).transpose(1, 2)

            rotary_emb_mid = self.rope(
                frame_indices=indices_latents_history_mid,
                height=H1,
                width=W1,
                device=latents_history_mid.device,
            )
            rotary_emb_mid = pad_for_3d_conv(rotary_emb_mid, (2, 2, 2))
            rotary_emb_mid = center_down_sample_3d(rotary_emb_mid, (2, 2, 2))
            rotary_emb_mid = rotary_emb_mid.flatten(2).transpose(1, 2)
            hidden_states = torch.cat([latents_history_mid, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_mid, rotary_emb], dim=1)

        # 5. Long history
        if latents_history_long is not None and indices_latents_history_long is not None:
            latents_history_long = latents_history_long.to(hidden_states)
            latents_history_long = pad_for_3d_conv(latents_history_long, (4, 8, 8))
            latents_history_long = self.patch_long(latents_history_long)
            latents_history_long = latents_history_long.flatten(2).transpose(1, 2)

            rotary_emb_long = self.rope(
                frame_indices=indices_latents_history_long,
                height=H1,
                width=W1,
                device=latents_history_long.device,
            )
            rotary_emb_long = pad_for_3d_conv(rotary_emb_long, (4, 4, 4))
            rotary_emb_long = center_down_sample_3d(rotary_emb_long, (4, 4, 4))
            rotary_emb_long = rotary_emb_long.flatten(2).transpose(1, 2)
            hidden_states = torch.cat([latents_history_long, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_long, rotary_emb], dim=1)

        history_context_length = hidden_states.shape[1] - original_context_length

        # 6. Timestep/text embeddings
        if indices_hidden_states is not None and self.zero_history_timestep:
            timestep_t0 = torch.zeros((1,), dtype=timestep.dtype, device=timestep.device)
            temb_t0, timestep_proj_t0, _ = self.condition_embedder(
                timestep_t0, encoder_hidden_states, is_return_encoder_hidden_states=False
            )
            temb_t0 = temb_t0.unsqueeze(1).expand(batch_size, history_context_length, -1)
            timestep_proj_t0 = (
                timestep_proj_t0.unflatten(-1, (6, -1))
                .view(1, 6, 1, -1)
                .expand(batch_size, -1, history_context_length, -1)
            )

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)
        timestep_proj = timestep_proj.unflatten(-1, (6, -1))

        if indices_hidden_states is not None and not self.zero_history_timestep:
            main_repeat_size = hidden_states.shape[1]
        else:
            main_repeat_size = original_context_length
        temb = temb.view(batch_size, 1, -1).expand(batch_size, main_repeat_size, -1)
        timestep_proj = timestep_proj.view(batch_size, 6, 1, -1).expand(batch_size, 6, main_repeat_size, -1)

        if indices_hidden_states is not None and self.zero_history_timestep:
            temb = torch.cat([temb_t0, temb], dim=1)
            timestep_proj = torch.cat([timestep_proj_t0, timestep_proj], dim=2)

        if timestep_proj.ndim == 4:
            timestep_proj = timestep_proj.permute(0, 2, 1, 3)

        # ------------------------------------------------------------------ #
        # 7. VACE: process control stream → collect hints
        # ------------------------------------------------------------------ #
        vace_hints = []   # list of (hint_tensor, scale) in forward order
        if control_hidden_states is not None:
            # Resize control latents to the current pyramid stage resolution.
            # In stage2_sample the latents are spatially downsampled; the control
            # is always prepared at the full latent resolution, so we must match.
            target_T = post_patch_num_frames * p_t
            target_H = post_patch_height * p_h
            target_W = post_patch_width * p_w
            if control_hidden_states.shape[2:] != torch.Size([target_T, target_H, target_W]):
                control_hidden_states = F.interpolate(
                    control_hidden_states.float(),
                    size=(target_T, target_H, target_W),
                    mode="trilinear",
                    align_corners=False,
                ).to(control_hidden_states.dtype)

            # Embed 96-ch control latents (now at the correct pyramid resolution)
            ctrl = self.vace_patch_embedding(control_hidden_states)  # [B, inner_dim, T', H', W']
            ctrl = ctrl.flatten(2).transpose(1, 2)  # [B, ctrl_seq_len, inner_dim]

            # Pad control tokens to match original_context_length (handles short last chunk)
            if ctrl.shape[1] < original_context_length:
                padding = ctrl.new_zeros(batch_size, original_context_length - ctrl.shape[1], ctrl.shape[2])
                ctrl = torch.cat([ctrl, padding], dim=1)

            # RoPE for control tokens (same frame_indices as current chunk)
            ctrl_rotary = self.rope(
                frame_indices=indices_hidden_states,
                height=post_patch_height,
                width=post_patch_width,
                device=ctrl.device,
            )
            ctrl_rotary = ctrl_rotary.flatten(2).transpose(1, 2)

            # temb for the current chunk only (last original_context_length tokens)
            temb_chunk = timestep_proj[:, -original_context_length:, :, :]  # [B, orig_len, 6, inner_dim]

            # Current-chunk main hidden states (for proj_in fusion in block 0)
            chunk_hidden = hidden_states[:, -original_context_length:]

            for i, vace_block in enumerate(self.vace_blocks):
                hint, ctrl = vace_block(
                    chunk_hidden,
                    encoder_hidden_states,
                    ctrl,
                    temb_chunk,
                    ctrl_rotary,
                )
                # Scale: scalar or per-layer tensor
                if isinstance(control_hidden_states_scale, torch.Tensor):
                    scale = control_hidden_states_scale[i]
                else:
                    scale = control_hidden_states_scale
                vace_hints.append((hint, scale))

            vace_hints = vace_hints[::-1]   # reverse so we can pop() in order

        # ------------------------------------------------------------------ #
        # 8. Main transformer blocks (same logic as base model + hint injection)
        # ------------------------------------------------------------------ #
        vace_layers_set = set(self.config.vace_layers)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()

        for i, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                )

            if vace_hints and i in vace_layers_set:
                hint, scale = vace_hints.pop()
                if hint is not None:
                    # Apply hint only to current-chunk tokens
                    hidden_states[:, -original_context_length:] = (
                        hidden_states[:, -original_context_length:] + hint * scale
                    )

        # ------------------------------------------------------------------ #
        # 9. Output norm, projection, unpatchify
        # ------------------------------------------------------------------ #
        hidden_states = self.norm_out(hidden_states, temb, original_context_length)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
