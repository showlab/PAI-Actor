# Channel-concat variant of CustomWanTransformer3DModel for PAIActor.
#
# Mirrors channel-concat editing's split:
#   - input / source video (encoder_condition_states): CHANNEL concat onto the
#     noisy latent  ->  x = torch.cat([x, condition], dim=1)  (the reference design wan_video_dit.py:369).
#   - reference (encoder_ref_states): UNCHANGED in-context TOKEN concat, exactly
#     like the original PAIActor (the reference design also injects reference as token concat,
#     wan_video_mllm.py:526-544).
#
# patch_embedding is widened in_channels -> 2*in_channels (48 -> 96): the original
# pretrained weights go into the first in_channels input slots, the new condition
# slots are zero-init (expand_patch_embedding_inplace()), so at init the condition
# contributes nothing and the model is numerically identical to the base model.
#
# The reference latent is zero-padded on the channel dim up to 96 (its condition
# half = zeros) so it shares the widened patch_embedding; because the first
# in_channels weights are the pretrained ones, reference patchification is
# numerically identical to the base model -> reference behaviour is untouched.
#
# This file is new and modifies no existing file.

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from models.wan2.transformer_wan import CustomWanTransformer3DModel


class ChannelCatWanTransformer3DModel(CustomWanTransformer3DModel):
    """input video via channel concat; reference via in-context token concat."""

    @torch.no_grad()
    def expand_patch_embedding_inplace(self):
        """Widen patch_embedding from `in_channels` to `2 * in_channels` inputs.

        Copies the pretrained weights into the first `in_channels` input slots and
        zero-inits the new (condition) slots. Idempotent.

        `config.in_channels` is intentionally left untouched (=48), so the
        inference pipeline still prepares 48-channel latents; only the conv module
        is widened to accept the [noisy ; condition] channel concat.
        """
        old = self.patch_embedding
        base_in = self.config.in_channels
        target_in = 2 * base_in
        if old.in_channels == target_in:
            return  # already expanded
        new = nn.Conv3d(
            target_in,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
        ).to(device=old.weight.device, dtype=old.weight.dtype)
        nn.init.zeros_(new.weight)
        new.weight[:, :old.in_channels].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
        self.patch_embedding = new

    def _pad_channels(self, x: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Zero-pad x on the channel dim up to target_channels (condition-half = 0)."""
        if x.shape[1] == target_channels:
            return x
        pad = target_channels - x.shape[1]
        zeros = x.new_zeros(x.shape[0], pad, *x.shape[2:])
        return torch.cat([x, zeros], dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
            encoder_condition_states = attention_kwargs.pop("encoder_condition_states", None)
            encoder_ref_states = attention_kwargs.pop("encoder_ref_states", None)
            use_collaborative_position_encoding = attention_kwargs.pop("use_collaborative_position_encoding", False)
        else:
            lora_scale = 1.0
            encoder_condition_states = None
            encoder_ref_states = None
            use_collaborative_position_encoding = False

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # ------------------------------------------------------------------
        # input video (condition) -> CHANNEL concat onto the noisy latent
        # ------------------------------------------------------------------
        if encoder_condition_states is not None:
            main = torch.cat([hidden_states, encoder_condition_states], dim=1)
        elif self.patch_embedding.in_channels == 2 * self.config.in_channels:
            # widened conv but no condition supplied: pad condition-half with zeros
            main = torch.cat([hidden_states, torch.zeros_like(hidden_states)], dim=1)
        else:
            main = hidden_states

        # ------------------------------------------------------------------
        # reference -> UNCHANGED in-context TOKEN concat (channel-pad to match
        # `main` so it shares the widened patch_embedding; pretrained first
        # in_channels weights make its patchification identical to base).
        # ------------------------------------------------------------------
        if encoder_ref_states is not None:  # one2three (reference control)
            num_refs = encoder_ref_states.shape[2]
            extra_tokens = True
            fake_nums = num_refs + post_patch_num_frames
            ref_in = self._pad_channels(encoder_ref_states, main.shape[1])

            if use_collaborative_position_encoding:
                # independent (collaborative) RoPE for ref and target. Target RoPE is
                # computed on the noisy latent (channel-agnostic) -> identical to the
                # original; reference RoPE untouched.
                rotary_emb_ref = self.rope(encoder_ref_states)
                rotary_emb_hidden = self.rope(hidden_states)
                rotary_emb = (
                    torch.cat([rotary_emb_ref[0], rotary_emb_hidden[0]], dim=-2),
                    torch.cat([rotary_emb_ref[1], rotary_emb_hidden[1]], dim=-2),
                )
                ref_tokens = self.patch_embedding(ref_in).flatten(2).transpose(1, 2)
                main_tokens = self.patch_embedding(main).flatten(2).transpose(1, 2)
                hidden_states = torch.cat([ref_tokens, main_tokens], dim=1)
            else:
                stacked = torch.cat([ref_in, main], dim=2)  # temporal (token) concat
                rotary_emb = self.rope(stacked)
                hidden_states = self.patch_embedding(stacked).flatten(2).transpose(1, 2)

        else:  # no reference
            extra_tokens = False
            fake_nums = post_patch_num_frames
            rotary_emb = self.rope(hidden_states)  # noisy grid (channel-agnostic)
            hidden_states = self.patch_embedding(main).flatten(2).transpose(1, 2)

        num_tokens = hidden_states.shape[1]

        # timestep: scalar (batch,) or per-token (batch, seq_len). The shared
        # CustomWanPipeline prepends zeros for the (original) in-context CONDITION
        # tokens; since the condition is now in channels, keep only the trailing
        # `num_tokens` (all leading condition/ref slots are zeros -> correct by value).
        if timestep.ndim == 2:
            if timestep.shape[1] != num_tokens:
                timestep = timestep[:, -num_tokens:]
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        attn_mask = None

        # Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attn_mask
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attn_mask)

        # Output norm & projection
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # drop the in-context reference tokens, keep the target tokens (same as base)
        if extra_tokens:
            hidden_states = hidden_states[:, -hidden_states.shape[-2] // fake_nums * post_patch_num_frames:]

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
