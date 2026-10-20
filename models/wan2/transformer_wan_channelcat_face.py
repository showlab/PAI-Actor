# Channel-concat + FACE-ADAPTER variant for PAIActor.
#
# Builds on ChannelCatWanTransformer3DModel (input video -> channel concat,
# reference -> in-context token concat) and ADDS Wan-Animate's expression
# injection on top:
#
#   face crops (src_face*.mp4, 512x512, frame-aligned to the TARGET video)
#     -> motion_encoder (LIA, FROZEN, pretrained from Wan2.2-Animate-14B)   [B*N,T,512]
#     -> face_encoder (trainable, hidden_dim = 5B inner_dim = 3072)         [B*N,T',5,C]
#     -> FaceAdapter cross-attention into the MAIN (target) tokens, every
#        `face_inject_every` blocks; per-person spatial mask routes each
#        person's tokens (multi-person);  single person == N=1.
#
# At init the adapter output proj (linear2) is ZERO -> the face branch contributes
# nothing, so the model is numerically identical to the channel-concat base until
# the adapter learns. motion_encoder is frozen and reloaded from a small .pth at
# inference (NOT saved in the trainable checkpoint).
#
# This file is new and modifies no existing file.

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from models.wan2.transformer_wan_channelcat import ChannelCatWanTransformer3DModel
from models.wan2.face_inject.motion_encoder import Generator
from models.wan2.face_inject.face_blocks import FaceEncoder, FaceAdapter


class ChannelCatFaceWanTransformer3DModel(ChannelCatWanTransformer3DModel):
    """channel-concat condition + in-context ref + Wan-Animate face adapter."""

    def add_face_modules(self, face_inject_every: int = 5, motion_dim: int = 20,
                         motion_heads: int = 4):
        """Build motion_encoder (frozen) + face_encoder + face_adapter sized for
        this model's inner_dim. Call AFTER from_pretrained + expand_patch_embedding.
        Idempotent."""
        if getattr(self, "_face_ready", False):
            return
        heads = self.config.num_attention_heads
        head_dim = self.config.attention_head_dim
        inner_dim = heads * head_dim
        num_layers = self.config.num_layers

        self.face_inject_every = face_inject_every
        self.num_face_adapters = num_layers // face_inject_every

        # LIA expression extractor (frozen, pretrained). model-size independent.
        self.motion_encoder = Generator(size=512, style_dim=512, motion_dim=motion_dim)
        self.motion_encoder.requires_grad_(False)
        self.motion_encoder.eval()

        # trainable face encoder + adapter, sized to THIS model (3072 / 24 heads)
        self.face_encoder = FaceEncoder(in_dim=512, hidden_dim=inner_dim, num_heads=motion_heads)
        self.face_adapter = FaceAdapter(
            hidden_dim=inner_dim, heads_num=heads, num_adapter_layers=self.num_face_adapters,
        )
        # zero-init the adapter output projection -> zero contribution at init
        for fb in self.face_adapter.fuser_blocks:
            nn.init.zeros_(fb.linear2.weight)
            nn.init.zeros_(fb.linear2.bias)

        self.motion_encode_bs = 8
        self._face_ready = True

    def load_motion_encoder(self, path: str, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        missing, unexpected = self.motion_encoder.load_state_dict(sd, strict=False)
        self.motion_encoder.requires_grad_(False)
        self.motion_encoder.eval()
        return missing, unexpected

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_motion(self, face_pixel_values):
        """face_pixel_values: [B, N, 3, T, 512, 512] in [-1,1].
        Returns motion latents [B, N, T, 512] (frozen LIA, no grad)."""
        B, N, C, T, H, W = face_pixel_values.shape
        me_dtype = next(self.motion_encoder.parameters()).dtype
        flat = face_pixel_values.reshape(B * N * T, C, H, W).to(me_dtype)
        outs = []
        for i in range(0, flat.shape[0], self.motion_encode_bs):
            outs.append(self.motion_encoder.get_motion(flat[i:i + self.motion_encode_bs]))
        motion = torch.cat(outs, dim=0).reshape(B, N, T, -1)
        return motion

    def _build_motion_tokens(self, face_pixel_values, t_lat, dtype):
        """-> motion tokens [B, N, t_lat, n_motion_tokens, C] aligned to latent frames."""
        B, N = face_pixel_values.shape[:2]
        motion = self._encode_motion(face_pixel_values)            # [B,N,T,512]
        motion = motion.reshape(B * N, motion.shape[2], 512).to(dtype)
        tok = self.face_encoder(motion)                            # [B*N, T', n, C]
        Tp, n, Cc = tok.shape[1], tok.shape[2], tok.shape[3]
        if Tp != t_lat:                                            # align temporal -> t_lat
            m = tok.permute(0, 2, 3, 1).reshape(B * N, n * Cc, Tp)
            m = F.interpolate(m, size=t_lat, mode="linear", align_corners=False)
            tok = m.reshape(B * N, n, Cc, t_lat).permute(0, 3, 1, 2)
        return tok.reshape(B, N, t_lat, n, Cc)

    def _align_mask_time(self, masks, t_lat):
        """[B,N,T,h,w] -> [B,N,t_lat,h,w] (nearest temporal pooling)."""
        B, N, T, h, w = masks.shape
        if T == t_lat:
            return masks
        m = masks.reshape(B * N, T, h * w).transpose(1, 2)   # B*N, h*w, T
        m = F.interpolate(m.float(), size=t_lat, mode="nearest")
        return m.transpose(1, 2).reshape(B, N, t_lat, h, w).to(masks.dtype)

    def _inject_face(self, adapter_idx, main_tokens, motion_tokens, face_masks):
        """main_tokens: [B, t_lat*h*w, C]; motion_tokens: [B,N,t_lat,n,C];
        face_masks: [B,N,t_lat,h,w] or None. Returns residual to add to main_tokens."""
        fb = self.face_adapter.fuser_blocks[adapter_idx]
        N = motion_tokens.shape[1]
        # gradient-checkpoint EACH person's FaceBlock: don't store all K persons'
        # cross-attn activations for backward (that spiked VRAM on many-person clips
        # -> OOM). Forward keeps only the summed residual; backward recomputes one
        # person at a time. Keeps full multi-person + 97 frames + 1920x1056.
        use_ckpt = torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False)
        residual = None
        for p in range(N):
            mask_p = None if face_masks is None else face_masks[:, p]
            if use_ckpt:
                out = torch.utils.checkpoint.checkpoint(
                    fb, main_tokens, motion_tokens[:, p], mask_p, use_reentrant=False)
            else:
                out = fb(main_tokens, motion_tokens[:, p], mask_p)
            residual = out if residual is None else residual + out
        return residual

    # ------------------------------------------------------------------
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
            face_pixel_values = attention_kwargs.pop("face_pixel_values", None)
            face_masks = attention_kwargs.pop("face_masks", None)
        else:
            lora_scale = 1.0
            encoder_condition_states = None
            encoder_ref_states = None
            use_collaborative_position_encoding = False
            face_pixel_values = None
            face_masks = None

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # input video (condition) -> CHANNEL concat onto the noisy latent
        if encoder_condition_states is not None:
            main = torch.cat([hidden_states, encoder_condition_states], dim=1)
        elif self.patch_embedding.in_channels == 2 * self.config.in_channels:
            main = torch.cat([hidden_states, torch.zeros_like(hidden_states)], dim=1)
        else:
            main = hidden_states

        # reference -> in-context TOKEN concat (channel-padded to share patch_embedding)
        if encoder_ref_states is not None:  # one2three (reference control)
            extra_tokens = True
            num_refs = encoder_ref_states.shape[2]
            fake_nums = num_refs + post_patch_num_frames
            ref_in = self._pad_channels(encoder_ref_states, main.shape[1])
            if use_collaborative_position_encoding:
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
            rotary_emb = self.rope(hidden_states)
            hidden_states = self.patch_embedding(main).flatten(2).transpose(1, 2)

        num_tokens = hidden_states.shape[1]

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

        # ---- face expression tokens (computed once, reused at every injection) ----
        do_face = getattr(self, "_face_ready", False) and face_pixel_values is not None
        motion_tokens = None
        if do_face:
            import os, time
            _dbg = os.environ.get("PAI_DBG")
            if _dbg:
                torch.cuda.synchronize(); _t = time.time()
                print(f"[DBG] fwd: building motion from face {tuple(face_pixel_values.shape)} ...", flush=True)
            motion_tokens = self._build_motion_tokens(
                face_pixel_values, post_patch_num_frames, hidden_states.dtype
            )
            if _dbg:
                torch.cuda.synchronize()
                print(f"[DBG] fwd: motion_tokens {tuple(motion_tokens.shape)} done in {time.time()-_t:.1f}s", flush=True)
            if face_masks is not None:
                face_masks = self._align_mask_time(face_masks, post_patch_num_frames).to(hidden_states.dtype)
        else:
            import os
            _dbg = os.environ.get("PAI_DBG")
        main_count = post_patch_num_frames * post_patch_height * post_patch_width

        def maybe_inject(idx, hs):
            if not do_face or (idx % self.face_inject_every != 0):
                return hs
            adapter_idx = idx // self.face_inject_every
            if adapter_idx >= self.num_face_adapters:
                return hs
            ref_part = hs[:, :-main_count] if extra_tokens else None
            main_part = hs[:, -main_count:]
            residual = self._inject_face(adapter_idx, main_part, motion_tokens, face_masks)
            main_part = main_part + residual
            return torch.cat([ref_part, main_part], dim=1) if extra_tokens else main_part

        # Transformer blocks (+ face injection every `face_inject_every` blocks)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for idx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attn_mask
                )
                hidden_states = maybe_inject(idx, hidden_states)
        else:
            import time as _time
            for idx, block in enumerate(self.blocks):
                if _dbg:
                    torch.cuda.synchronize(); _tb = _time.time()
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attn_mask)
                if _dbg:
                    torch.cuda.synchronize()
                    print(f"[DBG] block {idx}: {_time.time()-_tb:.2f}s", flush=True)
                hidden_states = maybe_inject(idx, hidden_states)
                if _dbg and (idx % self.face_inject_every == 0) and do_face:
                    torch.cuda.synchronize()
                    print(f"[DBG]   + face inject at block {idx} done", flush=True)

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

        # drop the in-context reference tokens, keep the target tokens
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
