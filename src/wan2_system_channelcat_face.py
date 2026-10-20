"""
PAI-Actor channel-concat + face-adapter system (inference).

Extends the channel-concat system with the Wan-Animate expression branch
(face crops -> motion_encoder [frozen, pretrained] -> face_encoder / face_adapter
-> per-person masked cross-attention). This file builds the face-aware transformer
(`configure_model`) and exposes the face-conditioning helper (`_add_face_kwargs`)
used by the inference `predict_step`.

`configure_model` is kept byte-for-byte identical to the training-time model build
so the saved checkpoint (LoRA + widened patch_embedding + face_encoder/face_adapter)
loads with matching state_dict keys. Training code has been removed.
"""
import os
import torch

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan

from tools.my_schedule import FlowMatchScheduler
from src.wan2_system_channelcat import PAIActorChannelCatSystem
from models.wan2.transformer_wan_channelcat_face import ChannelCatFaceWanTransformer3DModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PAIActorChannelCatFaceSystem(PAIActorChannelCatSystem):

    def configure_model(self):
        if self.is_configured:
            return
        self.is_configured = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_id, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.hparams.model_id, subfolder="text_encoder", torch_dtype=torch.float32)
        self.vae = AutoencoderKLWan.from_pretrained(
            self.hparams.model_id, subfolder="vae", torch_dtype=torch.float32)
        self.train_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler.set_timesteps(1000, training=True)

        # face-aware channel-concat transformer
        self.transformer = ChannelCatFaceWanTransformer3DModel.from_pretrained(
            self.hparams.model_id, subfolder="transformer", torch_dtype=torch.float32)
        self.transformer.expand_patch_embedding_inplace()            # 48 -> 96
        self.transformer.add_face_modules(
            face_inject_every=self.hparams.face.get("inject_every", 5))

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        if self.hparams.use_lora:
            self.transformer.requires_grad_(False)
        if self.hparams.training.gradient_checkpointing:
            self.transformer.gradient_checkpointing = True
            self.transformer.enable_gradient_checkpointing()
        self.register_buffer('latents_mean', torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1))
        self.register_buffer('latents_std', torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1))

        if self.hparams.use_lora:
            from peft import LoraConfig
            transformer_lora_config = LoraConfig(
                r=self.hparams.training.rank,
                lora_alpha=self.hparams.training.rank,
                init_lora_weights=True,
                target_modules=["to_k", "to_q", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"],
            )
            self.transformer.add_adapter(transformer_lora_config)   # freezes all, enables LoRA
            self.transformer.patch_embedding.requires_grad_(True)

        # face modules: motion_encoder is frozen/pretrained; face_encoder/face_adapter
        # are loaded from the checkpoint "trainable" blob.
        me_path = self.hparams.face.motion_encoder_path
        miss, unexp = self.transformer.load_motion_encoder(me_path)
        try:
            _gz = self.trainer.is_global_zero    # raises if no trainer attached (load_from_checkpoint)
        except Exception:
            _gz = True
        if _gz:
            print(f"[face] motion_encoder loaded from {me_path} (missing={len(miss)}, unexpected={len(unexp)}) -> FROZEN")
        self.transformer.motion_encoder.requires_grad_(False)
        self.transformer.face_encoder.requires_grad_(True)
        self.transformer.face_adapter.requires_grad_(True)

    def _add_face_kwargs(self, attention_kwargs, batch):
        face = batch.get("face_pixel_values", None)
        if face is not None:
            attention_kwargs["face_pixel_values"] = face.to(self.device)
            fmask = batch.get("face_masks", None)
            attention_kwargs["face_masks"] = fmask.to(self.device) if fmask is not None else None
        return attention_kwargs
