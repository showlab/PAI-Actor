"""
Face-adapter inference, built the SAME way as wan2_inference_channelcat.py
(subclass the train system + predict_step + L.Trainer(precision=bf16-mixed,
strategy=ddp).predict()) — NOT the broken standalone fp32 `_single` version.

Difference vs wan2_inference_channelcat.py:
  - subclasses the FACE train system (face transformer + motion_encoder/face_adapter)
  - dataset = CustomTrainDatasetFace (control + GT + ref + face crops + masks),
    matching training/validation
  - predict_step adds face via the SAME _add_face_kwargs used by training_step /
    validation_step  -> training / validation / inference are byte-for-byte consistent
  - condition = control_video (first_pixel_values), channel-concat; ref in-context
  - saves [control | generated | GT] per sample

Runs under Lightning Trainer => bf16-mixed autocast + multi-GPU DDP (fast). The
standalone single-gpu loop ran FP32 -> a 1920x1056x97 forward looked hung.
"""
import os
import argparse
import random

import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as L
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from diffusers.utils import export_to_video
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

from models.wan2.custom_pipeline import CustomWanPipeline
from src.wan2_system_channelcat_face import PAIActorChannelCatFaceSystem
from datasets.custom_dataset_face import CustomTrainDatasetFace, face_collate_fn


class PAIActorChannelCatFaceInferenceSystem(PAIActorChannelCatFaceSystem):
    # configure_model() (builds face transformer, widens patch_embedding 48->96,
    # adds motion_encoder[frozen]/face_encoder/face_adapter, LoRA) runs before
    # load_state_dict during load_from_checkpoint. load_state_dict (inherited) reads
    # ckpt["state_dict"]["trainable"] = LoRA + patch_embedding + face_encoder/adapter.

    def on_predict_epoch_start(self):
        self.pred_pipeline = CustomWanPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=UniPCMultistepScheduler.from_config(
                FlowMatchEulerDiscreteScheduler.from_pretrained(
                    self.hparams.model_id, subfolder="scheduler").config, flow_shift=5),
        )
        self.pred_path = self.hparams.pred_path
        os.makedirs(self.pred_path, exist_ok=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        first = batch["first_pixel_values"]    # B,C,F,H,W  control_video (condition)
        third = batch["third_pixel_values"]    # B,C,F,H,W  GT (target)

        ref_pixel_values = None
        if self.hparams.dataset.is_one2three:
            batch_ref = batch['ref_pixel_values']
            if isinstance(batch_ref, torch.Tensor):
                ref_pixel_values = batch_ref.permute(0, 2, 1, 3, 4)  # B,C,N,H,W

        # viz: control + GT
        ctrl_vis = ((first.squeeze(0).permute(1, 2, 3, 0) + 1) * 0.5).clamp(0, 1).cpu().numpy()
        gt_vis = ((third.squeeze(0).permute(1, 2, 3, 0) + 1) * 0.5).clamp(0, 1).cpu().numpy()

        meta_ref = None
        if self.hparams.dataset.is_one2three and ref_pixel_values is not None:
            ref_imgs = ((ref_pixel_values.squeeze(0) + 1) * 0.5).clamp(0, 1)  # C,N,H,W
            H, W = ref_imgs.shape[2], ref_imgs.shape[3]
            num_refs = ref_imgs.shape[1]
            r = F.interpolate(ref_imgs.permute(1, 0, 2, 3), size=(H, W // num_refs),
                              mode='bilinear', align_corners=False).permute(0, 2, 3, 1).cpu().numpy()
            ref_concat = np.concatenate([r[i] for i in range(num_refs)], axis=1)
            meta_ref = np.stack([ref_concat] * self.hparams.dataset.sample_n_frames, axis=0)

        # encode condition + reference latents
        cond_lat = self.vae.encode(first).latent_dist.sample()
        cond_lat = (cond_lat - self.latents_mean) / self.latents_std
        ref_lat = None
        if ref_pixel_values is not None:
            N = ref_pixel_values.shape[2]
            encs = [self.vae.encode(ref_pixel_values[:, :, i:i + 1]).latent_dist.sample() for i in range(N)]
            ref_lat = (torch.cat(encs, dim=2) - self.latents_mean) / self.latents_std

        attention_kwargs = {
            'encoder_condition_states': cond_lat,                 # channel concat
            'encoder_ref_states': ref_lat,                        # in-context token concat
            'use_collaborative_position_encoding': self.hparams.use_collaborative_position_encoding,
        }
        # FACE: same helper as training_step / validation_step -> consistent
        attention_kwargs = self._add_face_kwargs(attention_kwargs, batch)

        video_generate = self.pred_pipeline(
            prompt=batch["prompts"],
            height=self.hparams.dataset.height, width=self.hparams.dataset.width,
            num_frames=self.hparams.dataset.sample_n_frames,
            guidance_scale=5.0, attention_kwargs=attention_kwargs,
        ).frames[0]

        parts = ([meta_ref] if meta_ref is not None else []) + [ctrl_vis, video_generate, gt_vis]
        # pad widths to match for concat along H
        tw = max(p.shape[2] for p in parts)
        parts = [np.pad(p, ((0, 0), (0, 0), (0, tw - p.shape[2]), (0, 0)), mode='constant') if p.shape[2] != tw else p for p in parts]
        cat = np.concatenate(parts, axis=1)
        out = os.path.join(self.pred_path, f"pred_{batch_idx}_rank{self.trainer.global_rank}.mp4")
        export_to_video(cat, output_video_path=out, fps=self.hparams.dataset.fps)
        print(f"[ok] {out}", flush=True)
        return


def main(opt):
    L.seed_everything(opt.seed)
    ds = CustomTrainDatasetFace(
        data_jsonl=opt.dataset.data_jsonl, height=opt.dataset.height, width=opt.dataset.width,
        resize_long=opt.dataset.resize_long, sample_n_frames=opt.dataset.sample_n_frames,
        is_one2three=opt.dataset.is_one2three, max_persons=opt.face.get("max_persons", 0))
    # a few random samples
    rng = random.Random(opt.seed)
    idxs = rng.sample(range(len(ds)), min(opt.num_samples, len(ds)))
    ds = Subset(ds, idxs)
    dl = DataLoader(ds, batch_size=1, num_workers=opt.dataset.num_workers,
                    drop_last=False, pin_memory=opt.dataset.pin_memory, shuffle=False,
                    collate_fn=face_collate_fn)

    system = PAIActorChannelCatFaceInferenceSystem.load_from_checkpoint(opt.ckpt_path, opt=opt)
    trainer = L.Trainer(
        logger=False, precision=opt.training.precision, log_every_n_steps=1,
        accelerator=opt.training.accelerator, strategy=opt.training.strategy,
        benchmark=opt.training.benchmark, num_nodes=opt.num_nodes)
    trainer.predict(system, dataloaders=dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wan2-2_channelcat_face_background_replace.yaml")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args, extras = parser.parse_known_args()
    args = vars(args)
    opt = OmegaConf.merge(
        OmegaConf.load(args['config']),
        OmegaConf.from_cli(extras),
        OmegaConf.create(args),
        OmegaConf.create({"num_nodes": int(os.environ.get("NUM_NODES", 1))}),
        OmegaConf.create({"num_gpus": int(torch.cuda.device_count())}),
    )
    main(opt)
