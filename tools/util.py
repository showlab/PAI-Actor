from pytorch_lightning.callbacks import TQDMProgressBar
import torch
from typing import Any, Dict, List, Tuple, Optional
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Callback
import pytorch_lightning as L

import os
import torchvision


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, sanity_checks=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sanity_checks = sanity_checks

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        training_step = batch_idx + trainer.current_epoch * trainer.num_training_batches
        #
        if self.sanity_checks and training_step == 0: # sanity_checks
            samples_path = os.path.join(pl_module.hparams.output_root, pl_module.hparams.experiment_name, 'sanity_checks')
            os.makedirs(samples_path, exist_ok=True)
            pl_module.print(f"Sanity_checks, Save to {samples_path}")
            #
            samples = batch['first_pixel_values'].cpu()
            [
                torchvision.io.write_video(
                    os.path.join(samples_path, f"first_video_{idx}.mp4"),
                    ((sample.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0),
                    fps=8,
                )
                for idx, sample in enumerate(samples)
            ]
            #
            samples = batch['third_pixel_values'].cpu()
            [
                torchvision.io.write_video(
                    os.path.join(samples_path, f"third_video_{idx}.mp4"),
                    ((sample.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0),
                    fps=8,
                )
                for idx, sample in enumerate(samples)
            ]
        #
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # log save information
        pl_module.print(f"⏰ Save ckpt at: epoch={trainer.current_epoch}, step={trainer.global_step}.")


class CustomProgressBar(TQDMProgressBar):
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.train_progress_bar.reset(self.total_train_batches)
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Training step")

    def on_train_epoch_start(self, trainer=None, pl_module=None):
        if self._leave:
            self.train_progress_bar = self.init_train_tqdm()
        self.train_progress_bar.reset(self.total_train_batches)
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Training step")

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        #
        formatted = {}
        for k, v in metrics.items():
            if isinstance(v, float):
                if "lr" in k:
                    formatted[k] = f"{v:.7f}"
                elif "step" in k:
                    formatted[k] = int(v)
                else:
                    formatted[k] = f"{v:.3f}"
            else:
                formatted[k] = v
        return formatted


def resolve_strategy(strategy_name):
    if strategy_name != 'fsdp':
        return strategy_name
    else:
        print("*" * 25 + "use fsdp!!!" + "*" * 25)
        # from pytorch_lightning.strategies import FSDPStrategy
        from torch.distributed.fsdp import ShardingStrategy
        from transformers.models.umt5.modeling_umt5 import UMT5Block
        from models.wan2.wan_block import WanTransformerBlock
        from pytorch_lightning.strategies import FSDPStrategy
        def custom_auto_wrap_policy(module, recurse, nonwrapped_numel):
            if isinstance(module, (WanTransformerBlock, UMT5Block)):
                return True
            else:
                return False
        '''
            if state_dict_type='sharded', merge by the example below:
            cd lightning_logs/version_0/checkpoints
            python -m lightning.pytorch.utilities.consolidate_checkpoint epoch=0-step=3.ckpt
        '''
        strategy = FSDPStrategy(
            auto_wrap_policy=custom_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            state_dict_type="full",
            use_orig_params=True,
            limit_all_gathers=True,
            sync_module_states=True,
        )
        return strategy


# copy from wan2.2
def masks_like(tensor, zero=False, generator=None, p=0.2):
    if not isinstance(tensor, list):
        tensor = [tensor]
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]
    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]
    #
    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[:, 0] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[:, 0]).exp()
                    v[:, 0] = torch.zeros_like(v[:, 0])
                else:
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]
        else:
            for u, v in zip(out1, out2):
                u[:, :, 0] = torch.zeros_like(u[:, :, 0])
                v[:, :, 0] = torch.zeros_like(v[:, :, 0])

    return out1, out2
