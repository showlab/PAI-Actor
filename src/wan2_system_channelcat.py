"""
PAI-Actor channel-concat system (inference).

Builds the channel-concat Wan2.2 transformer and restores the trained parameters
(LoRA + the widened patch_embedding [+ face modules in the face subclass]) from a
checkpoint. Only the model construction and checkpoint-loading logic needed for
inference is kept here; the training loop (training_step / optimizers / dataloaders
/ loss / checkpoint saving) has been removed.

The model-building (`configure_model`) lives in the face subclass; this base class
provides `__init__` and the custom `load_state_dict` that reads the "trainable"
blob produced during training.
"""
import torch
import pytorch_lightning as L


class PAIActorChannelCatSystem(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        self.is_configured = False

    def load_state_dict(self, state_dict, strict: bool = True):
        # Channel-concat checkpoints store ALL trained params under "trainable"
        # (LoRA adapters + the widened patch_embedding, and the face_encoder/
        # face_adapter in the face model). Fall back to the legacy "lora"-only
        # format for compatibility. The frozen base weights come from
        # `from_pretrained(model_id)` in configure_model.
        if 'trainable' in state_dict:
            self.transformer.load_state_dict(state_dict['trainable'], strict=False)
        elif 'lora' in state_dict:
            self.transformer.load_state_dict(state_dict['lora'], strict=False)
