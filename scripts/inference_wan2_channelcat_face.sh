#!/usr/bin/env bash
# Face-adapter inference for PAI-Actor (channel-concat condition + in-context ref + face adapter).
# bf16 is required (fp32 may hang at 1920x1056x97).
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Download weights from https://huggingface.co/ghyggg/PAI-Actor-release
CKPT="checkpoints/wan2-2_channelcat_face_bg_replace_step8000.ckpt"
OUT="outputs/infer"

CUDA_VISIBLE_DEVICES=0 python src/wan2_inference_channelcat_face.py \
    --config=configs/wan2-2_channelcat_face_background_replace.yaml \
    --ckpt_path "${CKPT}" \
    --pred_path "${OUT}" \
    --num_samples 4 \
    --seed 1234
