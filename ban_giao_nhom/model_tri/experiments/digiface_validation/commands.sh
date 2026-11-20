#!/usr/bin/env bash

# GPU selection used for these experiments
export CUDA_VISIBLE_DEVICES=1

# Preflight (run before training)
python -u training/train.py --mode B1 --preflight-only

# B0 baseline
python -u training/train.py --mode B0 \
  --device cuda --eval-batch-size 32 --num-workers 2 \
  --skip-hash-check

# B1 full fine-tuning
python -u training/train.py --mode B1 \
  --device cuda --epochs 5 --batch-size 8 \
  --eval-batch-size 32 --num-workers 2 \
  --seed 20261028 --run-name B1_seed20261028_bs8 \
  --skip-hash-check

# B2 full fine-tuning
python -u training/train.py --mode B2 \
  --device cuda --epochs 5 --batch-size 8 \
  --eval-batch-size 32 --num-workers 2 \
  --seed 20261028 --occlusion-prob 0.5 \
  --run-name B2_seed20261028_bs8 \
  --skip-hash-check
