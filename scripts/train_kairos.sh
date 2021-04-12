#!/usr/bin/env bash
set -e 
set -x 

CKPT_NAME='gen-KAIROS'
rm -rf checkpoints/${CKPT_NAME}
python train.py --model=gen --ckpt_name=${CKPT_NAME} \
    --dataset=KAIROS \
    --train_file=data/kairos/train.jsonl \
    --val_file=data/kairos/dev.jsonl \
    --test_file=data/kairos/test.jsonl \
    --train_batch_size=2 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=8 \
    --num_train_epochs=3 \
    --mark_trigger \
    --use_info \
    --coref_dir=data/kairos/coref_outputs 
