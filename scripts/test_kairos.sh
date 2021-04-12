#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=gen-KAIROS
MODEL=gen

rm -rf checkpoints/${CKPT_NAME}-pred 
python train.py --model=$MODEL --ckpt_name=${CKPT_NAME}-pred \
    --load_ckpt=checkpoints/${CKPT_NAME}/epoch=2.ckpt \
    --dataset=KAIROS \
    --eval_only \
    --train_file=data/kairos/train.jsonl \
    --val_file=data/kairos/dev.jsonl \
    --test_file=data/kairos/test.jsonl \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=3

python genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/kairos/test_info.jsonl \
--dataset=KAIROS \
--coref-file=data/kairos/coref_outputs/test.jsonlines \
--coref 

python genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/kairos/test_info.jsonl \
--dataset=KAIROS \
--coref-file=data/kairos/coref_outputs/test.jsonlines \
--head-only \
--coref 