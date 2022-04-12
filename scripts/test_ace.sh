#!/usr/bin/env bash
set -e 
set -x 
CKPT_PATH=constrained-gen-ACE
MODEL=constrained-gen
DATA_DIR=data/ace/pro_mttrig_id/json
CKPT_NAME=epoch=2-v0.ckpt

python train.py --model=$MODEL --ckpt_name=$CKPT_PATH-pred \
    --load_ckpt=checkpoints/$CKPT_PATH/$CKPT_NAME \
    --dataset=ACE \
    --mark_trigger \
    --eval_only \
    --train_file=${DATA_DIR}/train.oneie.json \
    --val_file=${DATA_DIR}/dev.oneie.json \
    --test_file=${DATA_DIR}/test.oneie.json \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 


python src/genie/scorer.py --gen-file=checkpoints/$CKPT_PATH-pred/predictions.jsonl --dataset=ACE \
--test-file=data/ace/pro_mttrig_id/json/test.oneie.json 

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_PATH-pred/predictions.jsonl --dataset=ACE \
--test-file=data/ace/pro_mttrig_id/json/test.oneie.json --head-only 