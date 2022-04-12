#!/usr/bin/env bash
set -e 
set -x 
DATA_DIR=data/ace/pro_mttrig_id/json
MODEL=constrained-gen
CKPT_NAME=constrained-gen-ACE 


rm -rf checkpoints/${CKPT_NAME}
python train.py --model=${MODEL} --ckpt_name=${CKPT_NAME} \
    --dataset=ACE \
    --tmp_dir=preprocessed_ACE \
    --train_file=${DATA_DIR}/train.oneie.json \
    --val_file=${DATA_DIR}/dev.oneie.json \
    --test_file=${DATA_DIR}/test.oneie.json \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=6 \
    --mark_trigger 
