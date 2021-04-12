#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=gen-RAMS
MODEL=gen

python train.py --model=$MODEL --ckpt_name=$CKPT_NAME-pred \
    --load_ckpt=checkpoints/$CKPT_NAME/epoch=2-v0.ckpt \
    --dataset=RAMS \
    --eval_only \
    --train_file=data/RAMS_1.0/data/train.jsonlines \
    --val_file=data/RAMS_1.0/data/dev.jsonlines \
    --test_file=data/RAMS_1.0/data/test.jsonlines \
    --train_batch_size=2 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=3


#span eval 
python genie/convert_gen_to_output.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--output-file=checkpoints/$CKPT_NAME-pred/span_output.jsonl 

python data/RAMS_1.0/scorer/scorer.py -g=data/RAMS_1.0/data/test.jsonlines -p=checkpoints/$CKPT_NAME-pred/span_output.jsonl \
--reuse_gold_format --do_all > checkpoints/$CKPT_NAME-pred/span_metrics.txt 

# head eval
python genie/convert_gen_to_output.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--output-file=checkpoints/$CKPT_NAME-pred/output.jsonl --head-only

python data/RAMS_1.0/scorer/scorer.py -g=data/RAMS_1.0/data/test_head.jsonlines -p=checkpoints/$CKPT_NAME-pred/output.jsonl \
--reuse_gold_format --do_all > checkpoints/$CKPT_NAME-pred/head_metrics.txt 

# head + coref eval 
python genie/convert_gen_to_output.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/RAMS_1.0/data/test_head_coref.jsonlines \
--output-file=checkpoints/$CKPT_NAME-pred/coref_output.jsonl --head-only --coref

python data/RAMS_1.0/scorer/scorer.py -g=data/RAMS_1.0/data/test_head_coref.jsonlines -p=checkpoints/$CKPT_NAME-pred/coref_output.jsonl \
--reuse_gold_format --do_all > checkpoints/$CKPT_NAME-pred/coref_metrics.txt 


# visualize 
python visualize_output.py --result-file=checkpoints/$CKPT_NAME-pred/span_output.jsonl 