import argparse 
import logging 
import os 
import random 
import timeit 
from datetime import datetime 

import torch 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from zstagger.data_module import TaggerDataModule
from zstagger.KAIROS_data_module import DocTaggerDataModule
from zstagger.model import TaggerModel
from zstagger.zs_model import ZSTaggerModel


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=['crf','zs-crf','proto']
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['ACE', 'ERE','KAIROS']
    ),
    parser.add_argument(
        '--task',
        type=str,
        default='trigger',
        choices=['trigger', 'arg'],
        help='whether to perform trigger extraction or argument extraction.'
    )
    parser.add_argument(
        "--pretrained_model",
        type=str, 
        default='roberta-base',
    )
    parser.add_argument(
        '--tmp_dir', 
        type=str, 
        default='tag_data', 
        help='temporary directory for saving the preprocessed data. If this exists, will directly read from dir.')

    parser.add_argument(
        '--event_n', 
        type=int, 
        default=33,
    )
    parser.add_argument('--no_projection', action='store_true')
    parser.add_argument('--token_classification', action='store_true')
    parser.add_argument('--use_pl', action='store_true')
    parser.add_argument('--proj_dim', type=int, default=500)
    parser.add_argument('--use_transition', action='store_true')
    parser.add_argument('--reg_weight',type=float, default=0.5)
    parser.add_argument('--use_bilstm', action='store_true')
    parser.add_argument('--bilstm_hidden_size', type=int, default=100)
    parser.add_argument('--bilstm_dropout', type=float, default=0.5)
    parser.add_argument(
        "--ckpt_name",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--load_ckpt",
        default=None,
        type=str, 
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--val_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default=None,
    )
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--eval_only", action="store_true",
    )
    parser.add_argument("--bert_learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--bert_weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--bert_dropout", default=0.5, type=float, help="Dropout after BERT encoding.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    
    parser.add_argument("--gpus", default=-1, help='-1 means train on all gpus')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Set seed
    seed_everything(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    
    if not args.ckpt_name:
        d = datetime.now() 
        time_str = d.strftime('%m-%dT%H%M')
        args.ckpt_name = '{}_{}lr{}_{}'.format(args.model,  args.train_batch_size * args.accumulate_grad_batches, 
                args.learning_rate, time_str)


    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')
    
    os.makedirs(args.ckpt_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        save_top_k=1,
        monitor='val/f1',
        mode='max',
        save_weights_only=True,
        filename='best',

    )

    early_stop_callback = EarlyStopping(
            monitor='val/f1',
            min_delta=0.0,
            patience=5,
            verbose=False,
            mode='max'
            )
            


    lr_logger = LearningRateMonitor() 
    tb_logger = TensorBoardLogger('logs/')
    # wb_logger = WandbLogger(project='genie', name=args.ckpt_name)

    # model = TaggerModel(args)
    if args.model in ['zs-crf', 'proto']:
        model = ZSTaggerModel(args)
    else:
        model = TaggerModel(args)
    if args.dataset == 'ACE':
        dm = TaggerDataModule(args)
    elif args.dataset=='KAIROS':
        dm = DocTaggerDataModule(args)
  



    if args.max_steps < 0 :
        args.max_epochs = args.min_epochs = args.num_train_epochs 
    
    

    trainer = Trainer(
        logger=tb_logger, 
        min_epochs=10,
        max_epochs=args.num_train_epochs, 
        gpus=args.gpus, 
        checkpoint_callback=checkpoint_callback, 
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, 
        num_sanity_val_steps=0, 
        val_check_interval=1.0, # use float to check every n epochs 
        precision=16 if args.fp16 else 32,
        callbacks = [lr_logger, early_stop_callback],
        reload_dataloaders_every_epoch=True

    ) 

    if args.load_ckpt:
        model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 

    
    if args.eval_only: 
        dm.setup('test')
        trainer.test(model, datamodule=dm) #also loads training dataloader 
    else:
        dm.setup('fit')
        trainer.fit(model, datamodule=dm) 
    

    

if __name__ == "__main__":
    main()