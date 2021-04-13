import os 
import argparse 
import torch 
import logging 
import json 


import pytorch_lightning as pl 
from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from .network import BartGen
from .constrained_gen import BartConstrainedGen

logger = logging.getLogger(__name__)

class GenIEModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams = args 
    

        self.config=BartConfig.from_pretrained('facebook/bart-large')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>'])

        
        if self.hparams.model=='gen':
            self.model = BartGen(self.config, self.tokenizer)
            self.model.resize_token_embeddings() 
        elif self.hparams.model == 'constrained-gen':
            self.model = BartConstrainedGen(self.config, self.tokenizer)
            self.model.resize_token_embeddings() 
        else:
            raise NotImplementedError



    def forward(self, inputs):
    
        return self.model(**inputs)


    def training_step(self, batch, batch_idx):
        '''
        processed_ex = {
                            'doc_key': ex['doc_key'],
                            'input_tokens_ids':input_tokens['input_ids'],
                            'input_attn_mask': input_tokens['attention_mask'],
                            'tgt_token_ids': tgt_tokens['input_ids'],
                            'tgt_attn_mask': tgt_tokens['attention_mask'],
                        }
        '''
        inputs = {
                    "input_ids": batch["input_token_ids"],
                    "attention_mask": batch["input_attn_mask"],
                    "decoder_input_ids": batch['tgt_token_ids'],
                    "decoder_attention_mask": batch["tgt_attn_mask"],   
                    "task": 0 
                }

        outputs = self.model(**inputs)
        loss = outputs[0]
        loss = torch.mean(loss)

        log = {
            'train/loss': loss, 
        } 
        return {
            'loss': loss, 
            'log': log 
        }
    

    def validation_step(self,batch, batch_idx):
        inputs = {
                    "input_ids": batch["input_token_ids"],
                    "attention_mask": batch["input_attn_mask"],
                    "decoder_input_ids": batch['tgt_token_ids'],
                    "decoder_attention_mask": batch["tgt_attn_mask"],  
                    "task" :0,   
                }
        outputs = self.model(**inputs)
        loss = outputs[0]
        loss = torch.mean(loss)

       
        
        return loss  

    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        log = {
            'val/loss': avg_loss, 
        } 
        return {
            'loss': avg_loss, 
            'log': log 
        }
        
        
        

    def test_step(self, batch, batch_idx):
        if self.hparams.sample_gen:
            sample_output = self.model.generate(batch['input_token_ids'], do_sample=True, 
                                top_k=20, top_p=0.95, max_length=30, num_return_sequences=1,num_beams=1,
                            )
        else:
            sample_output = self.model.generate(batch['input_token_ids'], do_sample=False, 
                                max_length=30, num_return_sequences=1,num_beams=1,
                            )
        
        sample_output = sample_output.reshape(batch['input_token_ids'].size(0), 1, -1)
        doc_key = batch['doc_key'] # list 
        tgt_token_ids = batch['tgt_token_ids']

        return (doc_key, sample_output, tgt_token_ids) 

    def test_epoch_end(self, outputs):
        # evaluate F1 
        with open('checkpoints/{}/predictions.jsonl'.format(self.hparams.ckpt_name),'w') as writer:
            for tup in outputs:
                for idx in range(len(tup[0])):
                    
                    pred = {
                        'doc_key': tup[0][idx],
                        'predicted': self.tokenizer.decode(tup[1][idx].squeeze(0), skip_special_tokens=True),
                        'gold': self.tokenizer.decode(tup[2][idx].squeeze(0), skip_special_tokens=True) 
                    }
                    writer.write(json.dumps(pred)+'\n')

        return {} 


    def configure_optimizers(self):
        self.train_len = len(self.train_dataloader())
        if self.hparams.max_steps > 0:
            t_total = self.hparams.max_steps
            self.hparams.num_train_epochs = self.hparams.max_steps // self.train_len // self.hparams.accumulate_grad_batches + 1
        else:
            t_total = self.train_len // self.hparams.accumulate_grad_batches * self.hparams.num_train_epochs

        logger.info('{} training steps in total.. '.format(t_total)) 
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # scheduler is called only once per epoch by default 
        scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'linear-schedule',
        }

        return [optimizer, ], [scheduler_dict,]