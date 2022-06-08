import os 
import torch 
import logging 
import json 
from collections import defaultdict 

import pytorch_lightning as pl 
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch_struct import LinearChainCRF

from .layers import CRFFeatureHead, BERTEncoder
from .utils import load_ontology, evaluate_trigger_f1, load_role_mapping, evaluate_arg_f1


logger = logging.getLogger(__name__)

class TaggerModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams = args 
    

        self.config=AutoConfig.from_pretrained(args.pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        self.n_classes = self.hparams.event_n +1 # IO tagging 
        # network internals 
        self.encoder = BERTEncoder(args, self.config.hidden_size)
        self.feature_size = self.config.hidden_size  # extra bert layer 

        self.head = CRFFeatureHead(self.hparams, self.feature_size, self.n_classes)


    def forward(self, inputs, stage='training'):

        lengths = inputs['word_lengths']
        features = self.encoder(input_ids=inputs['input_token_ids'], 
            attention_mask=inputs['input_attn_mask'],
            token_lens=inputs['token_lens']) # (batch,word_seq_len, hidden_dim)
        
        dist = self.head(features, lengths)
        

        if stage=='training':
            assert(torch.max(lengths) == inputs['word_tags'].size(1))
            assert(torch.max(inputs['word_tags']) < self.n_classes) # should be [0, C-1]
            label_ = LinearChainCRF.struct.to_parts(inputs['word_tags'], self.n_classes,
                                                    lengths=lengths).type_as(dist.log_potentials)
            
            
            loss = -dist.log_prob(label_).mean() 
            return loss 
        
        else:
            # Compute predictions
            argmax = dist.argmax
            preds = dist.from_event(argmax)[0] # (batch, seq)
            return preds 


    def training_step(self, batch, batch_idx):
        '''
        'doc_key': ex['sent_id'],
        'input_token_ids':input_tokens['input_ids'],
        'input_attn_mask': input_tokens['attention_mask'],
        'labels': labels,
        'bpe_mapping': bpe2word, 
        'token_lens': token_lens, 
        'word_tags': word_tags,
   
        '''
        loss = self.forward(batch, stage='training')

        log = {
            'train/loss': loss, 
        } 
        return {
            'loss': loss, 
            'log': log 
        }
    

    def validation_step(self,batch, batch_idx):
        preds = self.forward(batch, stage='testing')

        return {
            "doc_key": batch['doc_key'],
            "chunk_idx" : batch['chunk_idx'], 
             # "input_ids": batch["input_token_ids"].cpu().tolist(),
             # "attention_mask": batch["input_attn_mask"].cpu().tolist(),
             "preds": preds.cpu().tolist(),
             "word_tags": batch['word_tags'].cpu().tolist(),
             "word_lengths": batch["word_lengths"].cpu().tolist(),
             # "labels": batch["labels"].cpu().tolist(),
             # "bpe_mapping": batch['bpe_mapping'].cpu().tolist()
        }

    
    def validation_epoch_end(self, outputs):
        predictions = defaultdict(list)

        # aggregate predictions with the same doc_key
        for batch_dict in outputs:
            for i in range(len(batch_dict['doc_key'])):
                # length = sum(batch_dict['attention_mask'][i])
                length=batch_dict['word_lengths'][i]
                doc_key = batch_dict['doc_key'][i]
                chunk_idx = batch_dict['chunk_idx'][i]
                # input_ids = batch_dict['input_ids'][i][:length]
                pred = batch_dict['preds'][i][:length]
                labels = batch_dict['word_tags'][i][:length]
                # bpe_mapping = batch_dict['bpe_mapping'][i][:length]
                predictions[doc_key].append({
                    'pred_tags': pred,
                    'labels': labels,
                    'chunk_idx': chunk_idx
                })
        
        combined_predictions =  {} # doc_key -> predictions 
        for doc_key, pred_list in predictions.items():
            # sort by chunk_idx and merge 
            sorted_pred_list = sorted(pred_list, key=lambda x:x['chunk_idx'])
            combined_predictions[doc_key] = {
                'pred_tags': [tag for pred in sorted_pred_list for tag in pred['pred_tags']],
                'labels': [label for pred in sorted_pred_list for label in pred['labels']]
            }
        if self.hparams.task == 'trigger':
            with open(self.hparams.val_file,'r') as f:
                for line in f:
                    ex = json.loads(line)
                    if self.hparams.dataset == 'ACE':
                        combined_predictions[ex['sent_id']]['event_mentions'] = ex['event_mentions']
                    else:
                        combined_predictions[ex['doc_id']]['event_mentions'] = ex['event_mentions']
            ontology_dict = load_ontology(self.hparams.dataset)
            tgr_id_f1, tgr_f1 = evaluate_trigger_f1(combined_predictions, ontology_dict)
            log = {
                'val/id_f1': torch.Tensor([tgr_id_f1,]),
                'val/f1': torch. Tensor([tgr_f1,])
            } 
            return {
                'f1': tgr_f1, 
                'log': log 
            }
        else:
            with open(self.hparams.val_file, 'r') as f:
                for line in f:
                    ex = json.loads(line)
                    if self.hparams.dataset == 'ACE':
                        for doc_key in combined_predictions:
                            sent_id, evt_idx = doc_key.split(':')
                            if sent_id == ex['sent_id']:
                                combined_predictions[doc_key]['event_mentions'] = [ex['event_mentions'][int(evt_idx)], ]
                                combined_predictions[doc_key]['entity_mentions'] = ex['entity_mentions']
                    else:
                        for doc_key in combined_predictions:
                            doc_id, evt_idx = doc_key.split(':')
                            if doc_id == ex['doc_id']:
                                combined_predictions[doc_key]['event_mentions'] = [ex['event_mentions'][int(evt_idx)], ]
                                combined_predictions[doc_key]['entity_mentions'] = ex['entity_mentions']
                                combined_predictions[doc_key]['sent_lens'] = [len(sent[0]) for sent in ex['sentences']]

            with open('checkpoints/{}/predictions.json'.format(self.hparams.ckpt_name),'w') as f:
                json.dump(combined_predictions, f)
            
            role_mapping = load_role_mapping(self.hparams.dataset)
            arg_id_f1 , arg_f1 = evaluate_arg_f1(combined_predictions, role_mapping)
            log = {
                'val/id_f1':arg_id_f1,
                'val/f1': arg_f1
            } 
            return {
                'f1': arg_f1, 
                'log': log 
            }
        
        
        

    def test_step(self, batch, batch_idx):
        
        preds = self.forward(batch, stage='testing')

        return {
            "doc_key": batch['doc_key'],
            "chunk_idx" : batch['chunk_idx'], 
             # "input_ids": batch["input_token_ids"].cpu().tolist(),
             # "attention_mask": batch["input_attn_mask"].cpu().tolist(),
             "preds": preds.cpu().tolist(),
             "word_tags": batch['word_tags'].cpu().tolist(),
             "word_lengths": batch["word_lengths"].cpu().tolist(),
             # "labels": batch["labels"].cpu().tolist(),
             # "bpe_mapping": batch['bpe_mapping'].cpu().tolist()
        }

    def test_epoch_end(self, outputs):
        ontology_dict = load_ontology(self.hparams.dataset)
        predictions = defaultdict(list)

        # aggregate predictions with the same doc_key
        for batch_dict in outputs:
            for i in range(len(batch_dict['doc_key'])):
                # length = sum(batch_dict['attention_mask'][i])
                length=batch_dict['word_lengths'][i]
                doc_key = batch_dict['doc_key'][i]
                chunk_idx = batch_dict['chunk_idx'][i]
                # input_ids = batch_dict['input_ids'][i][:length]
                pred = batch_dict['preds'][i][:length]
                labels = batch_dict['word_tags'][i][:length]
                # bpe_mapping = batch_dict['bpe_mapping'][i][:length]
                predictions[doc_key].append({
                    'pred_tags': pred,
                    'labels': labels,
                    'chunk_idx': chunk_idx
                })
        
        combined_predictions =  {} # doc_key -> predictions 
        for doc_key, pred_list in predictions.items():
            # sort by chunk_idx and merge 
            sorted_pred_list = sorted(pred_list, key=lambda x:x['chunk_idx'])
            combined_predictions[doc_key] = {
                'pred_tags': [tag for pred in sorted_pred_list for tag in pred['pred_tags']],
                'labels': [label for pred in sorted_pred_list for label in pred['labels']]
            }

        
        if self.hparams.task == 'trigger':
            with open(self.hparams.test_file,'r') as f:
                for line in f:
                    ex = json.loads(line)
                    if self.hparams.dataset == 'ACE':
                        combined_predictions[ex['sent_id']]['event_mentions'] = ex['event_mentions']
                    else:
                        combined_predictions[ex['doc_id']]['event_mentions'] = ex['event_mentions']
            with open('checkpoints/{}/predictions.json'.format(self.hparams.ckpt_name),'w') as f:
                json.dump(combined_predictions, f)

            ontology_dict = load_ontology(self.hparams.dataset)
            tgr_id_f1, tgr_f1 = evaluate_trigger_f1(combined_predictions, ontology_dict)
            log = {
                'val/id_f1': torch.Tensor([tgr_id_f1,]),
                'val/f1': torch. Tensor([tgr_f1,])
            } 
            return {
                'f1': tgr_f1, 
                'log': log 
            }
        else:
            with open(self.hparams.test_file, 'r') as f:
                for line in f:
                    ex = json.loads(line)
                    if self.hparams.dataset == 'ACE':
                        for doc_key in combined_predictions:
                            sent_id, evt_idx = doc_key.split(':')
                            if sent_id == ex['sent_id']:
                                combined_predictions[doc_key]['event_mentions'] = [ex['event_mentions'][int(evt_idx)], ]
                                combined_predictions[doc_key]['entity_mentions'] = ex['entity_mentions']
                    else:
                        for doc_key in combined_predictions:
                            doc_id, evt_idx = doc_key.split(':')
                            if doc_id == ex['doc_id']:
                                combined_predictions[doc_key]['event_mentions'] = [ex['event_mentions'][int(evt_idx)], ]
                                combined_predictions[doc_key]['entity_mentions'] = ex['entity_mentions']
                                combined_predictions[doc_key]['sent_lens'] = [len(sent[0]) for sent in ex['sentences']]

            with open('checkpoints/{}/predictions.json'.format(self.hparams.ckpt_name),'w') as f:
                json.dump(combined_predictions, f)
            
            role_mapping = load_role_mapping(self.hparams.dataset)
            arg_id_f1 , arg_f1 = evaluate_arg_f1(combined_predictions, role_mapping)
            log = {
                'val/id_f1':arg_id_f1,
                'val/f1': arg_f1
            } 
            return {
                'f1': arg_f1, 
                'log': log 
            }


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
                "params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.bert_weight_decay,
                "lr":self.hparams.bert_learning_rate,
            },
            {
                "params": [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr":self.hparams.bert_learning_rate,
            },
            {"params": [p for n, p in self.head.named_parameters() if not any(nd in n for nd in no_decay)], 
            "weight_decay": self.hparams.weight_decay
            },
            {"params": [p for n, p in self.head.named_parameters() if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0
            }

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