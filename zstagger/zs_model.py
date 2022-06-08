import os 
import torch 
import logging 
import json 
from collections import defaultdict

import pytorch_lightning as pl 
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch_struct import LinearChainCRF
import torch.nn.functional as F

from .layers import CRFFeatureHead, BERTEncoder, ZeroShotCRFFeatureHead, ZeroShotCollapsedTransitionCRFHead, PrototypeNetworkHead
from .utils import load_ontology, evaluate_trigger_f1


logger = logging.getLogger(__name__)

class ZSTaggerModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams = args 
    

        self.config=AutoConfig.from_pretrained(args.pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        self.n_classes = self.hparams.event_n +1 # IO tagging 
        # network internals 
        self.encoder = BERTEncoder(args, self.config.hidden_size)
        self.feature_size = self.config.hidden_size  
        
        if self.hparams.dataset == 'ACE':
            class_vectors = torch.load('all_class_vec.pt') 
        elif self.hparams.dataset == 'KAIROS':
            class_vectors = torch.load('all_class_vec_KAIROS.pt')
            print('loading KAIROS vectors...')

        assert(class_vectors.shape[0] == self.hparams.event_n)
        
        if self.hparams.model == 'zs-crf':
            self.head = ZeroShotCollapsedTransitionCRFHead(self.hparams, self.feature_size, 
                self.n_classes, 
                self.hparams.proj_dim,
                class_vectors,)
        elif self.hparams.model == 'proto':
            self.head = PrototypeNetworkHead(self.hparams, self.feature_size, self.n_classes,class_vectors)


    def forward(self, inputs, stage='training'):

        lengths = inputs['word_lengths']
        features = self.encoder(input_ids=inputs['input_token_ids'], 
            attention_mask=inputs['input_attn_mask'],
            token_lens=inputs['token_lens']) # (batch,word_seq_len, hidden_dim)
        if self.hparams.token_classification:
            scores = self.head(features, lengths) # (batch, seq, C)
            batch_size, seq, C = scores.shape 
            if stage == 'training':
                labels = inputs['word_tags'] # batch, seq
                loss = F.cross_entropy(scores.reshape(-1, C), labels.reshape(-1), ignore_index=-1, reduction='sum') 
                loss = loss/ batch_size # per sequence loss 
                return loss 
            else:
                preds = torch.argmax(scores, dim=2)
                return preds 


        else:
            dist = self.head(features, lengths)
            

            if stage=='training':
                assert(torch.max(lengths) == inputs['word_tags'].size(1))
                assert(torch.max(inputs['word_tags']) < self.n_classes) # should be [0, C-1]
                label_ = LinearChainCRF.struct.to_parts(inputs['word_tags'], self.n_classes,
                                                        lengths=lengths).type_as(dist.log_potentials)
                
                
                loss = -dist.log_prob(label_).mean() 
                if hasattr(self.head, 'regularize_params'):
                    reg = self.head.regularize_params() 
                    loss += self.hparams.reg_weight * reg 
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
        with open(self.hparams.val_file,'r') as f:
            for line in f:
                ex = json.loads(line)
                if self.hparams.dataset == 'ACE':
                    combined_predictions[ex['sent_id']]['event_mentions'] = ex['event_mentions']
                else:
                    combined_predictions[ex['doc_id']]['event_mentions'] = ex['event_mentions']
        
       
        tgr_id_f1, tgr_f1 = evaluate_trigger_f1(combined_predictions, ontology_dict)

        log = {
            'val/tgr_id_f1': torch.Tensor([tgr_id_f1,]),
            'val/tgr_f1': torch. Tensor([tgr_f1,])
        } 
        return {
            'f1': tgr_f1, 
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

        with open(self.hparams.test_file,'r') as f:
            for line in f:
                ex = json.loads(line)
                if self.hparams.dataset == 'ACE':
                    combined_predictions[ex['sent_id']]['event_mentions'] = ex['event_mentions']
                else:
                    combined_predictions[ex['doc_id']]['event_mentions'] = ex['event_mentions']
        
        tgr_id_f1, tgr_f1 = evaluate_trigger_f1(combined_predictions, ontology_dict)

        with open('checkpoints/{}/predictions.json'.format(self.hparams.ckpt_name),'w') as f:
            json.dump(combined_predictions, f)

        log = {
            'test/tgr_id_f1': torch.Tensor([tgr_id_f1,]),
            'test/tgr_f1': torch. Tensor([tgr_f1,])
        } 
        return {
            'f1': tgr_f1, 
            'log': log 
        }
        

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False, 
        using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        if hasattr(self.head, 'normalize_params'):
            self.head.normalize_params() 

        if batch_idx % 20== 0:
            if hasattr(self.head, 'update_params'):
                self.head.update_params() 
        return 



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