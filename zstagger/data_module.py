import os 
import json 
import argparse

from transformers import AutoTokenizer
from torch.utils.data import DataLoader 
import pytorch_lightning as pl 

from .data import IEDataset, adaptive_length_collate
from .utils import load_ontology
MAX_LENGTH=200
WORD_START_CHAR='\u0120'

class TaggerDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams = args 
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)


    def get_word_tags(self, ex, ontology_dict):
        tokens = ex['tokens']
        tags = [0,] * len(tokens)

        for event in ex['event_mentions']:
            event_type = event['event_type']
            for idx in range(event['trigger']['start'], event['trigger']['end']):
                tags[idx] =  ontology_dict[event_type]['i-label']
            
        return tags 

    def get_labels(self, ex, word_tags, pretrained_model):
        if pretrained_model.startswith('roberta'):
            bpe_tokens = self.tokenizer.tokenize(ex['sentence'], add_prefix_space=True)
                        
            widx =-1
            bpe_tags = []
            token_lens = [1, ] * len(word_tags)
            bpe2word = []
            for b in bpe_tokens:
                if b[0] == WORD_START_CHAR:
                    widx +=1 
                else:
                    token_lens[widx]+=1 
                bpe_tags.append(word_tags[widx])
                bpe2word.append(widx)
                
            assert(len(bpe_tags) == len(bpe_tokens))

            labels = [0, ] + bpe_tags[:MAX_LENGTH-2] + [0,] # 0 for <s> token and  0 for </s> token 
            bpe2word = [-1, ] + bpe2word + [-1, ] 
            return labels, bpe2word, token_lens

        elif pretrained_model.startswith('bert'):
            words = ex['tokens']
            bpe_tags = []
            token_lens = [1, ] * len(word_tags)
            bpe2word = []
            for widx, w in enumerate(words):
                bpe_tokens = self.tokenizer.tokenize(w)
                token_lens[widx] = len(bpe_tokens)
                bpe_tags.extend([word_tags[widx],] * len(bpe_tokens))
                bpe2word.extend([widx,] * len(bpe_tokens))
            

            labels = [0, ] + bpe_tags[:MAX_LENGTH-2] + [0,] # 0 for [CLS] token and  0 for [SEP] token 
            bpe2word = [-1, ] + bpe2word + [-1, ] 
            return labels, bpe2word, token_lens


    def prepare_data(self):
        if not os.path.exists(self.hparams.tmp_dir):
            print('preprocessing data to {}'.format(self.hparams.tmp_dir))
            
            os.makedirs(self.hparams.tmp_dir)

            ontology_dict = load_ontology(self.hparams.dataset) 
            
            for split,f in [('train',self.hparams.train_file), ('val',self.hparams.val_file), ('test',self.hparams.test_file)]:
                with open(f,'r') as reader,  open(os.path.join(self.hparams.tmp_dir, '{}.jsonl'.format(split)), 'w') as writer:
                    for line in reader:
                        ex = json.loads(line.strip())
                        if split =='train' and len(ex['tokens']) < 4:
                            # removing headers 
                            continue 
                        word_tags = self.get_word_tags(ex, ontology_dict)
                        # token tags from word tags 
                        
                        input_tokens = self.tokenizer.encode_plus(ex['sentence'],
                            add_prefix_space=True if self.hparams.pretrained_model.startswith('roberta') else False,
                            add_special_tokens=True, 
                            max_length=MAX_LENGTH,
                            truncation=True, padding=False) 
                            # is_pretokenized does not work with bpe at this version of transformers 
                        labels, bpe2word, token_lens  = self.get_labels(ex, word_tags, self.hparams.pretrained_model)
                        assert(len(labels) == len(input_tokens['input_ids']))
                        
                        processed_ex = {
                            'doc_key': ex['sent_id'],
                            'input_token_ids':input_tokens['input_ids'],
                            'input_attn_mask': input_tokens['attention_mask'],
                            'labels': labels,
                            'bpe_mapping': bpe2word, 
                            'token_lens': token_lens, 
                            'word_tags': word_tags,
                        }
                        writer.write(json.dumps(processed_ex) + '\n')
            


    
    def train_dataloader(self):
        if self.hparams.use_pl:
            dataset = IEDataset(os.path.join(self.hparams.tmp_dir, 'pl_train.jsonl'), split='pl')
        else:
            dataset = IEDataset('tag_data/train.jsonl',split='train')
        
        dataloader = DataLoader(dataset, 
            pin_memory=True, num_workers=4, 
            collate_fn=adaptive_length_collate,
            batch_size=self.hparams.train_batch_size, 
            shuffle=True)
        return dataloader 

    
    def val_dataloader(self):
        dataset = IEDataset(os.path.join(self.hparams.tmp_dir, 'val.jsonl'),split='val')
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=4, 
            collate_fn=adaptive_length_collate,
            batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        dataset = IEDataset(os.path.join(self.hparams.tmp_dir,'test.jsonl'),split='test')
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=4, 
            collate_fn=adaptive_length_collate, 
            batch_size=self.hparams.eval_batch_size, shuffle=False)

        return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train-file',type=str)
    parser.add_argument('--val-file', type=str)
    parser.add_argument('--test-file', type=str)
    parser.add_argument('--pretrained-model', type=str, default='bert-large-cased')
    parser.add_argument('--dataset', default='ACE')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    args = parser.parse_args() 

    dm = TaggerDataModule(args=args)
    dm.prepare_data() 

    # training dataloader 
    dataloader = dm.train_dataloader() 

    for idx, batch in enumerate(dataloader):
        print(batch)
        if idx==5:
            break 
