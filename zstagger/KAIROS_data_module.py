import os 
import json 
import argparse

from transformers import AutoTokenizer
from torch.utils.data import DataLoader 
import pytorch_lightning as pl 

from .data import IEDataset, MAX_LENGTH, adaptive_length_collate
from .utils import load_ontology, load_role_mapping, find_ent_span


# For KAIROS trigger extraction 
MAX_CONTEXT_LENGTH=300
MAX_LENGTH=400
# ensure that documents with MAX_CONTENT_LENGTH are not truncated 

WORD_START_CHAR='\u0120'
def get_chunk(ex, window):
    start =0
    for i in range(window[0]):
        start += len(ex['sentences'][i][0])
    end = start 
    for i in range(window[0], window[1]+1):
        end += len(ex['sentences'][i][0])
    
    return (start, end)


class DocTaggerDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams = args 
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)


    def get_trigger_word_tags(self, ex, ontology_dict):
        tokens = ex['tokens']
        tags = [0,] * len(tokens)

        for event in ex['event_mentions']:
            event_type = event['event_type']
            if event_type not in ontology_dict:
                continue 
            for idx in range(event['trigger']['start'], event['trigger']['end']):# idx not inclusive 
                tags[idx] =  ontology_dict[event_type]['i-label']
            
        return tags 

    def get_arg_word_tags(self, ex, role_mapping):
        '''
        Returns a list of word tags, one for each trigger.
        '''
        tokens = ex['tokens']
        tag_list = []
        

        for event in ex['event_mentions']:
            tags = [0,] * len(tokens)
            center_sent = event['trigger']['sent_idx']
            trigger = event['trigger']['text']
            for arg in event['arguments']:
                ent_id = arg['entity_id']
                role = arg['role']
                if role not in role_mapping:
                    continue 
                span = find_ent_span(ex, ent_id) # not inclusive of span end 
              
                # tags[span[0]] = role_mapping[role]['b-label']

                # for idx in range(span[0]+1, span[1]):
                    # tags[idx] = role_mapping[role]['i-label']

                for idx in range(span[0], span[1]):
                    tags[idx] = role_mapping[role]['i-label']
                
            tag_list.append({
                'word_tags': tags,
                'trigger': trigger,
                'center_sent': center_sent
            })
        
        return tag_list 
                

    def get_labels(self, ex, word_tags, pretrained_model, start=0, end=-1):
        if pretrained_model.startswith('roberta'):
            if start!=0:
                raise NotImplementedError
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
            words = ex['tokens'][start:end]
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
        else:
            raise NotImplementedError


    def prepare_data(self):
        if not os.path.exists(self.hparams.tmp_dir):
            os.makedirs(self.hparams.tmp_dir)

            ontology_dict = load_ontology(self.hparams.dataset)
            role_mapping = load_role_mapping(self.hparams.dataset) 
            
            for split,f in [('train',self.hparams.train_file), ('val',self.hparams.val_file), ('test',self.hparams.test_file)]:
                with open(f,'r') as reader,  open(os.path.join(self.hparams.tmp_dir, '{}.jsonl'.format(split)), 'w') as writer:
                    for line in reader:
                        ex = json.loads(line.strip())
                        if split =='train' and len(ex['tokens']) < 4:
                            # removing headers 
                            continue 
                        if self.hparams.task == 'trigger':
                            word_tags = self.get_trigger_word_tags(ex, ontology_dict)
                            # token tags from word tags 
                            # chunking 
                            start = 0
                            # while (start < len(ex['tokens'])):
                            for chunk_idx in range( len(ex['sentences'])):
                                # chunk by sentence 
                                use_ex = True 
                                sentence_tokens, sentence_text  = ex['sentences'][chunk_idx] 
                                sent_len = len(sentence_tokens)
                                chunk = (start, start+sent_len)
                                word_tags_chunk = word_tags[chunk[0]: chunk[1]]
                                tokens_chunk = ex['tokens'][chunk[0]: chunk[1]]

                                start += sent_len
                                if sent_len < 4 and split=='train':
                                    use_ex = False 
                                try: 
                                    assert(len(tokens_chunk) <= MAX_CONTEXT_LENGTH)
                                except AssertionError:
                                    print(len(tokens_chunk), split)
                                    # discard this super long sentence 
                                    use_ex= False 
                                
                                if use_ex:
                                    tokenized = self.tokenizer.tokenize(' '.join(tokens_chunk)) 
                                    try: 
                                        assert(len(tokenized) <= MAX_LENGTH -2)
                                    except AssertionError:
                                        print('Original {}, tokenized {}'.format(len(tokens_chunk), len(tokenized))) 
                                        continue 
                                        #add_prefix_space=True if self.hparams.pretrained_model.startswith('roberta') else False,
                                    input_tokens = self.tokenizer.encode_plus(' '.join(tokens_chunk),
                                        add_special_tokens=True, 
                                        max_length=MAX_LENGTH,
                                        truncation=True, padding=False) 
                                    
                                    # is_pretokenized does not work with bpe at this version of transformers 
                                    labels, bpe2word, token_lens  = self.get_labels(ex, word_tags_chunk, 
                                                                        self.hparams.pretrained_model,
                                                                        start=chunk[0],
                                                                        end=chunk[1])
                                    assert(len(labels) == len(input_tokens['input_ids']))
                                    assert(len(token_lens) == len(word_tags_chunk))
                                    processed_ex = {
                                        'doc_key': ex['doc_id'],
                                        'chunk_idx' : chunk_idx,
                                        'input_token_ids':input_tokens['input_ids'],
                                        'input_attn_mask': input_tokens['attention_mask'],
                                        'labels': labels, # bpe level labels, not used in current version 
                                        'bpe_mapping': bpe2word, 
                                        'token_lens': token_lens, 
                                        'word_tags': word_tags_chunk,
                                    }
                                    writer.write(json.dumps(processed_ex) + '\n')
                        else:
                            word_tag_list = self.get_arg_word_tags(ex, role_mapping)
                            sentences_n = len(ex['sentences'])
                            for evt_idx, word_tag_dict in enumerate(word_tag_list):
                                use_ex = True 
                                # find a three sentence window 

                                center_idx = word_tag_dict['center_sent']
                                word_tags = word_tag_dict['word_tags']
                                trigger = word_tag_dict['trigger']

                                # window = (max(0, center_idx-1), min(sentences_n, center_idx+1))
                                # # get the start and end idx of this window 
                                # chunk = get_chunk(ex, window)
                                # if (chunk[1]-chunk[0]) < 4:
                                #     use_ex = False 
                                # if (chunk[1]-chunk[0])> MAX_CONTEXT_LENGTH:
                                #     # use only one sentence 
                                #     window = (center_idx, center_idx)
                                #     chunk = get_chunk(ex, window)
                                
                                window = (center_idx, center_idx)
                                chunk = get_chunk(ex, window)
                                word_tags_chunk = word_tags[chunk[0]: chunk[1]]
                                tokens_chunk = ex['tokens'][chunk[0]: chunk[1]]
                                try: 
                                    assert(len(tokens_chunk) <= MAX_CONTEXT_LENGTH)
                                except AssertionError:
                                    print(len(tokens_chunk), split)
                                    # discard this super long sentence 
                                    use_ex= False 
                                
                                if use_ex:
                                    tokenized = self.tokenizer.tokenize(' '.join(tokens_chunk)) 
                                    try: 
                                        assert(len(tokenized) <= MAX_LENGTH -2)
                                    except AssertionError:
                                        print('Original {}, tokenized {}'.format(len(tokens_chunk), len(tokenized))) 
                                        continue 
                                        #add_prefix_space=True if self.hparams.pretrained_model.startswith('roberta') else False,
                                    
                                    # [CLS] tokens [SEP] trigger [SEP]
                                    input_tokens = self.tokenizer.encode_plus( ' '.join(tokens_chunk), trigger,
                                        add_special_tokens=True, 
                                        max_length=MAX_LENGTH,
                                        truncation=True, padding=False) 
                                    
                                    trigger_token_len = len(self.tokenizer.tokenize(trigger)) 
                                    # is_pretokenized does not work with bpe at this version of transformers 
                                    labels, bpe2word, token_lens  = self.get_labels(ex, word_tags_chunk, 
                                                                        self.hparams.pretrained_model,
                                                                        start=chunk[0],
                                                                        end=chunk[1])
                                    labels.extend([0,] * (trigger_token_len +1) ) # bpe level labels for "trigger [SEP]"
                                    bpe2word.extend([-1,]* (trigger_token_len +1) )  
                                   
                                    assert(len(labels) == len(input_tokens['input_ids']))
                                    assert(len(token_lens) == len(word_tags_chunk))
                                    

                                    if len(token_lens) > 0: 
                                        processed_ex = {
                                            'doc_key': '{}:{}'.format(ex['doc_id'], evt_idx),
                                            'chunk_idx' : 0,
                                            'input_token_ids':input_tokens['input_ids'],
                                            'input_attn_mask': input_tokens['attention_mask'],
                                            'labels': labels, # bpe level labels, not used in current version 
                                            'bpe_mapping': bpe2word, 
                                            'token_lens': token_lens, 
                                            'word_tags': word_tags_chunk,
                                        }
                                        writer.write(json.dumps(processed_ex) + '\n')

                        

                                
            


    
    def train_dataloader(self):
        print('reading from {}'.format(self.hparams.tmp_dir))
        if self.hparams.use_pl:
            dataset = IEDataset(os.path.join(self.hparams.tmp_dir, 'pl_train.jsonl'), split='pl')
        else:
            dataset = IEDataset(os.path.join(self.hparams.tmp_dir,'train.jsonl'),split='train')
        
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
    parser.add_argument('--train-file',type=str,default='data/kairos/train.jsonl')
    parser.add_argument('--val-file', type=str, default='data/kairos/dev.jsonl')
    parser.add_argument('--test-file', type=str, default='data/kairos/test.jsonl')
    parser.add_argument('--pretrained-model', type=str, default='bert-large-cased')
    parser.add_argument('--task', default='arg')
    parser.add_argument('--dataset', default='KAIROS')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--tmp-dir', type=str, default='tag_kairos_arg')
    parser.add_argument('--use-pl', action='store_true', default=False)
    args = parser.parse_args() 

    dm = DocTaggerDataModule(args=args)
    dm.prepare_data() 

    # training dataloader 
    dataloader = dm.train_dataloader() 

    for idx, batch in enumerate(dataloader):
        print(batch)
        if idx==5:
            break 

    # val dataloader 