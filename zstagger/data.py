import os 
import json
import random 
import torch 
import time 
from torch.utils.data import Dataset

# from ACE_data_module import MAX_LENGTH
# For ACE triggers 
# MAX_LENGTH=200 
# For KAIROS
MAX_LENGTH=400 #this needs to be a hyperparameter


def adaptive_length_collate(batch):
    '''
    'doc_key': ex['sent_id'],
    'input_token_ids':input_tokens['input_ids'],
    'input_attn_mask': input_tokens['attention_mask'],
    'labels': labels,
    'bpe_mapping': bpe2word, 
    'token_lens': token_lens, 
    'word_tags': word_tags,
    '''
    doc_keys = [ex['doc_key'] for ex in batch]
    token_lens = [ex['token_lens'] for ex in batch]
    max_len = min(max([sum(ex['input_attn_mask']) for ex in batch]), MAX_LENGTH) 
    batch_size = len(batch)

    input_token_ids = torch.ones((batch_size, max_len), dtype=torch.long)
    input_attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    bpe_mapping = torch.ones((batch_size, max_len), dtype=torch.long) * (-1)
    labels = torch.zeros((batch_size, max_len), dtype=torch.long)
    word_lengths = []

    for i in range(batch_size):
        ex = batch[i]
        word_lengths.append(len(ex['word_tags']))
        l = min(sum(ex['input_attn_mask']), MAX_LENGTH) 
        input_token_ids[i, :l] = torch.LongTensor(ex['input_token_ids'][:MAX_LENGTH])
        input_attn_mask[i, :l] = torch.BoolTensor(ex['input_attn_mask'][:MAX_LENGTH])
        labels[i, :l] = torch.LongTensor(ex['labels'][:MAX_LENGTH])
        bpe_mapping[i, :l] = torch.LongTensor(ex['bpe_mapping'][:MAX_LENGTH])
        
        
    max_word_len = min(max(word_lengths), MAX_LENGTH) 
    word_tags = torch.ones((batch_size, max_word_len), dtype=torch.long) *(-1) 
    for i in range(batch_size):
        ex = batch[i]
        l = min(len(ex['word_tags']), MAX_LENGTH)
        word_tags[i, :l] = torch.LongTensor(ex['word_tags'][:MAX_LENGTH]) 
    
    chunk_idx = [ex['chunk_idx'] if 'chunk_idx' in ex else 0 for ex in batch ]
    return {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'labels': labels,
        'doc_key': doc_keys,
        'bpe_mapping': bpe_mapping,
        'token_lens': token_lens, 
        'word_lengths': torch.LongTensor(word_lengths),
        'word_tags': word_tags,
        'chunk_idx': chunk_idx
    }


class IEDataset(Dataset):
    def __init__(self, input_file, split):
        super().__init__()
        self.examples = []
        self.pos_examples = []
        self.neg_examples = []
        with open(input_file, 'r') as f:
            for line in f:
                ex = json.loads(line.strip())
                if sum(ex['word_tags']) == 0:
                    # no event mention
                    self.neg_examples.append(ex)
                else:
                    self.pos_examples.append(ex)
                
                self.examples.append(ex)
        
        if split == 'train' and len(self.neg_examples) > 2*len(self.pos_examples):
            # downsample negatives 
            
            def seed_random():
                t = int( time.time() * 1000.0 )
                random.seed( ((t & 0xff000000) >> 24) +
                            ((t & 0x00ff0000) >>  8) +
                            ((t & 0x0000ff00) <<  8) +
                            ((t & 0x000000ff) << 24)   )
            seed_random() 
            K = len(self.pos_examples)
            selected_negs = random.sample(self.neg_examples, k=K*2)
            self.examples = self.pos_examples + selected_negs

    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    

