import os 
import json 

import torch 
from torch.utils.data import DataLoader, Dataset

def my_collate(batch):
    '''
    'doc_key': ex['doc_key'],
    'input_token_ids':input_tokens['input_ids'],
    'input_attn_mask': input_tokens['attention_mask'],
    'tgt_token_ids': tgt_tokens['input_ids'],
    'tgt_attn_mask': tgt_tokens['attention_mask'],
    '''
    doc_keys = [ex['doc_key'] for ex in batch]
    input_token_ids = torch.stack([torch.LongTensor(ex['input_token_ids']) for ex in batch]) 
    input_attn_mask = torch.stack([torch.BoolTensor(ex['input_attn_mask']) for ex in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(ex['tgt_token_ids']) for ex in batch]) 
    tgt_attn_mask = torch.stack([torch.BoolTensor(ex['tgt_attn_mask']) for ex in batch])

    return {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
        'doc_key': doc_keys,
    }


class IEDataset(Dataset):
    def __init__(self, input_file):
        super().__init__()
        self.examples = []
        with open(input_file, 'r') as f:
            for line in f:
                ex = json.loads(line.strip())
                self.examples.append(ex)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    

