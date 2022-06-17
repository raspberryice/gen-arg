'''
Based on the keywords from the ontology, create pseudo labels for the training set. 
'''

import os 
import json
import argparse 
from typing import List, Dict, Tuple, Optional 

import torch 
from transformers import BertTokenizer
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from nltk.corpus import stopwords 


from data import IEDataset, adaptive_length_collate
from utils import load_ontology, expand_keywords_inflection
from layers import BERTEncoder
BATCH_SIZE=16

def get_vector_labels(ex:Dict, class_vectors: torch.FloatTensor, idx2event:Dict[int, str],
     pos_thres: float=0.65, unk_thres: float=0.4, keyword_bonus: float=0.1):
    '''
    Assign pseudo label to single instance. 
    '''
    vec = ex['vec'] # type: torch.FloatTensor 
    sim = F.normalize(vec, dim=1) @ F.normalize(class_vectors,dim=1).t() 
    max_sim_score, max_idx = torch.max(sim, dim=1)
    max_class = max_idx +1 
    stop_words = set(stopwords.words('english')) 
    predicted_mentions = []
    total_pred =0 
    for i, score in enumerate(max_sim_score.tolist()):
        evt_name = idx2event[max_class[i].item()]
        keywords = ontology_dict[evt_name]['keywords']
        keywords = expand_keywords_inflection(set(keywords))
        text = ex['tokens'][i]
        if text in stop_words: # stopwords are always O-label
            continue 
        if ex['tokens'][i] in keywords: 
            score += keyword_bonus
        if score >= pos_thres:
            
            predicted_mentions.append({
                'evt_type': evt_name,
                'start': i,
                'score': score,
                'text': ex['tokens'][i]
            }
                )
            total_pred +=1 
        elif score >=unk_thres: # uncertain 
            predicted_mentions.append({
                'evt_type': 'unknown',
                'start': i,
                'score': score,
                'text': ex['tokens'][i]
            })
    
    if len(predicted_mentions) == 0:
        return None, (0, 0, 0)

    res = {
        'doc_key': ex['sent_id'],
        'sentence': ex['sentence'],
        'gold_mentions':[],
        'predicted_mentions':predicted_mentions,
    }
    total_matched =0 
    total_gold =0 
    for e in ex['event_mentions']:
        gold = {
            'evt_type': e['event_type'],
            'start': e['trigger']['start'],
            'score': max_sim_score[e['trigger']['start']].item(),
            'text': e['trigger']['text']
        }
        total_gold +=1 
        for m in predicted_mentions:
            if m['start'] == gold['start'] and m['evt_type'] == gold['evt_type']:
                total_matched += 1 
            if m['start'] == gold['start'] and m['evt_type'] == 'unknown':
                total_gold -=1  # ignore this gold mention 
        res['gold_mentions'].append(gold)
    
    return res , (total_gold, total_pred, total_matched)



def move_batch(batch:Dict)-> Dict:
    moved_batch = {} 
    for k, v in batch.items():
        if hasattr(v, 'dtype'):
            # move to cuda 
            moved_batch[k] = v.to('cuda:0')
        else:
            moved_batch[k] = v 
    return moved_batch 



def embed_instances(args, encoder: BERTEncoder)-> None:
    if not os.path.exists(args.tmp_dir):
        raise FileNotFoundError
    
    dataset = IEDataset(os.path.join(args.tmp_dir, 'train.jsonl'),split='pl') 
    dataloader = DataLoader(dataset, 
            pin_memory=True, num_workers=4, 
            collate_fn=adaptive_length_collate,
            batch_size=BATCH_SIZE, 
            shuffle=True)

    encoder = encoder.to('cuda:0')
    encoder.eval() 

    results = {} # doc_key -> tensor of (word_seq_len, hidden_dim) 
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch(batch)
            features = encoder(input_ids=batch['input_token_ids'], 
                attention_mask=batch['input_attn_mask'],
                token_lens=batch['token_lens']) # (batch,word_seq_len, hidden_dim)
            doc_key = batch['doc_key']
            lengths = batch['word_lengths']
            # removing padding 
            batch_size = features.size(0)
            for i in range(batch_size):
                vec = features[i, :lengths[i], :].cpu() 
                results[doc_key[i]] = vec 
            
    
    torch.save(results, f'training_embedded_{args.ontology}.pt')
    print('training embedding saved ....')
    return 


def assign_pl(pos_exs: List, neg_exs:List, class_vectors: torch.FloatTensor, idx2event: Dict[int, str], 
        output_file: str='pl_training.jsonl', log_file: str='pl_label_errors.jsonl', 
        pos_thres: float=0.65, unk_thres:float=0.4, keyword_bonus: float=0.1):
    '''
    :pos_thres: the instance has to score higher than this to be considered as positive 
    :unk_thres: if the instance is higher than this and lower than pos_thres, then considered as unknown 
    :keyword_bonus: if the keyword matches, then assign this bonus score  
    '''
    writer = open(output_file,'w')
    error_logs = open(log_file,'w')

    total_gold = 0
    total_pred = 0
    total_matched = 0
    for ex in pos_exs:
        res, stats  = get_vector_labels(ex, class_vectors, idx2event, pos_thres, unk_thres, keyword_bonus)
        if res == None or stats[1]==0:
            continue 
        writer.write(json.dumps(res) + '\n')
        total_gold += stats[0]
        total_pred += stats[1]
        total_matched += stats[2] 

        if stats[0] != stats[2]:
            error_logs.write(json.dumps(res) + '\n')
        

    for ex in neg_exs:# nothing is being predicted 
        vec = ex['vec']
        sim = F.normalize(vec, dim=1) @ F.normalize(class_vectors,dim=1).t() 
        max_sim_score, max_idx = torch.max(sim, dim=1)
        if torch.max(max_sim_score).item() > unk_thres: # grey area 
            continue 
        else: # use as negative 
            res = {
            'doc_key': ex['sent_id'],
            'sentence': ex['sentence'],
            'gold_mentions':[],
            'predicted_mentions':[],
            }
            for e in ex['event_mentions']:
                gold = {
                    'evt_type': e['event_type'],
                    'start': e['trigger']['start'],
                    'score': max_sim_score[e['trigger']['start']].item(),
                    'text': e['trigger']['text']
                }
                res['gold_mentions'].append(gold)
                total_gold += len(res['gold_mentions'])

            if len(res['gold_mentions']) != 0:
                error_logs.write(json.dumps(res) + '\n')
            writer.write(json.dumps(res) + '\n')
    writer.close() 
    # check label quality 
    recall = total_matched *1.0/ total_gold
    precision = total_matched * 1.0/total_pred 
    f1 = 2* recall * precision/ (recall+precision)

    print('R:{}, P:{}, F1:{}'.format(recall, precision, f1))

    return 




def convert_pl_file(args, pl_file_name='pl_training.jsonl')-> None:
    pl_exs = {}
    with open(pl_file_name,'r') as f:
        for line in f:
            ex = json.loads(line)
            pl_exs[ex['doc_key']] = ex['predicted_mentions']
    
    writer = open(os.path.join(args.tmp_dir, 'pl_train.jsonl'),'w')
    with open(os.path.join(args.tmp_dir, 'train.jsonl'),'r') as f:
        for line in f:
            ex = json.loads(line)
            if ex['doc_key'] not in pl_exs:
                continue 
            pred = pl_exs[ex['doc_key']]
            pl_word_tags = [0,] * len(ex['token_lens'])
            for e in pred:
                evt_type = e['evt_type']
                i = e['start']
                if evt_type == 'unknown':
                    pl_word_tags[i] = -1
                else:
                    label_idx = ontology_dict[evt_type]['i-label']
                    pl_word_tags[i] = label_idx
            processed_ex = {
                'doc_key': ex['doc_key'],
                'input_token_ids':ex['input_token_ids'],
                'input_attn_mask': ex['input_attn_mask'],
                'labels': ex['labels'],
                'bpe_mapping': ex['bpe_mapping'], 
                'token_lens': ex['token_lens'], 
                'word_tags': pl_word_tags,
            }
            writer.write(json.dumps(processed_ex) +'\n')
    writer.close() 
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ontology', type=str, default='KAIROS', choices=['ACE','KAIROS'])
    parser.add_argument('--keyword_n',type=int, default=3, help='number of keywords to use for each event type.')
    parser.add_argument('--tmp-dir',type=str, help='directory for preprocessed data. Will raise error if this directory does not exist.')
    parser.add_argument('--data_file', type=str, default='data/ace/pro_mttrig_id/json/train.oneie.json')
    parser.add_argument('--class-vectors-file',type=str)
    args = parser.parse_args() 


    ontology_dict = load_ontology(args.ontology)

    idx2event = {} 
    for e in ontology_dict:
        eidx = ontology_dict[e]['i-label']
        idx2event[eidx] = e 
    
    all_keywords = set() 
    for e in ontology_dict:
        keywords = set(ontology_dict[e]['keywords'][:args.keyword_n])
        keywords = expand_keywords_inflection(keywords)
        all_keywords.update(keywords)
    
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    encoder = BERTEncoder(args, bert_dim=1024)

    
    if not os.path.exists(f'training_embedded_{args.ontology}.pt'):
        embed_instances(args, encoder)
    training_embedded = torch.load(f'training_embedded_{args.ontology}.pt')

    # collect positive and negative instances from the training data 
    training_exs = []
    pos_exs = []
    neg_exs = []
    with open(args.data_file) as f:
        for line in f:
            ex = json.loads(line)
            doc_key = ex['sent_id']
            if doc_key not in training_embedded:
                continue 
            vec = training_embedded[doc_key]
            ex['vec'] = vec 
            has_keyword=False 
            for token in ex['tokens']:
                if token in all_keywords:
                    pos_exs.append(ex)
                    has_keyword = True 
                    break 
            if not has_keyword:
                neg_exs.append(ex)
            
            training_exs.append(ex)

    
    # assign pseudo labels 
    class_vectors = torch.load('all_class_vec_{}.pt'.format(args.ontology)) # type: torch.FloatTensor 
    assign_pl(pos_exs, neg_exs, class_vectors, idx2event, output_file='pl_training.jsonl', log_file='pl_label_errors.jsonl')
    # convert pl file to preprocessed file format 
    convert_pl_file(args, pl_file_name='pl_training.jsonl')
    
    

    
