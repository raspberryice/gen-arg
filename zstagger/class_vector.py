
import torch 
from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm 
import argparse 
import json 


from utils import load_ontology, expand_keywords_inflection, MAX_LENGTH

RARE_TYPES = {'Justice:Appeal','Justice:Extradite','Justice:Acquit', 'Justice:Convict', "Justice:Sentence",
    "Justice.Acquit.Unspecified",
    "Medical.Vaccinate.Unspecified",
    "Movement.Transportation.IllegalTransportation",
    "ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse",
    "ArtifactExistence.DamageDestroyDisableDismantle.Dismantle"
    } 



def get_label_occurrence(ex, evt_type):
    '''
    ex: a dict object with 'document' and 'event_mentions' key.
    evt_type: event type name from ontology.

    Assumes that only one instance of the evt_type appears in the sentence, which could be unsuitable.

    return:
    matches: tuples of (trigger, word index)
    '''
    match = None 
    for e in ex['event_mentions']:
        if e['event_type'] == evt_type:
            trigger = e['trigger']['text']
            widx = e['trigger']['start']
            match = (trigger, widx)
            break 
    if match:
        return {
        'document': ex['document'],
        'match': match,
    }
    else:
        return None 

def get_keyword_occurrence(ex, keywords):
    '''
    ex: a Gigaword document with 'document' key.
    keywords: set of keywords

    return: 
    matches: tuples of (keyword, word index)
    '''
    match = None 
    words = ex['document'].split() 
    for widx, w in enumerate(words):
        if w in keywords:
            match=(w, widx)
            # only match one instance of one event type per sentence 
            break 
    if match:
        return {
        'document': ex['document'],
        'match': match,
    }
    else:
        return None 


def get_substitute(m_dict_list, model, tokenizer, top_k=50, use_mask=False, strategy='first', bert_dim=1024):
    '''
    m_dict: List of match dictionary with 'document' and 'match' keys.
    model: a pretrained bert model 
    strategy: 'first' use first subword token, 'mean' average tokens. only useful when use_mask=False. Not implemented for batch.

    '''
    def tokenize_instance(m_dict):
        words = m_dict['document'].split()
        match_idx = m_dict['match'][1]
        prefix = tokenizer.tokenize(' '.join(words[:match_idx]) )
        mask_idx = len(prefix) +1 # add [CLS]
        token_len =1 
        if use_mask:
            suffix = tokenizer.tokenize(' '.join(words[match_idx+1:]))
            encoded = tokenizer.encode(prefix + [tokenizer.mask_token, ] + suffix, 
                add_special_tokens=True, 
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH)
        else:
            token_len = len(tokenizer.tokenize(words[match_idx]))
            suffix = tokenizer.tokenize(' '.join(words[match_idx:]))
            encoded = tokenizer.encode(prefix + suffix, 
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH)
        return {
            'encoded': torch.LongTensor(encoded), 
            'mask_idx': mask_idx,
            'token_len': token_len
        }
    tokenized_batch = [tokenize_instance(ex) for ex in m_dict_list]
    batch_size = len(tokenized_batch)
    mask_idx_batch = torch.LongTensor([ex['mask_idx'] for ex in tokenized_batch]).to(model.device) # (batch)
    token_len_batch = torch.LongTensor([ex['token_len'] for ex in tokenized_batch]).to(model.device)
    token_ids = torch.stack([ex['encoded'] for ex in tokenized_batch]).to(model.device) #(batch, max_len) 
    
    outputs = model(token_ids, output_hidden_states=True)#(batch, max_len, hidden_dim)
    vocab_size = outputs[0].size(2) 
    prediction_scores = torch.gather(outputs[0], 1, 
         mask_idx_batch.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, vocab_size)).squeeze(1) #(batch, vocab)
    vec = torch.gather(outputs[1][-1], 1,  
        mask_idx_batch.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, bert_dim)).squeeze(1) #(batch, hidden_dim)
    vec = vec.cpu() 
    # prediction_scores = outputs[0].squeeze(0)[mask_idx]
    # if use_mask or strategy == 'first' :
    #     vec = outputs[1][-1].squeeze(0)[mask_idx]
    # else:
    #     vec = torch.mean(outputs[1][-1].squeeze(0)[mask_idx: mask_idx+ token_len])
        
    
    # decode one by one 
    top_scores = torch.argsort(prediction_scores, dim=1, descending=True)[:, :top_k]
    top_scores = top_scores.cpu()
    top_sub_list = []
    for i in range(batch_size):
        top_subs = tokenizer.decode(top_scores[i, :]).split()
        top_sub_list.append(top_subs)
    return top_sub_list , vec

def compute_sub_score(top_subs, match, keywords):
    score = 0 
    for idx, sub in enumerate(top_subs):
        rank = idx +1 
        if sub == match['match'][0]:
            score += 1 
        elif sub in keywords:
            score += 1/ rank 
    return score 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gigaword')
    parser.add_argument('--data_file', type=str, default='data/ace/pro_mttrig_id/json/train.oneie.json')
    parser.add_argument('--ontology', type=str, default='KAIROS', choices=['ACE', 'KAIROS'])
    parser.add_argument('--keyword_n', type=int, default=3)
    parser.add_argument('--min_score', type=float, default=1.0)
    parser.add_argument('--keep_all', action='store_true')
    parser.add_argument('--from_labels', action='store_true')
    args = parser.parse_args() 


    ontology_dict = load_ontology(args.ontology)
    docs = []
    if args.dataset == 'gigaword':
        gigaword = load_dataset('gigaword')
        docs = gigaword['train']
    elif args.dataset == 'training':
        docs = []
        with open(args.data_file,'r') as f:
            for line in f:
                ex = json.loads(line)
                docs.append({
                    'document': ex['sentence'],
                    'event_mentions': ex['event_mentions']
                })
    

    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BertForMaskedLM.from_pretrained('bert-large-cased')

    model = model.to('cuda:0')
    BATCH_SIZE=64

    event_vectors = {} 
    for eidx, event in enumerate(ontology_dict):
        keywords = set(ontology_dict[event]['keywords'][:args.keyword_n])
        keywords = expand_keywords_inflection(keywords)
        matches = [] 
        for i in range(len(docs)):
            ex = docs[i]
            if args.from_labels:
                m_dict = get_label_occurrence(ex, event)
            else:
                m_dict = get_keyword_occurrence(ex, keywords)
            
            if m_dict:
                matches.append(m_dict)
            if len(matches) == 100:
                break 
        if len(matches) == 0:
            print('{} has no occurrences'.format(event))
            continue 

        vec_list = []
        accepted_m = []
        start_idx = 0
        with torch.no_grad():
            with tqdm(total=len(matches)) as pbar:
                while start_idx < len(matches):
                    match_list = matches[start_idx:start_idx + BATCH_SIZE] 
                    top_sub_list, vec = get_substitute(match_list, model, tokenizer)
                    for i in range(len(match_list)):
                        score = compute_sub_score(top_sub_list[i], match_list[i], keywords)
                        if event in RARE_TYPES or len(keywords) == 1 or score > args.min_score:
                            vec_list.append(vec[i, :])
                            accepted_m.append(match_list[i])
                    start_idx += BATCH_SIZE
                    pbar.update(len(match_list))
                
       
       

        if len(vec_list) == 0:
            print('{} has no accepted occurrences'.format(event))
            continue 
        if args.keep_all:
            # don't do average
            class_vector = torch.stack(vec_list, dim=0)
        else:
            class_vector = torch.stack(vec_list, dim=0).mean(dim=0) 
        event_vectors[event] = class_vector

    with open('class_vectors_{}_{}.pkl'.format(args.ontology,args.dataset),'wb') as f:
        torch.save(event_vectors, f)

    C = len(ontology_dict)
    # convert dictionary into single matrix for model input 
    all_cv = torch.zeros((C+1, 1024))
    for e in ontology_dict:
        vector = event_vectors[e]
        idx = ontology_dict[e]['i-label']
        all_cv[idx, :] = vector
    torch.save(all_cv[1:, :], 'all_class_vec_{}.pt'.format(args.ontology))