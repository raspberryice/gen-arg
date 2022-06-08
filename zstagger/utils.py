import json 
from lemminflect import getInflection

### Constants 
MAX_LENGTH=200
WORD_START_CHAR='\u0120'
from spacy.tokens import Doc

PRONOUN_FILE='pronoun_list.txt'
pronoun_set = set() 
with open(PRONOUN_FILE, 'r') as f:
    for line in f:
        pronoun_set.add(line.strip())
    

def check_pronoun(text):
    if text.lower() in pronoun_set:
        return True 
    else:
        return False 

def expand_keywords_inflection(keywords):
    '''
    Takes a list of keywords and return the expanded list.
    '''
    VERB_TAGS = ['VB', 'VBD', 'VBG', 'VBN','VBP', 'VBZ']
    results = set()
    for keyword in keywords:
        results.add(keyword)
        for tag in VERB_TAGS:
            inflected = getInflection(keyword, tag=tag)
            results.update(set(inflected))
    return list(results)

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)

def find_head(arg_start, arg_end, doc):
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <=arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head 
            break 
        else:
            cur_i = doc[cur_i].head.i
        
    arg_head = cur_i
    
    return (arg_head, arg_head)

### Utilities 
def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def load_ontology(dataset):
        '''
        Read ontology file for event to argument mapping.
        ''' 
        ontology_dict ={} 
        with open('event_role_{}.json'.format(dataset),'r') as f:
            ontology_dict = json.load(f)

        for evt_name, evt_dict in ontology_dict.items():
            for i, argname in enumerate(evt_dict['roles']):
                evt_dict['arg{}'.format(i+1)] = argname
                # argname -> role is not a one-to-one mapping 
                if argname in evt_dict:
                    evt_dict[argname].append('arg{}'.format(i+1))
                else:
                    evt_dict[argname] = ['arg{}'.format(i+1)]
        
        return ontology_dict

def load_role_mapping(dataset):
    '''
    Get label mapping for arg extraction.
    '''
    with open('role_label_{}.json'.format(dataset), 'r') as f:
        role_mapping = json.load(f)
    
    return role_mapping



# (start, end, type)
def get_pred_tgr_mentions(ex, tag2event_type):
    pred_mentions = set() 
    prev_tag = 0 
    cur_start = 0
    cur_end = 0 
    if 'bpe_mapping' in ex: # need to map bpe tokens to words 
        for i in range(len(ex['input_ids'])):
            pred_tag = ex['pred_tags'][i]
            word_idx = ex['bpe_mapping'][i]
            if (word_idx != -1) and (word_idx != ex['bpe_mapping'][i-1]):
                # predicting the beginning of a word 
                if (pred_tag > 0):
                    if (prev_tag !=pred_tag):
                        if prev_tag > 0:
                            # end the previous span 
                            pred_mentions.add((cur_start, cur_end, tag2event_type[prev_tag]))
                        # the beginning of a new span 
                        cur_start = word_idx 
                        cur_end = word_idx +1
                    
                    else: cur_end += 1# continue the current span 
                else:
                    if (prev_tag > 0 ):
                        # end of a span
                        pred_mentions.add((cur_start, cur_end, tag2event_type[prev_tag]))
                
                prev_tag = pred_tag 
    else:
        for i in range(len(ex['pred_tags'])):
            pred_tag = ex['pred_tags'][i]
            if pred_tag > 0:
                if (prev_tag!= pred_tag):
                    if (prev_tag>0):
                        # close the prev tag 
                        pred_mentions.add((cur_start, cur_end, tag2event_type[prev_tag]))
                    cur_start = i
                    cur_end = i+1 
                else:
                    cur_end = i+1 
            else:
                if (prev_tag >0):
                    pred_mentions.add((cur_start, cur_end, tag2event_type[prev_tag]))
            
            prev_tag = pred_tag 
                
                    
    return pred_mentions 

def get_pred_arg_mentions_io(ex, tag2role):
    mentions = set() 
    prev_tag = 0 
    cur_start = 0
    cur_end = 0 
    for i in range(len(ex['pred_tags'])):
        tag = ex['pred_tags'][i]
        if tag > 0: # not a O-tag 
            if tag != prev_tag: # begin a new span 
                if prev_tag > 0: 
                    # close the prev tag 
                    mentions.add((cur_start, cur_end, tag2role[prev_tag]))
                cur_start = i 
                cur_end = i+1 
            else: # should be a continuation 
                cur_end = i+1 
        else: # tag is O
            if prev_tag >0:
                mentions.add((cur_start, cur_end, tag2role[prev_tag]))
        
        prev_tag = tag 
    # last tag 
    if prev_tag > 0:
        mentions.add((cur_start, cur_end, tag2role[prev_tag]))
    
    return mentions 


def get_pred_arg_mentions_bio(ex, tag2role, b_tags, i_tags):
    '''
    Return a set of predicted args.
    Predicted tags are on the word level.
    '''
    mentions = set() 
    prev_tag = 0 
    cur_start = 0
    cur_end = 0 
    cur_role = None 
    for i in range(len(ex['pred_tags'])):
        tag = ex['pred_tags'][i]
        if tag > 0: # not a O-tag 
            if tag in b_tags: # begin a new span 
                if prev_tag > 0: 
                    # close the prev tag 
                    mentions.add((cur_start, cur_end, tag2role[prev_tag]))
                cur_start = i 
                cur_end = i+1 
                cur_role = tag2role[tag]
            elif tag in i_tags: # should be a continuation 
                if cur_role == tag2role[tag]:
                    # labeling is correct 
                    cur_end = i+1 
                else:
                    # labeling is wrong, ignore this 
                    tag = 0
        else: # tag is O
            if prev_tag >0:
                mentions.add((cur_start, cur_end, tag2role[prev_tag]))
        
        prev_tag = tag 
    # last tag 
    if prev_tag > 0:
        mentions.add((cur_start, cur_end, tag2role[prev_tag]))
    
    return mentions 
                


    

def find_ent_span(ex, ent_id, offset=0):
    '''
    The ent_span from entity_mentions is from the document.
    If the document has been chunked, an offset is needed to align with the predictions.
    '''
    matched_ent = [ent for ent in ex['entity_mentions'] if ent['id'] == ent_id][0]
    return (matched_ent['start']-offset, matched_ent['end']-offset)


def get_tag_mapping(role_mapping):
        tag2role = {} 
        b_tags = set() 
        i_tags = set() 
        for role in role_mapping:
            if 'b-label' in role_mapping[role]:
                tag = role_mapping[role]['b-label']
                b_tags.add(tag)
                tag2role[tag] = role 
            tag = role_mapping[role]['i-label']
            i_tags.add(tag)
            tag2role[tag] = role 
        return tag2role, b_tags, i_tags 

def evaluate_arg_f1(predictions, role_mapping):
    

    tag2role, b_tags, i_tags  = get_tag_mapping(role_mapping)
    gold_cnt = 0
    arg_idn_cnt = 0
    arg_cls_cnt = 0
    pred_cnt = 0

    for ex in predictions.values():
        gold_mentions = set()
        for event_dict in ex['event_mentions']:
            offset = 0
            if 'sent_lens' in ex: # need to consider offset 
                sent_idx = event_dict['trigger']['sent_idx']
                offset = sum([ex['sent_lens'][i] for i in range(sent_idx)])
                # print(offset)
            
            for arg in event_dict['arguments']:
                role = arg['role']
                ent_id = arg['entity_id']
                span = find_ent_span(ex, ent_id, offset)
                gold_mentions.add((span[0], span[1], role))
            gold_cnt += len(gold_mentions)

        # pred_mentions = get_pred_arg_mentions(ex, tag2role, b_tags, i_tags)
        pred_mentions = get_pred_arg_mentions_io(ex, tag2role)
        pred_cnt += len(pred_mentions)

        for tup in pred_mentions:
            start, end, role = tup 
            gold_idn = {item for item in gold_mentions if item[0]==start and item[1]==end}
            
            if gold_idn:
                arg_idn_cnt +=1 
                gold_cls = {item for item in gold_idn if item[2] == role}
                if gold_cls:
                    arg_cls_cnt+=1 
            

    arg_id_prec, arg_id_rec, arg_id_f = compute_f1(
        pred_cnt, gold_cnt, arg_idn_cnt)
    arg_prec, arg_rec, arg_f = compute_f1(
        pred_cnt, gold_cnt, arg_cls_cnt)

    
    print('Argument identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        arg_id_prec * 100.0, arg_id_rec * 100.0, arg_id_f * 100.0))
    print('Argument: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        arg_prec * 100.0, arg_rec * 100.0, arg_f * 100.0))

    return arg_id_f, arg_f




        

def evaluate_trigger_f1(predictions, ontology_dict):

    def get_tag_mapping(ontology_dict):
        tag2event_type = {}
        for et in ontology_dict:
            tag = ontology_dict[et]['i-label']
            tag2event_type[tag] = et

        return tag2event_type

    tag2event_type = get_tag_mapping(ontology_dict)
    gold_cnt = 0
    trigger_idn_cnt = 0
    trigger_cls_cnt = 0
    pred_cnt = 0

    for ex in predictions.values():
        gold_mentions = set()
        for event_dict in ex['event_mentions']:
            gold_mentions.add((event_dict['trigger']['start'], event_dict['trigger']['end'], event_dict['event_type']))
        gold_cnt += len(gold_mentions)

        pred_mentions = get_pred_tgr_mentions(ex,tag2event_type)

        pred_cnt += len(pred_mentions)

        for tup in pred_mentions:
            start, end, evt_type = tup 
            gold_idn = {item for item in gold_mentions if item[0]==start and item[1]==end}
            
            if gold_idn:
                trigger_idn_cnt +=1 
                gold_cls = {item for item in gold_idn if item[2] == evt_type}
                if gold_cls:
                    trigger_cls_cnt+=1 
            

    tgr_id_prec, tgr_id_rec, tgr_id_f = compute_f1(
        pred_cnt, gold_cnt, trigger_idn_cnt)
    tgr_prec, tgr_rec, tgr_f = compute_f1(
        pred_cnt, gold_cnt, trigger_cls_cnt)

    
    print('Trigger identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        tgr_id_prec * 100.0, tgr_id_rec * 100.0, tgr_id_f * 100.0))
    print('Trigger: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        tgr_prec * 100.0, tgr_rec * 100.0, tgr_f * 100.0))

    return tgr_id_f, tgr_f


