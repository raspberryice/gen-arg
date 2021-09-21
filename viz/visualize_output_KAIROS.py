import os 
import json 
import argparse 
from copy import deepcopy
import spacy 
from spacy import displacy 
import re 
from collections import defaultdict

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

def extract_args_from_template(predicted, template, ontology_dict, evt_type):
    # extract argument text 
    template_words = template.strip().split()
    predicted_words = predicted.strip().split()    
    predicted_args = defaultdict(list) # argname -> List of text 
    t_ptr= 0
    p_ptr= 0 

    while t_ptr < len(template_words) and p_ptr < len(predicted_words):
        if re.match(r'<(arg\d+)>', template_words[t_ptr]):
            m = re.match(r'<(arg\d+)>', template_words[t_ptr])
            arg_num = m.group(1)
            arg_name = ontology_dict[evt_type][arg_num]

            if predicted_words[p_ptr] == '<arg>':
                # missing argument
                p_ptr +=1 
                t_ptr +=1  
            else:
                arg_start = p_ptr 
                while (p_ptr < len(predicted_words)) and (predicted_words[p_ptr] != template_words[t_ptr+1]):
                    p_ptr+=1 
                arg_text = predicted_words[arg_start:p_ptr]
                predicted_args[arg_name].append(arg_text)
                t_ptr+=1 
                # aligned 
        else:
            t_ptr+=1 
            p_ptr+=1 
    
    return dict(predicted_args)

def find_arg_span(arg, context_words, trigger_start, trigger_end, head_only=False, doc=None):
    match = None 
    arg_len = len(arg)
    min_dis = len(context_words) # minimum distance to trigger 
    for i, w in enumerate(context_words):
        if context_words[i:i+arg_len] == arg:
            if i < trigger_start:
                dis = abs(trigger_start-i-arg_len)
            else:
                dis = abs(i-trigger_end)
            if dis< min_dis:
                match = (i, i+arg_len-1)
                min_dis = dis 
    
    if match and head_only:
        assert(doc!=None)
        match = find_head(match[0], match[1], doc)
    return match 

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--result-file',type=str, default='checkpoints/gen-KAIROS-pointer-pred/predictions.jsonl')
    parser.add_argument('--test-file', type=str, default='data/kairos/test.jsonl')
    parser.add_argument('--gold', action='store_true')
    args = parser.parse_args() 

    ontology_dict = load_ontology('KAIROS')

    render_dicts = [] 

    reader= open(args.result_file, 'r') 

    with open(args.test_file,'r') as f:
        for line in f:
            doc = json.loads(line)
            # use sent_id for ACE 
            context_words = doc['tokens']
            render_dict = {
                "text":' '.join(context_words),
                "ents": [],
                "title": '{}_gold'.format(doc['doc_id']) if args.gold else doc['doc_id'],

            }
            word2char = {} # word index to start, end char index (end is not inclusive)
            ptr =0  
            for idx, w in enumerate(context_words):
                word2char[idx] = (ptr, ptr+ len(w))
                ptr = word2char[idx][1] +1  
            
            links = [] # (start_word, end_word, label)
            for eidx, e in enumerate(doc['event_mentions']):
                predicted = json.loads(reader.readline())
                filled_template = predicted['predicted']
                evt_type = e['event_type']
                label = 'E{}-{}'.format(eidx, e['event_type']) 
                trigger_start= e['trigger']['start']
                trigger_end = e['trigger']['end'] -1 
                trigger_tup = (trigger_start, trigger_end, label)
                links.append(trigger_tup)
                if args.gold:
                    # use gold arguments 
                    for arg in e['arguments']:
                        label = 'E{}-{}'.format(eidx, arg['role']) 
                        ent_id = arg['entity_id']
                        # get entity span 
                        matched_ent = [entity for entity in doc['entity_mentions'] if entity['id'] == ent_id][0]
                        arg_start = matched_ent['start']
                        arg_end = matched_ent['end'] -1 
                        links.append((arg_start, arg_end, label))
                else: # use predicted arguments 
                    template = ontology_dict[evt_type]['template']
                    # extract argument text 
                    predicted_args = extract_args_from_template(filled_template,template, ontology_dict, evt_type)
                    # get trigger 
                    # extract argument span
                    for argname in predicted_args:
                        for argtext in predicted_args[argname]:
                            arg_span = find_arg_span(argtext, context_words, 
                                trigger_start, trigger_end, head_only=False, doc=None) 
                            if arg_span:# if None means hullucination
                                label = 'E{}-{}'.format(eidx, argname) 
                                links.append((arg_span[0], arg_span[1], label))
                            
            sorted_links = sorted(links, key=lambda x: x[0]) # sort by start idx 
                
            for tup in sorted_links:
                arg_start, arg_end,  arg_name = tup 
                label = arg_name 
                render_dict["ents"].append({
                    "start": word2char[arg_start][0],
                    "end": word2char[arg_end][1],
                    "label": label, 
                })
            render_dicts.append(render_dict)


        

    file_name = args.result_file.split('.')[0]
    if args.gold:
        file_name += '.gold'

    html = displacy.render(render_dicts, style="ent", manual=True, page=True)

    with open('{}.html'.format(file_name), 'w') as f:
        f.write(html)

