'''
This file takes a prediction from the tagger model and converts it into the standard format for argument extraction.
'''
from json import load
import os 
import json 
import argparse 
from utils import load_ontology, get_pred_tgr_mentions

def get_tag_mapping(ontology_dict):
        tag2event_type = {}
        for et in ontology_dict:
            tag = ontology_dict[et]['i-label']
            tag2event_type[tag] = et

        return tag2event_type


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--pred-file',type=str)
    parser.add_argument('--ref-file', type=str)
    parser.add_argument('--output-file',type=str)
    parser.add_argument('--dataset', type=str, default='ACE')
    args = parser.parse_args() 


    
    ontology_dict = load_ontology(args.dataset)
    tag2event_type = get_tag_mapping(ontology_dict)



    with open(args.pred_file) as f:
        predictions = json.load(f)

    gold_exs = []
    with open(args.ref_file) as f:
        for line in f:
            ex = json.loads(line)
            gold_exs.append(ex)
        
    
    writer = open(args.output_file,'w')
    total_pred = 0
    total_gold = 0 
    for ex in gold_exs:
        doc_key = ex['sent_id']
        pred_tags = predictions[doc_key]['pred_tags']
        ex['pred_tags'] = pred_tags
        pred_exs = [] 
        if sum(pred_tags) > 0:
            pred_triggers = get_pred_tgr_mentions(ex, tag2event_type)
            # List of (cur_start, cur_end, tag2event_type[prev_tag])
            for tgr_tup in pred_triggers:
                start, end, evt_type = tgr_tup
                trigger_text = ' '.join(ex['tokens'][start:end]) 
                pred_exs.append({
                    'event_type': evt_type,
                    'arguments':[],
                    'trigger': {
                        'start': start, 
                        'end': end,
                        'text': trigger_text,
                         
                    }
                })
                total_pred +=1 
        total_gold += len(ex['event_mentions']) 
        ex['event_mentions'] = pred_exs
        writer.write(json.dumps(ex) + '\n')

    print(total_pred)
    print(total_gold)
    writer.close() 





