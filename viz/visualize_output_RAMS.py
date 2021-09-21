import os 
import json 
import argparse 
from copy import deepcopy
import spacy 
from spacy import displacy 
import re 



if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--result-file',type=str, default='gen-trigger-pred-output.jsonl')
    parser.add_argument('--gold', action='store_true')
    args = parser.parse_args() 

    render_dicts = [] 

    

    with open(args.result_file, 'r') as f:
        for line in f:
            ex = json.loads(line.strip())
            title = ex['doc_key']
            context_words = [w for sent in ex['sentences'] for w in sent ]
            
            
            render_dict = {
                "text":' '.join(context_words),
                "ents": [],
                "title": '{}_gold'.format(ex['doc_key']) if args.gold else ex['doc_key'],
            }
            
            word2char = {} # word index to start, end char index (end is not inclusive)
            ptr =0  
            for idx, w in enumerate(context_words):
                word2char[idx] = (ptr, ptr+ len(w))
                ptr = word2char[idx][1] +1  

            if args.gold:
                links = ex['ref_evt_links']
            else:
                links = ex['gold_evt_links']

            tmp = ex['evt_triggers'][0]
            trigger_start = tmp[0]
            trigger_end = tmp[1]
            trigger_type = tmp[2][0][0]

            links.append([(trigger_start, trigger_end), (trigger_start, trigger_end), trigger_type])

            sorted_links = sorted(links, key=lambda x: x[1][0])
                
            for tup in sorted_links:
                trigger_span, arg_span, arg_name = tup 
                m = re.match(r'evt\d+arg\d+(\w+)', arg_name)
                if m:
                    label = m.group(1)
                else:
                    label = arg_name
                render_dict["ents"].append({
                    "start": word2char[arg_span[0]][0],
                    "end": word2char[arg_span[1]][1],
                    "label": label, 
                })

            
            
            render_dicts.append(render_dict)
            

# ex = [{"text": "But Google is starting from behind.",
#        "ents": [{"start": 4, "end": 10, "label": "ORG"}],
#        "title": "doc1"}, 
#        {"text": "But Google is starting from behind.",
#        "ents": [{"start": 4, "end": 10, "label": "ORG"}],
#        "title": "doc2"}, 
       
#        ]

    file_name = args.result_file.split('.')[0]
    if args.gold:
        file_name += '.gold'

    html = displacy.render(render_dicts, style="ent", manual=True, page=True)

    with open('{}.html'.format(file_name), 'w') as f:
        f.write(html)

