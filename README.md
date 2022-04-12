# Argument Extraction by Generation

Code for paper "Document-Level Argument Extraction by Conditional Generation". NAACL 21'


## Dependencies 
- pytorch=1.6 
- transformers=3.1.0
- pytorch-lightning=1.0.6
- spacy=3.0 # conflicts with transformers
- pytorch-struct=0.4 


## Model Checkpoints 
Checkpoints trained from this repo are shared for the WikiEvents dataset and the ACE dataset are available at: [s3://gen-arg-data/checkpoints/].

You can download all the contents from the S3 bucket using AWS cli: `aws s3 cp s3://gen-arg-data/checkpoints/ ./ --recursive` 



## Datasets
- RAMS (Download at [https://nlp.jhu.edu/rams/])
- ACE05 (Access from LDC[https://catalog.ldc.upenn.edu/LDC2006T06] and preprocessing following OneIE[http://blender.cs.illinois.edu/software/oneie/])
- WikiEvents (Available here [s3://gen-arg-data/wikievents/])

You can download the data through the AWS cli or AWS console. 
Alternatively, you can download individual files by 
- `wget https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/<split>.jsonl` for split={train, dev,test}.
- `wget https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/coref/<split>.jsonlines` for split={train, dev, test}.
  
Additional processed test files for RAMS can be downloaded by
- `wget https://gen-arg-data.s3.us-east-2.amazonaws.com/RAMS/test_head_coref.jsonlines`
- `wget https://gen-arg-data.s3.us-east-2.amazonaws.com/RAMS/test_head.jsonlines`
