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


## Datasets
- RAMS (Download at [https://nlp.jhu.edu/rams/])
- ACE05 (Access from LDC[https://catalog.ldc.upenn.edu/LDC2006T06] and preprocessing following OneIE[http://blender.cs.illinois.edu/software/oneie/])
- WikiEvents (Available here [s3://gen-arg-data/wikievents/])
