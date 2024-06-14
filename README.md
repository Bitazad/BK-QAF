# BK-QAF
Biomedical Knowledge-enhanced Question Answering Framework

Loading the BioASQ dataset:
First create an account in the BioASQ official website:
BioASQ 6B: http://participants-area.bioasq.org/Tasks/6b/trainingDataset/
BioASQ 7B: http://participants-area.bioasq.org/Tasks/7b/trainingDataset/
BioASQ 8B: http://participants-area.bioasq.org/Tasks/8b/trainingDataset/

Filtering the questions of type factoid:
```
python factoid_extraction.py [input_file_path] [output_file_path]
```
