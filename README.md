# BK-QAF
Biomedical Knowledge-enhanced Question Answering Framework

Loading the BioASQ dataset:
First create an account in the BioASQ official website:<br>
BioASQ 6B: http://participants-area.bioasq.org/Tasks/6b/trainingDataset/<br>
BioASQ 7B: http://participants-area.bioasq.org/Tasks/7b/trainingDataset/<br>
BioASQ 8B: http://participants-area.bioasq.org/Tasks/8b/trainingDataset/<br>


Filtering the questions of type factoid:
```
python factoid_extraction.py [input_file_path] [output_file_path]
```
