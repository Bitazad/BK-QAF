# BK-QAF
**Biomedical Knowledge-enhanced Question Answering Framework (BK-QAF)**

This framework retrieves, ranks, and employs domain-specific concepts from the Unified Medical Language System (UMLS) to enhance the comprehension and reasoning capabilities of language models. By using Graph Attention Networks (GATs) to analyze the interconnections and relationships between terms and entities in biomedical texts, we identify the most relevant concepts for the query. The framework then ranks these UMLS concepts by relevance, expands the questions with the top-ranked concepts, and processes them using a fine-tuned language model.

## Framework

Here is the framework's figure:

<div align="center">
  <img src="https://github.com/Bitazad/BK-QAF/blob/main/MainFigure.png" alt="Framework Figure" width="700"/>
</div>

# Code and Data

**Below is the code necessary to produce the results. Follow the instructions provided to run the code:**


1. Loading the BioASQ dataset:<br>
First create an account in [the BioASQ official website](http://participants-area.bioasq.org/):<br>
BioASQ 6B: http://participants-area.bioasq.org/Tasks/6b/trainingDataset/<br> 
BioASQ 7B: http://participants-area.bioasq.org/Tasks/7b/trainingDataset/<br>
BioASQ 8B: http://participants-area.bioasq.org/Tasks/8b/trainingDataset/<br>


2. Filtering the questions of type factoid:
```
python factoid_extraction.py [input_file_path] [output_file_path]
```


For downloading the updated training and testing files containing the concepts and categories for the samples:

  - [Training sets](https://github.com/Bitazad/BK-QAF/blob/main/Trainingsets.zip)
  - [Testing sets](https://github.com/Bitazad/BK-QAF/blob/main/Testsets.zip)


**Requirements**

Install the required packages using pip:

```bash
pip install -r requirements.txt
```
3. GAT Training
```
python train_gat_model.py [json_file_path] [output_file_path] [structure] [num_epochs] [lr]
```

4. GAT Testing

```
python test_gat_model.py [json_file_path] [model_path] [output_file_path] [structure]
```

5. Cross-encoder Fine-tuning
```
python fine_tune_cross_encoder.py [train_file_path] [output_model_path] [num_epochs] [batch_size] [learning_rate]
```

**After extracting and storing the top concepts in the `concepts_file_path`, we can use the sequence tagging (span extraction) model and the dataset to predict the span with respct to the top two snippets ranked by the cross-encoder model.**

6. Span Prediction
```
python Span_prediction.py [model_path] [qa_data_path] [concepts_file_path] [output_file_path]
```
