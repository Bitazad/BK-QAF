import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import json
import numpy as np
from unidecode import unidecode
import re
from copy import deepcopy
import warnings
from sklearn.metrics.pairwise import cosine_similarity

# Disable all warnings
warnings.filterwarnings("ignore")

def prepare_input(question, context, tokenizer, device):
    inputs = tokenizer(question, context, return_tensors="pt", padding='max_length', truncation=True, max_length=512, add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def normalize_text(text):
    text_with_english_equivalents = unidecode(text)
    cleaned_text = text_with_english_equivalents.replace('£', 'ps')
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', cleaned_text)
    return cleaned_text.lower()

def load_model_and_tokenizer(model_path, device):
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, from_tf=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, from_tf=True)
    return model, tokenizer

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def main(model_path, qa_data_path, concepts_file_path, output_file_path, cosine_similarity_threshold=0.95):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    data = load_data(qa_data_path)
    questions = data.get("questions", [])
    ea = [deepcopy([normalize_text(exa) for exa in question.get("exact_answer", [""])[0]]) for question in questions]

    concepts_data = load_data(concepts_file_path)

    all_ground_truth = []
    all_top_5_answers = []

    for i, example in enumerate(questions):
        question_text = example.get("body", "")
        context = ' '.join(snippet.get("text", "") for snippet in example.get("snippets", []))

        question_inputs = tokenizer.encode_plus(question_text, return_tensors="pt", max_length=512, add_special_tokens=True)
        question_outputs = model.base_model(**question_inputs.to(device), return_dict=True)
        question_embedding = question_outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

        similarities = []
        for concept_item in concepts_data[i]["top_related_concepts"]:
            concept_inputs = tokenizer.encode_plus(concept_item, return_tensors="pt", max_length=512, add_special_tokens=True)
            concept_outputs = model.base_model(**concept_inputs.to(device), return_dict=True)
            concept_embedding = concept_outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            concept_similarity = cosine_similarity(question_embedding, concept_embedding)[0][0]
            if concept_similarity >= cosine_similarity_threshold:
                similarities.append((concept_similarity, concept_item))

        similarities.sort(key=lambda x: x[0], reverse=True)
        selected_concepts = [concept_item for _, concept_item in similarities][:1]
        expanded_question = question_text + " " + " ".join(selected_concepts)
        inp = prepare_input(expanded_question, context, tokenizer, device)

        with torch.no_grad():
            outputs = model(**inp)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_probs = torch.nn.functional.softmax(start_logits, dim=-1).squeeze()
        end_probs = torch.nn.functional.softmax(end_logits, dim=-1).squeeze()

        top_start_probs, top_start_indices = torch.topk(start_probs, 5)
        top_end_probs, top_end_indices = torch.topk(end_probs, 5)

        top_5_answers = []
        for start_idx, start_prob in zip(top_start_indices, top_start_probs):
            for end_idx, end_prob in zip(top_end_indices, top_end_probs):
                if start_idx <= end_idx:
                    answer_span = inp["input_ids"][0][start_idx: end_idx + 1]
                    predicted_answer = normalize_text(tokenizer.decode(answer_span, skip_special_tokens=True))
                    top_5_answers.append((predicted_answer, start_prob.item() * end_prob.item()))

        top_5_answers = sorted(top_5_answers, key=lambda x: x[1], reverse=True)[:5]
        top_5_answers = [ans[0] for ans in top_5_answers]

        ground_truth = ea[i]
        all_ground_truth.append(ground_truth)
        all_top_5_answers.append(top_5_answers)

    mrr_sum = 0.0
    correct_predictions_lacc = 0
    correct_predictions_sacc = 0

    for true_answers, predicted_answers in zip(all_ground_truth, all_top_5_answers):
        rank = next((i + 1 for i, ans in enumerate(predicted_answers) if any(true_answer == ans for true_answer in true_answers)), 0)
        reciprocal_rank = 1.0 / rank if rank > 0 else 0.0
        mrr_sum += reciprocal_rank

        if any(predicted_answer in true_answers for predicted_answer in predicted_answers):
            correct_predictions_lacc += 1

        if predicted_answers[0] in true_answers:
            correct_predictions_sacc += 1

    mrr = mrr_sum / len(all_ground_truth)
    print(f"(MRR): {mrr}")

    lenient_accuracy = correct_predictions_lacc / len(all_ground_truth)
    print(f"LAcc: {lenient_accuracy}")

    strict_accuracy = correct_predictions_sacc / len(all_ground_truth)
    print(f"SAcc: {strict_accuracy}")

if __name__ == "__main__":
    model_path = "scite/roberta-base-squad2-nq-bioasq"
    qa_data_path = "/path/to/qa_data.json"
    concepts_file_path = "/path/to/concepts_data.json"
    output_file_path = "/path/to/output_results.json"

    # Set cosine_similarity_threshold based on the dataset
    if '7b' in qa_data_path or '8b' in qa_data_path:
        cosine_similarity_threshold = 0.9
    elif '6b' in qa_data_path:
        cosine_similarity_threshold = 0.8

    main(model_path, qa_data_path, concepts_file_path, output_file_path, cosine_similarity_threshold)




