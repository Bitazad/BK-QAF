import json
import torch
import dgl
from dgl.nn.pytorch import GATConv
from transformers import BertTokenizer, BertModel
from tqdm import trange
import random
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# GAT model definition
class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 41, num_heads=1, activation=torch.nn.functional.elu)
        self.conv2 = GATConv(41, num_classes, num_heads=1, activation=None)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = x.view(-1, 41)  # Reshape the output for the next layer
        x = self.conv2(g, x)
        return x.mean(1)

# Utility functions
def find_concepts_in_text(concepts, text):
    found_concepts = set()
    for concept in concepts:
        if concept.lower() in text.lower():
            found_concepts.add(concept)
    return found_concepts

def get_bert_embedding(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=768)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def compute_cosine_similarity(v1, v2):
    return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

def create_graph(structure):
    if structure == 's1':
        num_nodes = 31
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(list(range(1, num_nodes)), (num_nodes - 1) * [0])
        g = dgl.add_self_loop(g)
        return g, num_nodes
    
    elif structure == 's2':
        g = dgl.DGLGraph()
        num_q_concepts = 5
        num_s_concepts = 25
        num_nodes = 1 + num_q_concepts + num_s_concepts
        g.add_nodes(num_nodes)
        g.add_edges(list(range(1, num_q_concepts + 1)), num_q_concepts * [0])
        g.add_edges(num_s_concepts * [0], list(range(num_q_concepts + 1, num_q_concepts + num_s_concepts + 1)))
        g = dgl.add_self_loop(g)
        return g, num_nodes

    elif structure == 's3':
        g = dgl.DGLGraph()
        num_categories = 10
        num_concepts_per_category = 3
        num_nodes = num_categories + num_categories * num_concepts_per_category + 1
        total_concepts = 200
        g.add_nodes(num_nodes)
        
        # Add edges for categories
        concept_iter = num_categories + 1
        for category_id in range(1, num_categories + 1):
            g.add_edge(0, category_id)
            g.add_edge(category_id, 0)
            g.add_edges(num_concepts_per_category * [category_id], range(concept_iter, concept_iter + num_concepts_per_category))
            g.add_edges(range(concept_iter, concept_iter + num_concepts_per_category), num_concepts_per_category * [category_id])
            concept_iter += num_concepts_per_category

        g = dgl.add_self_loop(g)
        return g, num_nodes

def test_gat_model(json_file_path, model_path, output_file_path, structure):
    # Load the pre-trained GAT model
    model = GAT(num_features=768, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load test data
    with open(json_file_path, 'r') as json_file:
        test_data = json.load(json_file)["questions"]

    # Create graph based on structure
    g, num_nodes = create_graph(structure)
    
    predicted_concepts = []

    for i in trange(len(test_data)):
        test_question_data = test_data[i]
        test_question = test_question_data['body']
        test_snippets = test_question_data.get('snippets', [])
        test_context_concepts = set(test_question_data.get('context_concepts', {}).keys())
        test_question_concepts = set(test_question_data.get('question_concepts', {}).keys())

        if structure == 's1' or structure == 's2':
            positive_samples = set(test_question_concepts)
            for snippet in test_snippets:
                if snippet['start_index'] != -1 and snippet['end_index'] != -1:
                    snippet_text = snippet['text']
                    found_concepts = find_concepts_in_text(test_context_concepts, snippet_text)
                    positive_samples.update(found_concepts)
            negative_samples = set()
            for snippet in test_snippets:
                if snippet['start_index'] == -1 and snippet['end_index'] == -1:
                    snippet_text = snippet['text']
                    found_concepts = find_concepts_in_text(test_context_concepts, snippet_text)
                    negative_samples.update(found_concepts)

            if not positive_samples:
                positive_samples.update(test_question_concepts)

            negative_samples = negative_samples - positive_samples
            all_samples = list(positive_samples | negative_samples)
            concepts = all_samples[:30] if len(all_samples) >= 30 else all_samples + (30 - len(all_samples)) * ['']
            node_texts = [test_question] + concepts
            node_features = [get_bert_embedding(text) for text in node_texts]
            features = torch.stack(node_features)

            with torch.no_grad():
                logits = model(g, features)
                max_indices = torch.argsort(logits[1:, 1], descending=True)
                top_related_concepts = [node_texts[idx + 1] for idx in max_indices]

        elif structure == 's3':
            all_samples = list(test_context_concepts | test_question_concepts)
            similarity_scores = []
            for concept in all_samples:
                similarity_score = compute_cosine_similarity(get_bert_embedding(test_question), get_bert_embedding(concept))
                similarity_scores.append((concept, similarity_score))
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            top_concepts = [concept for concept, _ in similarity_scores[:200]]
            concepts = top_concepts if len(top_concepts) >= 200 else top_concepts + [''] * (200 - len(top_concepts))
            
            categories_dict = {}
            for concept in concepts:
                if concept != '':
                    category = test_question_data['context_concepts_2'].get(concept, test_question_data['question_concepts_2'].get(concept, ''))
                    if category not in categories_dict:
                        categories_dict[category] = [concept]
                    else:
                        categories_dict[category].append(concept)
            category_counts = {category: len(concepts) for category, concepts in categories_dict.items()}
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            top_10_categories = sorted_categories[:min(10, len(sorted_categories))]

            if len(sorted_categories) < 10:
                missing_sorted_categories = 10 - len(sorted_categories)
                top_10_categories.extend([('', 0)] * missing_sorted_categories)
            random.shuffle(top_10_categories)
            top_10_categories_dict = {}
            all_selected_concepts = []
            emp_cat_idx = 0
            for category, _ in top_10_categories:
                if category != '':
                    concepts_in_category = categories_dict.get(category, [])
                    random.shuffle(concepts_in_category)
                    selected_concepts = concepts_in_category[:min(3, len(concepts_in_category))]
                    selected_concepts.extend([''] * (3 - len(selected_concepts)))
                    top_10_categories_dict[category] = selected_concepts
                else:
                    selected_concepts = [''] * 3
                    top_10_categories_dict[emp_cat_idx] = selected_concepts
                    emp_cat_idx += 1
                all_selected_concepts.extend(selected_concepts)

            node_texts = [test_question] + list(map(lambda x: x[0], top_10_categories)) + all_selected_concepts
            node_features = [get_bert_embedding(text) for text in node_texts]
            features = torch.stack(node_features)

            with torch.no_grad():
                logits = model(g, features)
                max_indices = torch.argsort(logits[1:, 1], descending=True)
                top_related_concepts = []
                for idx in max_indices:
                    concept = node_texts[idx + 1]
                    if concept in all_selected_concepts:
                        top_related_concepts.append(concept)

        predicted_concepts.append({
            "question": test_question,
            "top_related_concepts": top_related_concepts
        })

    with open(output_file_path, 'w') as output_file:
        json.dump(predicted_concepts, output_file, indent=2)

    print(f"Results saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GAT model for concept extraction.')
    parser.add_argument('--json_file_path', type=str, required=True, help='Path to the JSON file containing the test data.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained GAT model.')
    parser.add_argument('--output_file_path', type=str, required=True, help='Path to save the predicted concepts.')
    parser.add_argument('--structure', type=str, required=True, choices=['s1', 's2', 's3'], help='GAT structure to use.')
    args = parser.parse_args()

    test_gat_model(args.json_file_path, args.model_path, args.output_file_path, args.structure)
