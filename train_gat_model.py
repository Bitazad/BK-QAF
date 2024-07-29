import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.nn.pytorch import GATConv
from transformers import BertTokenizer, BertModel
from tqdm import trange
import random
import warnings
import argparse

# Disable all warnings
warnings.filterwarnings("ignore")

# GAT model definition
class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 100, num_heads=2, activation=F.elu)
        self.conv2 = GATConv(200, num_classes, num_heads=1, activation=None)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = x.view(x.size(0), -1)
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
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def load_data(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)["questions"]

def create_graph(structure):
    if structure == 's1':
        num_nodes = 31
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(list(range(1, num_nodes)), (num_nodes - 1) * [0])
        g.add_edges((num_nodes - 1) * [0], list(range(1, num_nodes)))
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

def train_gat_model(json_file_path, output_model_path, structure, num_epochs=3, lr=0.0005):
    data = load_data(json_file_path)
    
    model = GAT(num_features=768, num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    g, num_nodes = create_graph(structure)
    g = g.to(device)

    questions, pos_samples, neg_samples, questions_concepts = [], [], [], []
    
    for question_data in data:
        question = question_data['body']
        snippets = question_data.get('snippets', [])
        context_concepts = set(question_data.get('context_concepts', {}).keys())
        question_concepts = set(question_data.get('question_concepts', {}).keys())
        questions.append(question)
        questions_concepts.append(question_concepts)
        positive_samples = set()
        for snippet in snippets:
            if snippet['start_index'] != -1 and snippet['end_index'] != -1:
                snippet_text = snippet['text']
                found_concepts = find_concepts_in_text(context_concepts, snippet_text)
                positive_samples.update(found_concepts)
        pos_samples.append(positive_samples)
        negative_samples = set()
        for snippet in snippets:
            if snippet['start_index'] == -1 and snippet['end_index'] == -1:
                snippet_text = snippet['text']
                found_concepts = find_concepts_in_text(context_concepts, snippet_text)
                negative_samples.update(found_concepts)
        negative_samples = negative_samples - positive_samples
        neg_samples.append(negative_samples)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for i in trange(len(questions)):
            if structure == 's1':
                concepts = list(questions_concepts[i] | pos_samples[i])
                concepts.extend(list(neg_samples[i]))
                labels = len(list(questions_concepts[i] | pos_samples[i])) * [1]
                labels.extend(len(list(neg_samples[i])) * [0])
                if len(concepts) > 30:
                    concepts = concepts[:30]
                    labels = labels[:30]
                elif len(concepts) < 30:
                    concepts.extend((30 - len(concepts)) * [''])
                    labels.extend((30 - len(labels)) * [0])

                node_texts = [questions[i]] + concepts
                node_features = [get_bert_embedding(text) for text in node_texts]
                features = torch.stack(node_features).to(device)

                model.train()
                logits = model(g, features)
                labels.insert(0, 1)
                labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

            elif structure == 's2':
                num_q_concepts = 5
                num_s_concepts = 25
                q_concepts = list(questions_concepts[i])[:min(num_q_concepts, len(questions_concepts[i]))]
                q_labels = [1] * len(q_concepts)
                if len(q_concepts) < num_q_concepts:
                    missing_q_concepts = num_q_concepts - len(q_concepts)
                    q_concepts.extend([''] * missing_q_concepts)
                    q_labels.extend([0] * missing_q_concepts)

                s_concepts_before = list(set(map(lambda x: (x, 1), pos_samples[i])) | set(map(lambda x: (x, 0), neg_samples[i])))
                random.shuffle(s_concepts_before)
                s_concepts = [x[0] for x in s_concepts_before[:min(num_s_concepts, len(s_concepts_before))]]
                s_labels = [x[1] for x in s_concepts_before[:min(num_s_concepts, len(s_concepts_before))]]
                if len(s_concepts) < num_s_concepts:
                    missing_s_concepts = num_s_concepts - len(s_concepts)
                    s_concepts.extend([''] * missing_s_concepts)
                    s_labels.extend([0] * missing_s_concepts)

                node_texts = [questions[i]] + q_concepts + s_concepts
                node_features = [get_bert_embedding(text) for text in node_texts]
                features = torch.stack(node_features).to(device)

                model.train()
                logits = model(g, features)
                combined_labels = [1] + q_labels + s_labels
                labels_tensor = torch.tensor(combined_labels, dtype=torch.long).to(device)

            elif structure == 's3':
                num_categories = 10
                num_concepts_per_category = 3
                total_concepts = 200

                # Randomly select concepts and label them
                labeled_concepts = [(concept, 1) for concept in pos_samples[i]] + [(concept, 0) for concept in neg_samples[i]]
                random.shuffle(labeled_concepts)
                random_concepts = labeled_concepts[:total_concepts]

                # Create categories dictionary
                categories_dict = {}
                for concept, label in random_concepts:
                    if concept != '':
                        category = question_data['context_concepts_2'].get(concept, question_data['question_concepts_2'].get(concept, ''))
                        if category not in categories_dict:
                            categories_dict[category] = [(concept, label)]
                        else:
                            categories_dict[category].append((concept, label))

                # Rank and select top categories
                category_counts = {category: len(concepts) for category, concepts in categories_dict.items()}
                sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                top_10_categories = sorted_categories[:min(num_categories, len(sorted_categories))]

                if len(sorted_categories) < num_categories:
                    missing_sorted_categories = num_categories - len(sorted_categories)
                    top_10_categories.extend([('', 0)] * missing_sorted_categories)

                random.shuffle(top_10_categories)
                top_10_categories_dict = {}
                all_selected_concepts = []
                emp_cat_idx = 0
                for category, _ in top_10_categories:
                    if category != '':
                        concepts_in_category = categories_dict.get(category, [])
                        random.shuffle(concepts_in_category)
                        selected_concepts = concepts_in_category[:min(num_concepts_per_category, len(concepts_in_category))]
                        selected_concepts.extend([('', 0)] * (num_concepts_per_category - len(selected_concepts)))
                        top_10_categories_dict[category] = selected_concepts
                    else:
                        selected_concepts = [('', 0)] * num_concepts_per_category
                        top_10_categories_dict[emp_cat_idx] = selected_concepts
                        emp_cat_idx += 1
                    all_selected_concepts.extend(selected_concepts)

                node_texts = [questions[i]] + list(map(lambda x: x[0], top_10_categories)) + list(map(lambda x: x[0], all_selected_concepts))
                node_features = [get_bert_embedding(text) for text in node_texts]
                features = torch.stack(node_features).to(device)

                labels = [1]
                categories_labels = []
                for category, concepts in top_10_categories_dict.items():
                    count_positive = sum([1 if label == 1 else 0 for _, label in concepts])
                    if count_positive >= 2:
                        categories_labels.append(1)
                    else:
                        categories_labels.append(0)
                labels += categories_labels
                labels += [0 if concept == '' or concept[1] == 0 else 1 for concept in all_selected_concepts]
                labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

            loss = loss_func(logits, labels_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'sample #{i}, loss: {loss.item()}')

    torch.save(model.state_dict(), output_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GAT model for concept extraction.')
    parser.add_argument('--json_file_path', type=str, required=True, help='Path to the JSON file containing the data.')
    parser.add_argument('--output_model_path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--structure', type=str, required=True, choices=['s1', 's2', 's3'], help='GAT structure to use.')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for training.')
    args = parser.parse_args()

    train_gat_model(args.json_file_path, args.output_model_path, args.structure, num_epochs=args.num_epochs, lr=args.lr)
