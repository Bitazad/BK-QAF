import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import trange
import argparse

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, contexts, labels):
        self.questions = questions
        self.contexts = contexts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenizer(self.questions[idx], self.contexts[idx], max_length=512, padding="max_length", truncation=True).items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def fine_tune_model(model_name, train_file, output_model_path, epochs=1, batch_size=8, lr=2e-5):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load training data
    with open(train_file, 'r') as json_file:
        data = json.load(json_file)["questions"]

    questions, contexts, labels = [], [], []
    for idx, item in enumerate(data):
        snippets = item["snippets"]
        questions.extend(len(snippets) * [item["body"]])
        exact_answer = item["exact_answer"][0]
        contexts.extend([snippet["text"] for snippet in snippets])
        labels.extend([1 if exact_answer in snippet["text"] else 0 for snippet in snippets])

    tokenized_inputs = tokenizer(questions, contexts, max_length=512, return_tensors="pt", padding="max_length", truncation=True)
    labels = torch.tensor(labels, dtype=torch.float)

    dataset = TensorDataset(tokenized_inputs.input_ids, tokenized_inputs.attention_mask, labels)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.4f}")

    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    print(f"Model saved to {output_model_path}")

def evaluate_model(model_path, test_file, output_file_path):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(test_file, 'r') as json_file:
        data = json.load(json_file)["questions"]

    output_data = []
    for question in data:
        question_text = question["body"]
        snippets = question["snippets"]
        texts = [snippet["text"] for snippet in snippets]

        features = tokenizer(len(texts) * [question_text], texts, return_tensors="pt", padding=True, truncation=True)
        features = {k: v.to(device) for k, v in features.items()}

        with torch.no_grad():
            scores = model(**features).logits.squeeze().tolist()

        if not isinstance(scores, list):
            scores = [scores]

        snippet_score_pairs = list(zip(snippets, scores))
        sorted_snippets = sorted(snippet_score_pairs, key=lambda x: x[1], reverse=True)
        sorted_snippets_data = [{"text": snippet["text"], "score": score} for snippet, score in sorted_snippets]

        output_data.append({"question": question_text, "sorted_snippets": sorted_snippets_data})

    with open(output_file_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)
    print(f"Processed data saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune and evaluate cross-encoder model.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Mode: train or test.')
    parser.add_argument('--model_name', type=str, default='cross-encoder/qnli-distilroberta-base', help='Model name or path.')
    parser.add_argument('--train_file', type=str, help='Path to the training data file.')
    parser.add_argument('--output_model_path', type=str, help='Path to save the fine-tuned model.')
    parser.add_argument('--test_file', type=str, help='Path to the test data file.')
    parser.add_argument('--output_file_path', type=str, help='Path to save the test results.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    args = parser.parse_args()

    if args.mode == 'train':
        fine_tune_model(args.model_name, args.train_file, args.output_model_path, args.epochs, args.batch_size, args.lr)
    elif args.mode == 'test':
        evaluate_model(args.output_model_path, args.test_file, args.output_file_path)
