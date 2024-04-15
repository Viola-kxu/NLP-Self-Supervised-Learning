import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import argparse
import json
import random
from tqdm import tqdm


class SATMathDataset(Dataset):
    """
    Dataset class for SAT Math questions with multiple choice answers.
    """

    def __init__(self, file_path, tokenizer, max_len=512, split='train'):
        with open(file_path, 'r') as file:
            data = json.load(file)
        random.shuffle(data)
        split_index = int(0.5 * len(data))

        if split == 'train':
            data = data[:split_index]
        elif split == 'validation':
            data = data[split_index:]

        self.questions = [item['question'] for item in data]
        self.answers = [item['correct_index'][0] for item in data]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        answer = self.answers[index]
        encoded_data = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded_data['input_ids'].squeeze(0),
            'attention_mask': encoded_data['attention_mask'].squeeze(0),
            'labels': torch.tensor(answer, dtype=torch.long)
        }


class CustomModelForHeadTuning(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        logits = self.classifier(pooled_output)
        return logits


def train(model, train_dataloader, optimizer, device, update: bool = True):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc='Training', leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if update:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

        total_loss += loss.item()
    return total_loss / len(train_dataloader)


def evaluate_model(model, dataloader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = SATMathDataset(args.file_path, tokenizer, split='train')
    validation_dataset = SATMathDataset(args.file_path, tokenizer, split='validation')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    model = CustomModelForHeadTuning(args.model, num_labels=4).to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch + 1}')
        avg_loss = train(model, train_dataloader, optimizer, device)
        print(f'Training Loss: {avg_loss:.4f}')
        val_loss = train(model, train_dataloader, optimizer, device, update=False)
        val_accuracy = evaluate_model(model, validation_dataloader, device)
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="../data/sat_math_seed.json")
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    main(args)
