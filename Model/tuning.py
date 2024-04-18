import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from transformers import T5Tokenizer, T5Model
import argparse
import json
import random
from tqdm import tqdm


class SATMathDataset(Dataset):
    """
    Dataset class for SAT Math questions with multiple choice answers.
    """

    def __init__(self, tokenizer, max_len=512, split='train'):
        data = []
        with (open(args.train_path if split == 'train' else args.val_path, 'r', encoding="utf-8") as f):
            for line in f:
                data.append(json.loads(line))

        random.shuffle(data)

        self.questions = [item['question'] for item in data]
        self.answers = [item['options'][item['correct_index'][0]] for item in data]
        self.correct_index = [item['correct_index'][0] for item in data]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        answer = self.answers[index]
        encoded_question = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoded_answer = self.tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded_question['input_ids'].squeeze(0),
            'attention_mask': encoded_question['attention_mask'].squeeze(0),
            'labels': encoded_answer['input_ids'].squeeze(0),
            'correct_index': self.correct_index[index]
        }


class CustomModelForHeadTuning(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state = outputs.last_hidden_state
        # pooled_output = torch.mean(last_hidden_state, dim=1)
        # logits = self.classifier(pooled_output)
        print("Output", outputs[0].size())
        return outputs


def train(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc='Training', leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_dataloader)


def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            correct_index = batch['correct_index'].to(device)

            # outputs = model(input_ids, attention_mask)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            print(preds)

            correct += accurate_count(preds, correct_index)
            total += len(preds)
    return correct / total


def accurate_count(preds, correct_index):
    count = 0
    for i in range(len(preds)):
        if preds[i].find("(" + chr(ord('A') + correct_index[i]) + ")") != -1:
            count += 1
    return count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    train_dataset = SATMathDataset(tokenizer, split='train')
    validation_dataset = SATMathDataset(tokenizer, split='validation')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    # model = CustomModelForHeadTuning(args.model, num_labels=4).to(device)
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    # optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(params=model.parameters())

    val_accuracy_best = -1
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch + 1}')
        avg_loss = train(model, train_dataloader, optimizer, device)
        print(f'Training Loss: {avg_loss:.4f}')
        val_accuracy = evaluate_model(model, validation_dataloader, tokenizer, device)
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        if val_accuracy > val_accuracy_best:
            val_accuracy_best = val_accuracy
            model.save_pretrained("model_sat_math", from_pt=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../data/archive/sat_math_seed_train.jsonl")
    parser.add_argument("--val_path", type=str, default="../data/archive/sat_math_validation.jsonl")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--model", type=str, default="t5-base")

    args = parser.parse_args()

    main()

