import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.utils import PaddingStrategy

import json
from dataclasses import dataclass
from typing import Union, Optional
import argparse
import torch
import evaluate


NUM_OPTIONS = 5
accuracy = evaluate.load("accuracy")


def get_dataset(mode='train'):
    dataset = {'id': [], 'question': [], 'options': [], 'correct_index': []}
    with (open(args.train_path if mode == 'train' else args.val_path, 'r', encoding="utf-8") as f):
        for line in f:
            data = json.loads(line)
            for key in dataset.keys():
                if key != "correct_index":
                    dataset[key].append(data[key])
                else:
                    dataset[key].append(data[key][0])
    return Dataset.from_dict(dataset)


def preprocess_function(dataset):
    questions = [[instance for _ in range(NUM_OPTIONS)] for instance in dataset['question']]
    options = [
        [f"[OPTION {j}] {dataset['options'][i][j]}" for j in range(NUM_OPTIONS)] for i in range(len(questions))
    ]

    questions = sum(questions, [])
    options = sum(options, [])

    # tokenized_examples = tokenizer(questions, options, truncation=True)
    tokenized_examples = tokenizer(questions, options, truncation=True, padding=True, max_length=512, add_special_tokens=True)

    return {k: [v[i: i + NUM_OPTIONS] for i in range(0, len(v), NUM_OPTIONS)] for k, v in tokenized_examples.items()}


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "correct_index" if "correct_index" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(NUM_OPTIONS)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        ).to(device)

        batch = {k: v.view(batch_size, NUM_OPTIONS, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64).to(device)
        return batch


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def train():
    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        logging_steps=100,
        learning_rate=1e-3,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01
    )

    # train_dateset = get_dataset('train').map(preprocess_function, batched=True)
    # eval_dataset = get_dataset('val').map(preprocess_function, batched=True)

    train_dateset = get_dataset('train').map(preprocess_function, batched=True).remove_columns(["id", "question", "options"])
    eval_dataset = get_dataset('val').map(preprocess_function, batched=True).remove_columns(["id", "question", "options"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dateset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()


def evaluation():
    dataset = get_dataset('val')
    correct = 0

    for idx in range(len(dataset['question'])):
        inputs = tokenizer([[dataset[idx]['question'], dataset[idx]['options'][option_idx]] for option_idx in range(4)], return_tensors="pt", padding=True).to(device)
        labels = torch.tensor(0).unsqueeze(0).to(device)
        outputs = model(**{k: v.unsqueeze(0).to(device) for k, v in inputs.items()}, labels=labels)
        logits = outputs.logits

        predicted_class = logits.argmax().item()
        if predicted_class == dataset[idx]['correct_index'][0]:
            correct += 1

    print(f"Accuracy: {correct / len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../data/aqua_rat.jsonl")
    parser.add_argument("--val_path", type=str, default="../data/aqua_rat_val.jsonl")
    parser.add_argument("--model_save_path", type=str, default="model_aqua_rat_tiny")
    # parser.add_argument("--model", type=str, default="google-bert/bert-base-uncased")
    # parser.add_argument("--model", type=str, default="google/bert_uncased_L-4_H-512_A-8") # BERT-small
    parser.add_argument("--model", type=str, default="google/bert_uncased_L-2_H-128_A-2")  # BERT-tiny

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMultipleChoice.from_pretrained(args.model).to(device)

    train()
    evaluation()
