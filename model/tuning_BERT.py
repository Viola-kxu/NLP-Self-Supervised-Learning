import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.utils import PaddingStrategy

import json
from dataclasses import dataclass
import random
from typing import Union, Optional
import argparse
import torch
import evaluate

accuracy = evaluate.load("accuracy")


def get_dataset(mode='train'):
    dataset = []
    with (open(args.train_path if mode == 'train' else args.val_path, 'r', encoding="utf-8") as f):
        for line in f:
            dataset.append(json.loads(line))
    random.shuffle(dataset)
    return dataset


def preprocess_function(dataset):
    questions = [[instance['question']] * 4 for instance in dataset]
    options = [[f"[OPTION] {instance['options'][_]}" for _ in range(4)] for instance in dataset]

    questions = sum(questions, [])
    options = sum(options, [])

    tokenized_examples = tokenizer(questions, options, truncation=True)
    return {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


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
        print("__call__ called ---------------\n")
        label_name = "label" if "label" in features[0].keys() else "correct_index"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
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
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01
    )

    print(preprocess_function(get_dataset('train')).keys())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocess_function(get_dataset('train')),
        eval_dataset=preprocess_function(get_dataset('val')),
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()


def evaluation():
    dataset = get_dataset('val')
    correct = 0

    for instance in dataset:
        inputs = tokenizer([[instance['question'], instance['options'][i]] for i in range(4)], return_tensors="pt", padding=True)
        labels = torch.tensor(0).unsqueeze(0)
        outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
        logits = outputs.logits

        predicted_class = logits.argmax().item()
        if predicted_class == instance['correct_index'][0]:
            correct += 1

    print(f"Accuracy: {correct / len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../data/finetune_data_GPT.jsonl")
    parser.add_argument("--val_path", type=str, default="../data/archive/sat_math_validation.jsonl")
    parser.add_argument("--model_save_path", type=str, default="model_sat_math")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="google-bert/bert-base-uncased")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMultipleChoice.from_pretrained(args.model)

    train()
    evaluation()
