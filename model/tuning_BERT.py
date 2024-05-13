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
# class DataCollatorForMultipleChoice:
#     """
#     Data collator that will dynamically pad the inputs for multiple choice received.
#     """
#
#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#
#     def __call__(self, features):
#         label_name = "correct_index" if "correct_index" in features[0].keys() else "labels"
#         labels = [feature.pop(label_name) for feature in features]
#         batch_size = len(features)
#         flattened_features = [
#             [{k: v[i] for k, v in feature.items()} for i in range(NUM_OPTIONS)] for feature in features
#         ]
#         flattened_features = sum(flattened_features, [])
#
#         batch = self.tokenizer.pad(
#             flattened_features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         ).to(device)
#
#         batch = {k: v.view(batch_size, NUM_OPTIONS, -1) for k, v in batch.items()}
#         batch["labels"] = torch.tensor(labels, dtype=torch.int64).to(device)
#         return batch

@dataclass
class DataCollatorForMultipleChoice:
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

        # Ensure tensors are on CPU and only move to device later
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # Do not move to device here; return CPU tensors
        batch = {k: v.view(batch_size, NUM_OPTIONS, -1) for k, v in batch.items()}
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
        remove_unused_columns=False,
        logging_steps=100,
        learning_rate=5e-5,
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

# def evaluation():
#     dataset = get_dataset('val')
#     correct = 0
#     for idx in range(len(dataset['question'])):
#         inputs = tokenizer(dataset[idx]['question'], dataset[idx]['options'], return_tensors="pt", padding=True)
#         labels = torch.tensor([dataset[idx]['correct_index']], dtype=torch.int64)
#         inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to device here to fix the bug
#         labels = labels.to(device)
#
#         outputs = model(**inputs, labels=labels)
#         logits = outputs.logits
#         predicted_class = logits.argmax(dim=1)
#         if predicted_class.item() == labels.item():
#             correct += 1
#
#     print(f"Accuracy: {correct / len(dataset)}")

def evaluation():
    dataset = get_dataset('val')
    correct = 0

    for item in dataset:
        question = item['question']
        options = item['options']
        correct_index = item['correct_index']  # Assuming correct_index is already an integer

        # Tokenize each option together with the question
        input_ids = []
        attention_masks = []
        for option in options:
            inputs = tokenizer(question, option, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
            input_ids.append(inputs['input_ids'][0])
            attention_masks.append(inputs['attention_mask'][0])

        # Convert list of tensors to a single tensor for each type
        input_ids = torch.stack(input_ids, dim=0).unsqueeze(0).to(device)  # Shape: (1, num_options, seq_length)
        attention_masks = torch.stack(attention_masks, dim=0).unsqueeze(0).to(device)  # Shape: (1, num_options, seq_length)

        # Create a dictionary for model inputs
        batch_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }

        # Forward pass
        outputs = model(**batch_inputs)
        logits = outputs.logits.squeeze(0)  # Remove the batch dimension, now shape should be (num_options, num_classes)
        predicted_class = logits.argmax(dim=0).item()  # Get the index of the highest logit score

        # Check if the prediction matches the correct index
        if predicted_class == correct_index:
            correct += 1

    accuracy = correct / len(dataset)
    print(f"Accuracy: {accuracy}")
    return accuracy





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--train_path", type=str, default="../data/sat_math/generated_sat_math_dataset.jsonl")
    parser.add_argument("--val_path", type=str, default="../data/aqua_rat_val.jsonl")
    parser.add_argument("--model_save_path", type=str, default="model_aqua_rat")
    #parser.add_argument("--model", type=str, default="google-bert/bert-base-uncased")
    #parser.add_argument("--model", type=str, default="google/bert_uncased_L-4_H-512_A-8") # BERT-small
    parser.add_argument("--model", type=str, default="../model/model_sat_math/checkpoint-150") #Bert-finetuned
    # parser.add_argument("--model", type=str, default="google/bert_uncased_L-2_H-128_A-2")  # BERT-tiny

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMultipleChoice.from_pretrained(args.model).to(device)

    #train()
    evaluation()