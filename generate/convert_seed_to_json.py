import json
from datasets import load_dataset


def convert_dataset_to_json(dataset_name: str, output_file: str):
    dataset = load_dataset(dataset_name)['test']
    data = []
    for i in range(dataset.num_rows):
        correct_index = dataset[i]['correct']  # process index
        if (correct_index >= 'A' and correct_index <= 'Z'):
            correct_index = ord(correct_index) - ord('A')
        question_dict = {
            "id": i,
            "question": dataset[i]['question'],
            "options": dataset[i]['options'],
            "correct_index": [correct_index]
        }
        data.append(question_dict)
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    dataset_name = "aqua_rat"
    output_file = '../data/aqua_rat.jsonl'
    convert_dataset_to_json(dataset_name, output_file)


