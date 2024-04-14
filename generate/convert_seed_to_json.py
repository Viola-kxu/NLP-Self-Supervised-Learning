import json
from datasets import load_dataset


def convert_dataset_to_json(dataset_name: str, output_file: str):
    dataset = load_dataset(dataset_name)['test']
    data = []
    for i in range(dataset.num_rows):
        question_dict = {
            "question": dataset[i]['query'],
            "options": dataset[i]['choices'],
            "correct_index": dataset[i]['gold']
        }
        data.append(question_dict)
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    dataset_name = "dmayhem93/agieval-sat-math"
    output_file = '../data/sat_math_seed.json'
    convert_dataset_to_json(dataset_name, output_file)
