import json


def read_correct_indices(filepath):
    """
    Read the correct indices from a given JSONL file.
    """
    correct_indices = []
    try:
        with open(filepath, 'r', encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                correct_indices.append((data['id'], data['correct_index'][0]))
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
        raise
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in the file {filepath}.")
        raise
    return correct_indices


def test_accuracy():
    """
    Test the accuracy of answers by comparing two files.
    """
    path1 = "../data/base_sat_math_with_answers.jsonl"
    path2 = "../data/archive/sat_math_validation.jsonl"

    # Read correct indices from both files
    predicted_answers = read_correct_indices(path1)
    actual_answers = read_correct_indices(path2)

    correct_count = 0
    correct_ids = []

    # Assuming both files have the same length and corresponding lines
    for predicted, actual in zip(predicted_answers, actual_answers):
        if predicted[1] == actual[1]:
            correct_count += 1
            correct_ids.append(actual[0])  # Save the ID of correctly answered questions

    accuracy = correct_count / len(actual_answers) if actual_answers else 0
    print("Accuracy:", accuracy)
    print("Correct IDs:", correct_ids)

def test_gsm8k_accuracy():
    predicted_answers = []
    actual_answers = []
    with (open(path1, 'r', encoding="utf-8") as f):
        for line in f:
            generated_answer = json.loads(line)['Correct Answer']
            predicted_answers.append(generated_answer)
    with (open(path2, 'r', encoding="utf-8") as f):
        for line in f:
            actual_answer = extract_actual_answer(json.loads(line)['answer'])
            actual_answers.append(actual_answer)
    print("actual answers", actual_answers)
    print("predicted answers", predicted_answers)
    correct_count = 0
    for predicted, actual in zip(predicted_answers, actual_answers):
        if predicted == actual:
            correct_count += 1
    print("Accuracy: ", correct_count / len(actual_answers))

def extract_actual_answer(answer):
    start = answer.index("####") + 4
    return answer[start:].strip()



if __name__ == "__main__":
    path1 = "../data/base_gsm8k_answer.jsonl"
    path2 = "../data/gsm8k_test.jsonl"

    # test_accuracy()
    test_gsm8k_accuracy()


