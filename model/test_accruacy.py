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
    path1 = "../data/finetuned_sat_math_with_answers2.jsonl"
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


if __name__ == "__main__":
    test_accuracy()
