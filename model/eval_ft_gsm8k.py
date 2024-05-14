import argparse
import re
import warnings

from ft_utils import *


def generate_answers(instruction, file):
    # read current questions
    num = 1
    instances = []
    with (open(file, 'r', encoding="utf-8") as f):
        for line in f:
            instances.append(json.loads(line))
            num += 1

    for i in range(0, num - 1):
        prompt = str(instances[i]['question']) + "\n#Your response#: "
        response = call_openai_api(instruction + prompt)
        print("Answer generated for id = " + str(i + 1) + " -------------------")
        print(response)
        explanation = extract_explanation(response)
        answer = extract_answer(response)
        generated = {"question": instances[i]['question'],
                     "Explanation": explanation, "Correct Answer": answer}
        instances[i] = generated

        # test - dump the generated answers to a file
        print("question", generated['question'], "Correct Answer: ", generated['Correct Answer'], sep="")
        dump_jsonl(generated, args.save_path)

    return instances


def extract_explanation(response):
    try:
        # Splitting the response to find the start of the explanation
        start = response.index("Explanation:") + len("Explanation:")
        end = response.index("Correct Answer:")

        # Extracting the explanation text
        explanation = response[start:end].strip()
        return explanation
    except ValueError:
        return "Explanation not found or incorrectly formatted response."


def extract_answer(response):
    try:
        start = response.index("Correct Answer:") + len("Correct Answer:")

        answer = response[start:].strip()
        return answer
    except ValueError:
        return "Answer not found or incorrectly formatted response."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='../data/gsm8k_test.jsonl')
    parser.add_argument('--save_path', type=str, default='../data/gsm8k_trained_model_evaluation.jsonl')
    parser.add_argument('--ins_file', type=str, default='../generate/instructions/instruction_gsm8k.txt')
    args = parser.parse_args()

    with open(args.ins_file, 'r', encoding="utf-8") as f:
        instruction = f.read()
    instances = generate_answers(instruction, args.file)

    # write the generated answers to the file
    target_file = open(args.save_path, mode="w").close()  # clear the file
    for instance in instances:
        dump_jsonl(instance, args.save_path)

