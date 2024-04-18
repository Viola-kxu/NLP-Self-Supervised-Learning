import argparse
import re
import warnings

from utils import *
from generate_questions import SEED_SIZE


def generate_answers(instruction, seed):
    # read current questions
    num = 1
    instances = []
    with (open(seed, 'r', encoding="utf-8") as f):
        for line in f:
            instances.append(json.loads(line))
            num += 1

    # generate an answer for each question
    for i in range(SEED_SIZE, num - 1):
        prompt = instances[i]['question'] + "\nA: "
        # extract options from the question
        options = extract_options(instances[i]['question'], instances[i]['id'])
        # generate the correct index for the question
        response = call_openai_api(instruction + prompt)
        print("Answer generated for id = " + str(i + 1) + " -------------------")
        print(response)
        ans = extract_idx(response)
        generated = {"id": instances[i]['id'], "question": instances[i]['question'], "options": options, "correct_index": [ans]}
        instances[i] = generated

        # test - dump the generated answers to a file
        print("Options: ", options, ", Correct index: ", generated['correct_index'], sep="")
        dump_jsonl(generated, '../data/generated_answers_test.jsonl')

    return instances


def extract_idx(input_text):
    match = re.search(r"Correct Index:\s*(\d+)", input_text)
    if match:
        return int(match.group(1))
    else:
        return None


def extract_options(question_text, quesiton_id):
    options = []
    option_labels = ['(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)',
                     '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
    first_option_index = question_text.find('(A)')
    if first_option_index == -1:
        warnings.warn(f"Question {quesiton_id} does not contain '(A)'")
    while True:
        next_option_index = min([question_text.find(label, first_option_index + 1) for label in option_labels if
                                 question_text.find(label, first_option_index + 1) != -1], default=-1)
        if next_option_index == -1:
            options.append(question_text[first_option_index:].strip())
            break
        option = question_text[first_option_index:next_option_index].strip()
        options.append(option)
        first_option_index = next_option_index
    return options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='../data/generated_answers.jsonl')
    parser.add_argument('--save_path', type=str, default='../data/generated_answers.jsonl')
    parser.add_argument('--ins_file', type=str, default='instructions/instruction_answer.txt')
    args = parser.parse_args()
    file = args.file

    with open(args.ins_file, 'r', encoding="utf-8") as f:
        instruction = f.read()
    instances = generate_answers(instruction, file)

    # clear the file
    target_file = open(args.save_path, mode="w").close()

    # write the generated answers to the file
    for instance in instances:
        dump_jsonl(instance, args.save_path)