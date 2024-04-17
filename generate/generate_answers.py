import argparse
import re
import warnings

from utils import *
from generate_questions import SEED_SIZE


def generate_answers(args, instruction, seed):
    prompt = instruction

    # read current questions
    num = 1
    instances = []
    with (open(seed, 'r', encoding="utf-8") as f):
        for line in f:
            instances.append(json.loads(line))
            num += 1

    # generate an answer for each question
    for i in range(SEED_SIZE, len(instances)):
        prompt += "Given question: " + instances[i]['question']
        # extract options from the question
        options = extract_options(instances[i]['question'], instances[i]['id'])
        # generate the correct index for the question
        ans = extract_idx(call_openai_api(prompt))
        gen = {"id": instances[i]['id'], "question": instances[i]['question'], "options": options, "correct_index": [ans]}
        dump_jsonl(gen, args.save_path)
        print(instances[i]['id'], "completed!")


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
    generate_answers(args, instruction, file)
