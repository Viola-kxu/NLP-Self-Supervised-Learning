import argparse
import re
import warnings

from utils import *


def generate_answers(instruction, file):
    # read current questions
    num = 1
    instances = []
    with (open(file, 'r', encoding="utf-8") as f):
        for line in f:
            instances.append(json.loads(line))
            num += 1

    # generate an answer for each question
    # for j in []: # generate for specific questions
    #     i = j - 1
    for i in range(0, num - 1):
        if 'answer' in instances[i].keys():
            continue
        prompt = str(instances[i]['question']) + "\n#Your response#: "
        # extract options from the question
        # options = extract_options(instances[i]['question'], instances[i]['id'])
        # generate the correct index for the question
        response = call_openai_api(instruction + prompt)
        print("Answer generated for id = " + str(i + 1) + " -------------------")
        # print(response)
        ans = response
        generated = {"question": instances[i]['question'], "answer": ans}
        instances[i] = generated

        # test - dump the generated answers to a file
        print("Correct answer: ", generated['answer'], sep="")
        dump_jsonl(generated, '../data/generated_answers_test.jsonl')

    return instances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='../data/gsm8k_generated.jsonl')
    parser.add_argument('--save_path', type=str, default='../data/gsm8k_generated.jsonl')
    parser.add_argument('--ins_file', type=str, default='instructions/instruction_gsm8k_answer.txt')
    args = parser.parse_args()

    with open(args.ins_file, 'r', encoding="utf-8") as f:
        instruction = f.read()
    instances = generate_answers(instruction, args.file)

    # write the generated answers to the file
    target_file = open(args.save_path, mode="w").close()  # clear the file
    for instance in instances:
        dump_jsonl(instance, args.save_path)

