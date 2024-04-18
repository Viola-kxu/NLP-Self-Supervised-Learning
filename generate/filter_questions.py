import argparse
import random
import re

from utils import *
from generate_questions import SEED_SIZE

COUNT = 5
FILTER_SELECTION = 5


def get_delete_idx(instruction, seed_file):
    # read current questions
    num = 1
    delete_idx = []
    instances = []
    with (open(seed_file, 'r', encoding="utf-8") as f):
        for line in f:
            instances.append(json.loads(line))
            num += 1

    for _ in range(COUNT):
        question_prompt = ""
        question_ids = random.sample(range(SEED_SIZE, num - 1), FILTER_SELECTION)
        for question_id in question_ids:
            question_prompt += ("#Given Question and Options#: " + "question id " + str(question_id) + ": " +
                                instances[question_id]['question'] + "\n\n")

        generated = call_openai_api(instruction + question_prompt)
        ids = int(extract_idx_from_msg(generated))
        print("Filtered index among " + str(question_ids) + ": " + str(ids))
        delete_idx.append(ids)

    return instances, delete_idx


def delete_question_by_id(instances, ids):
    instance_id = 1
    for i in range(len(instances)):
        if i not in ids:
            dump_jsonl(instances[i], args.save_path)
            instance_id += 1


def extract_idx_from_msg(text):
    match = re.search(r"Worst Question ID:\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='../data/generated_dataset.jsonl')
    parser.add_argument('--save_path', type=str, default='../data/generated_dataset.jsonl')
    parser.add_argument('--ins_file', type=str, default='instructions/instruction_filter.txt')
    args = parser.parse_args()
    file = args.file

    print("FILTERING IN PROGRESS -----------------------")

    # get delete index
    with open(args.ins_file, 'r', encoding="utf-8") as f:
        instruction = f.read()
    instances, delete_idx = get_delete_idx(instruction, file)
    # print("Filtered indices: ", delete_idx, " -----------------------")

    # clear target file
    target_file = open(args.save_path, mode="w").close()

    # delete questions
    delete_question_by_id(instances, delete_idx)

    print("FILTERING DONE -----------------------")
