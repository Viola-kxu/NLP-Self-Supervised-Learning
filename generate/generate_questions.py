import argparse
import random

from utils import *

SEED_SIZE = 120
NUM_SELECTION = 3
GENERATE_COUNT = 26


def generate_question(args, instruction, seed):
    # read current questions
    num = 1
    instances = []
    with (open(seed, 'r', encoding="utf-8") as f):
        for line in f:
            instances.append(json.loads(line))
            num += 1

    for _ in range(GENERATE_COUNT):
        # select NUM_QUES random questions from seeds
        #      + NUM_QUES random questions from new generated questions
        question_prompt = ""
        question_ids = random.sample(range(0, SEED_SIZE - 1), NUM_SELECTION) + random.sample(range(SEED_SIZE, num - 1), NUM_SELECTION)
        random.shuffle(question_ids) # shuffle order
        print("\nGenerating instance " + str(_) + f" (id = {num}) ----------------------- \nQuestions selected: ", question_ids)
        for question_id in question_ids:
            question_prompt += "#Given Question#: " + instances[question_id]['question'] + "\n\n"

        # generate a question
        generated = call_openai_api(instruction + question_prompt)
        gen_instance = {"id": num, "question": generated}
        dump_jsonl(gen_instance, args.save_path)
        # print("Generated question (id = " + str(num) + "): \n", generated)

        # update of questions instances
        num += 1
        instances.append(gen_instance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='../data/generated_dataset.jsonl')
    parser.add_argument('--save_path', type=str, default='../data/generated_dataset.jsonl')
    parser.add_argument('--ins_file', type=str, default='instructions/instruction_sat_math.txt')
    args = parser.parse_args()
    file = args.file

    print("GENERATING QUESTIONS -----------------------")

    with open(args.ins_file, 'r', encoding="utf-8") as f:
        instruction = f.read()
    generate_question(args, instruction, file)

    print("GENERATING QUESTIONS DONE -----------------------")
