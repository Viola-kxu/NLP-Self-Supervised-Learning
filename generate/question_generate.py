import argparse
import random

import openai
import time
import json

openai.api_key = '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'


def get_dataset(args, instruction, seed, count=5):
    prompt = instruction

    # read current questions
    num = 1
    instances = []
    with (open(seed, 'r', encoding="utf-8") as f):
        for line in f:
            instances.append(json.loads(line))
            num += 1

    for _ in range(count):
        # select 5 random questions seeds
        question_prompt = ""
        question_ids = random.sample(range(0, num - 1), 5)
        print("\nGenerating " + str(_) + " instance ----------------------- \nQuestions selected: ", question_ids)
        for question_id in question_ids:
            question_prompt += "#Given Question#: " + instances[question_id]['question'] + "\n\n"

        # generate a question
        generated = get_res_batch(prompt + question_prompt)
        gen_instance = {"id": num, "question": generated}
        dump_jsonl(gen_instance, args.save_path)

        # update of questions instances
        num += 1
        instances.append(gen_instance)

# def get_dataset(args, instruction, file):
#     with open(file, 'r', encoding="utf-8") as f:
#         data = []
#         for line_number, line in enumerate(f, 1):
#             try:
#                 if line.strip():
#                     data.append(json.loads(line))
#             except json.decoder.JSONDecodeError as e:
#                 print(f"Error decoding JSON on line {line_number}: {line}")
#                 print(e)
#                 continue
#
#     for idx, item in enumerate(data):
#         input = item["question"] + item.get("answer", "")
#         ans = get_res_batch(instruction, input)
#         gen = {"id": item["id"], "input": input, "question": ans}
#         dump_jsonl(gen, args.save_path)
#         print(idx + 1, "completed!")


def get_res_batch(prompt):
    message = [
        {"role": "user", "content": prompt + "\n#Rewritten Question#: "}
    ]

    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=1.0,
                max_tokens=512
            )
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(30)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    print(res['choices'][0]['message']['content'])
    return res['choices'][0]['message']['content']


def dump_jsonl(data, output_path, append=True):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='../data/result.json')
    parser.add_argument('--save_path', type=str, default='../data/result.json')
    parser.add_argument('--ins_file', type=str, default='instructions/instruction_sat_math.txt')
    args = parser.parse_args()
    file = args.file

    with open(args.ins_file, 'r', encoding="utf-8") as f:
        instruction = f.read()
    get_dataset(args, instruction, file)
