import argparse
import random
import re

import openai
import time
import json

openai.api_key = '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'


def get_dataset(args, instruction, seed, count=10):
    prompt = instruction

    # read current questions
    num = 1
    delete = []
    instances = []
    with (open(seed, 'r', encoding="utf-8") as f):
        for line in f:
            instances.append(json.loads(line))
            num += 1

    for _ in range(count):
        question_prompt = ""
        question_ids = random.sample(range(0, num - 1), 10)
        for question_id in question_ids:
            question_prompt += "#Given Question and Options#: " + "question id " + str(question_id) + ": " + instances[question_id]['question'] + "\n\n"

        # Assuming get_res_batch and the rest of your code is defined correctly
        generated = get_res_batch(prompt + question_prompt)
        id = extract_int(generated)
        print("the worst question is" + generated + ": " + instances[id]['question'] + "\n\n")
        delete.append(int(id))

    delete_question_by_id(seed, delete)



def delete_question_by_id(file, ids):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    new_data = [data[i] for i in range(len(data)) if i not in ids]
    for i in range(len(new_data)):
        dump_jsonl(new_data[i], args.save_path)

def extract_int(text):
    match = re.search(r"Worst Question ID:\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return None

def get_res_batch(prompt):
    message = [
        {"role": "user", "content": prompt +  "\n#Worst Question id#: "}
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
    parser.add_argument('--file', type=str, default='../data/generated_questions.jsonl')
    parser.add_argument('--save_path', type=str, default='../data/filtered_questions.jsonl')
    parser.add_argument('--ins_file', type=str, default='instructions/instruction_filter.txt')
    args = parser.parse_args()
    file = args.file

    with open(args.ins_file, 'r', encoding="utf-8") as f:
        instruction = f.read()
    get_dataset(args, instruction, file)
