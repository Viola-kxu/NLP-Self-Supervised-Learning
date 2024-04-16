import argparse
import random
import re
import openai
import time
import json
import warnings


openai.api_key = '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'


def get_dataset(args, instruction, seed):
    prompt = instruction

    # read current questions
    num = 1
    instances = []
    with (open(seed, 'r', encoding="utf-8") as f):
        for line in f:
            instances.append(json.loads(line))
            num += 1

    # generate an answer for each question
    for i in range(len(instances)):
        prompt = prompt + "Given question: " + instances[i]['question']
        # extract options from the question
        options = extract_options(instances[i]['question'], instances[i]['id'])
        # generate the correct index for the question
        ans = extract_int(get_res_batch(prompt))
        gen = {"id": instances[i]['id'], "question":instances[i]['question'], "options": options, "correct_index": [ans]}
        dump_jsonl(gen, args.save_path)
        print(instances[i]['id'], "completed!")

def extract_int(input_text):
    match = re.search(r"Correct Index:\s*(\d+)", input_text)
    if match:
        return int(match.group(1))
    else:
        return None

def extract_options(question_text, quesiton_id):
    # start_index = question_text.find("Answer Choices:")
    # if start_index == -1:
    #     raise ValueError("Question does not contain 'Answer Choices:'")
    # start_index += len("Answer Choices:") + 1
    # answer_text = question_text[start_index:].strip()
    # # print answers text character by character and show in one line
    # for c in answer_text:
    #     print(c, end='')
    # print('\n')
    # options = re.findall(r"\([A-D]\)[.\n]+?(?=\([A-D]\))", answer_text)
    # print(options)
    # formatted_options = [f'{option.strip()}' for option in options]
    # print(formatted_options)
    # return formatted_options

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



def get_res_batch(prompt):
    message = [
        {"role": "user", "content": prompt + "\n#Correct Index#: "}
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
    parser.add_argument('--file', type=str, default='../data/filtered_questions.jsonl')
    parser.add_argument('--save_path', type=str, default='../data/generated_answers.jsonl')
    parser.add_argument('--ins_file', type=str, default='instructions/instruction_answer.txt')
    args = parser.parse_args()
    file = args.file

    with open(args.ins_file, 'r', encoding="utf-8") as f:
        instruction = f.read()
    get_dataset(args, instruction, file)
