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

    # generate an answer for each question
    # for j in []: # generate for specific questions
    #     i = j - 1
    for i in range(0, num - 1):
        # if 'correct_index' in instances[i].keys():
        #     continue
        prompt = str(instances[i]['question']) + "\n#Your response#: "
        # extract options from the question
        # options = extract_options(instances[i]['question'], instances[i]['id'])
        # generate the correct index for the question
        response = call_openai_api(instruction + prompt)
        print("Answer generated for id = " + str(i + 1) + " -------------------")
        # print(response)
        options = extract_options(response)
        ans = extract_idx(response)
        generated = {"id": instances[i]['id'], "question": instances[i]['question'], "options": options,
                     "correct_index": [ans]}
        instances[i] = generated

        # test - dump the generated answers to a file
        print("Options: ", options, ", Correct index: ", generated['correct_index'], sep="")
        dump_jsonl(generated, args.save_path)

    return instances


def extract_idx(input_text):
    match = re.search(r"[\s\S]*Correct Index:\D*(\d+)", input_text)
    if match:
        return int(match.group(1))
    warnings.warn("ERROR: answer choice extraction failed for the following input\n" + input_text)
    return None


def extract_options(question_text):
    pattern = r'\([A-Z]\)[\s\S]*?(?=[\s\,]*\([A-Z]\)|[\s\,]*\n|[\s\,]*$|[\s\,]*;)'
    options = re.findall(pattern, question_text[question_text.find("New"):])
    if len(options) == 0:
        warnings.warn("ERROR: answer options extraction failed for the following input")
    return options[:4]


def report_invalid_answers(file):
    ids = []
    with (open(file, 'r', encoding="utf-8") as f):
        for line in f:
            data = json.loads(line)
            if 'options' not in data.keys() or 'correct_index' not in data.keys() or len(data['options']) != 4 or \
                    data['correct_index'][0] is None or not (0 <= data['correct_index'][0] < 4):
                ids.append(data['id'])
    return ids


def temp_accuracy():
    labels = []
    path1 = "../data/finetuned_sat_math_with_answers.jsonl"
    path2 = "../data/archive/sat_math_validation.jsonl"
    with (open(path1, 'r', encoding="utf-8") as f):
        for line in f:
            labels.append(json.loads(line)['correct_index'][0])
    idx, correct = 0, 0
    with (open(path2, 'r', encoding="utf-8") as f):
        for line in f:
            if json.loads(line)['correct_index'][0] == labels[idx]:
                correct += 1
            idx += 1
    print("Accuracy: ", correct / idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='../data/archive/sat_math_val_wout_ans.jsonl')
    parser.add_argument('--save_path', type=str, default='../data/finetuned_sat_math_with_answers2.jsonl')
    parser.add_argument('--ins_file', type=str, default='../generate/instructions/instruction_answer.txt')
    args = parser.parse_args()

    with open(args.ins_file, 'r', encoding="utf-8") as f:
        instruction = f.read()
    instances = generate_answers(instruction, args.file)

    # write the generated answers to the file
    target_file = open(args.save_path, mode="w").close()  # clear the file
    for instance in instances:
        dump_jsonl(instance, args.save_path)

    print("Invalid indices", report_invalid_answers(args.file))

