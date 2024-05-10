import json

# This program converts the jsonl file to formatted GPT3.5 finetuning jsonl file

def convert_GPT2dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Load the JSON object from each line of the original file
            data = json.loads(line)

            # Format the prompt to include the question and all options
            question = data['question']
            options = " ".join([f"{opt}" for opt in data['options']])
            prompt = f"{question} Answer Choices: {options}"

            # Get the correct answer using the correct_index
            completion = data['options'][data['correct_index'][0]]

            # Create a new JSON object for the fine-tuning dataset
            formatted_data = {
                "prompt": prompt,
                "completion": completion
            }

            # Write the new JSON object to the output file
            json.dump(formatted_data, outfile)
            outfile.write('\n')

def convert_GPT3dataset(input_file, output_file):
    system_message = "This is a math problem solver model, which will give the correct answer of each math problem."

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            question = data['question'] + " Answer Choices: " + " ".join(data['options'])
            correct_answer = data['options'][data['correct_index'][0]]

            formatted_data = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": correct_answer}
                ]
            }

            json.dump(formatted_data, outfile)
            outfile.write('\n')

# Specify the input and output file paths
input_file1 = '../data/generated_dataset.jsonl'
input_file2 = '../data/archive/sat_math_validation.jsonl'
output_file1 = '../data/GPT3.5_formatted_trained.jsonl'
output_file2 = '../data/GPT3.5_formatted_validation.jsonl'


if __name__ == "__main__":
    convert_GPT3dataset(input_file1, output_file1)
    convert_GPT3dataset(input_file2, output_file2)