import json


def remove_correct_index(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            # Remove 'correct_index' from the dictionary
            if 'correct_index' in data:
                del data['correct_index']

            # Write the modified data back to the new file
            json.dump(data, outfile)
            outfile.write('\n')


# Specify the input and output file paths
input_file = '../data/archive/sat_math_validation.jsonl'
output_file = '../data/archive/sat_math_val_wout_ans.jsonl'

if __name__ == "__main__":
    remove_correct_index(input_file, output_file)
