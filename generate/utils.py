import json
import time
import openai

openai.api_key = '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'


def dump_jsonl(data, output_path, append=True):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


def call_openai_api(prompt):
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
            time.sleep(60)
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

    # print(res['choices'][0]['message']['content'])
    return res['choices'][0]['message']['content']
