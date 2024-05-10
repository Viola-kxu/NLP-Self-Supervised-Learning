import json
import time
import openai

# openai.api_key = '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'  # Tian's API
openai.api_key = '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'  # Viola's API


def dump_jsonl(data, output_path, append=True):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


# def call_openai_api(prompt):
#     model_id = "ftjob-mnEEtWBCUYjubJq2YyBFRX6R"
#     message = [
#         {"role": "user", "content": prompt + "\n#Rewritten Question#: "}
#     ]
#
#     while True:
#         try:
#             res = openai.ChatCompletion.create(
#                 model=model_id,
#                 messages=message,
#                 temperature=1.0,
#                 max_tokens=512
#             )
#             break
#         except openai.error.RateLimitError:
#             print('openai.error.RateLimitError\nRetrying...')
#             time.sleep(60)
#         except openai.error.ServiceUnavailableError:
#             print('openai.error.ServiceUnavailableError\nRetrying...')
#             time.sleep(20)
#         except openai.error.Timeout:
#             print('openai.error.Timeout\nRetrying...')
#             time.sleep(20)
#         except openai.error.APIError:
#             print('openai.error.APIError\nRetrying...')
#             time.sleep(20)
#         except openai.error.APIConnectionError:
#             print('openai.error.APIConnectionError\nRetrying...')
#             time.sleep(20)
#
#     # print(res)
#     # print(res['choices'][0]['message']['content'])
#     return res['choices'][0]['message']['content']

def call_openai_api(prompt):
    model_id = "ft:gpt-3.5-turbo-1106:personal:math1:9N6jTW0L"
    messages = [
        {"role": "system", "content": "You are a math problem solver."},
        {"role": "user", "content": prompt}
    ]
    max_retries = 5
    attempts = 0

    while attempts < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model_id,
                messages=messages,
                temperature=0.5,
                max_tokens=512
            )
            return response['choices'][0]['message']['content']
        except openai.error.RateLimitError:
            print('Rate limit exceeded, retrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('Service unavailable, retrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('Request timed out, retrying...')
            time.sleep(20)
        except openai.error.APIError as e:
            print(f'API error: {e}, retrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('Connection error, retrying...')
            time.sleep(20)
        attempts += 1

    raise Exception("Failed to process the request after several attempts.")


