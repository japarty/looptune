import datetime
import os
import csv
# from tqdm.notebook import tqdm
from typing import Union
from tqdm.notebook import tqdm_notebook as tqdm

# Prompting

def get_response(message, parameters, client, model='local-model'):
    completion = client.chat.completions.create(
        model=model,
        messages=message,
        **parameters
    )
    return completion.choices[0].message.content


def looped_prompt_classification(texts, labels, model_name, parameters, client, message_creator, output_path: bool | str = False,
                  prompt_log: bool | str = False):
    if model_name == 'model-local':
        model_name = client.models.list().data[0].dict()['id']

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    if prompt_log:
        if prompt_log.endswith('.csv'):
            prompt_log_path = prompt_log
        else:
            prompt_log_path = prompt_log + f'/prompt_log.csv'
    else:
        prompt_log_path = f'prompt_log.csv'

    if output_path:
        if output_path.endswith('.csv'):
            response_log_path = output_path
        else:
            response_log_path = output_path + f"/{model_name.rsplit('/')[-1]}_{timestamp}.csv"
    else:
        response_log_path = f"{model_name.rsplit('/')[-1]}_{timestamp}.csv"

    # add row to prompt log
    with open(prompt_log_path, mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not os.path.exists(prompt_log_path):
            writer.writerow(['model', 'timestamp', 'message', 'params'])
        writer.writerow([model_name, timestamp, message_creator(), parameters])

    with open(response_log_path, mode='w', newline='', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['text', 'label', 'timestamp', 'response'])

        for i, (txt, label) in tqdm(
                enumerate(zip(texts, labels))):  # iterate through each text value of first df column
            message = message_creator(txt)
            response = get_response(message, parameters, client, model_name)
            writer.writerow([txt, label, timestamp, response])
            csv_file.flush()