import datetime
import os
import csv
from tqdm.notebook import tqdm_notebook
from typing import Union

# Prompting

def get_response(message, parameters, client, model='local-model'):
    completion = client.chat.completions.create(
        model=model,
        messages=message,
        **parameters
    )
    return completion.choices[0].message.content


def looped_prompt(texts, labels, parameters, client, message_creator, output_dir, prompt_log: bool | str = False):
    model_name = client.models.list().data[0].dict()['id']
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    if prompt_log:
        if prompt_log.endswith('.csv'):
            prompt_log_path = prompt_log
        else:
            prompt_log_path = prompt_log

    response_log_path = output_dir + f'/{model_name}_{timestamp}.csv'

    # add row to prompt log
    with open(prompt_log_path, mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not os.path.exists(prompt_log_path):
            writer.writerow(['model', 'timestamp', 'message', 'params'])
        writer.writerow([model_name, timestamp, message_creator(), parameters])

    with open(response_log_path, mode='w', newline='', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter='|')
        writer.writerow(['text', 'label', 'timestamp', 'response'])

        for i, (txt, label) in tqdm_notebook(
                enumerate(zip(texts, labels))):  # iterate through each text value of first df column
            message = message_creator(txt)
            response = get_response(message, parameters, client)
            writer.writerow([txt, label, timestamp, response])
            csv_file.flush()


def get_response(text, parameters, client, create_messages, model='local-model'):
    completion = client.chat.completions.create(
        model=model,
        messages=create_messages(text),
        **parameters
    )
    return completion.choices[0].message.content


def looped_prompt(texts, labels, parameters, client, root_path, create_messages, model='local-model'):
    if model == 'local-model':
        model_name = client.models.list().data[0].dict()['id']
    else:
        model_name = model
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    prompt_log_path = root_path / 'output/prompting/prompt_log.csv'
    response_log_path = root_path / f'output/prompting/responses/{timestamp}.csv'

    # add row to prompt log
    with open(prompt_log_path, mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not os.path.exists(prompt_log_path):
            writer.writerow(['model', 'timestamp', 'message', 'params'])
        writer.writerow([model_name, timestamp, create_messages(), parameters])

    with open(response_log_path, mode='w', newline='', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter='|')
        writer.writerow(['text', 'label', 'timestamp', 'response'])

        for i, (txt, label) in tqdm_notebook(
                enumerate(zip(texts, labels))):  # iterate through each text value of first df column
            response = get_response(txt, parameters, client, create_messages, model)
            writer.writerow([txt, label, timestamp, response])
            csv_file.flush()