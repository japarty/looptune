import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sn
import itertools

from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments, \
    DataCollatorWithPadding, AutoTokenizer, pipeline
from transformers.modelcard import parse_log_history
from yeelight import Bulb


# Prepare functions

def to_binary_classification(x, convert_dict={"Positive": 'Pathos', 'Negative': 'Pathos'}):
    """
    Converts labels to binary classification ('Pathos' or 'No_pathos').

    Args:
        x (str): The original label.
        convert_dict (dict, optional): A dictionary mapping original labels
            to their corresponding binary representation. Defaults to
            {"Positive":'Pathos', 'Negative':'Pathos'}.

    Returns:
        str: The converted binary label.
    """
    if x in convert_dict.keys():
        return convert_dict[x]
    else:
        return x

# # Create run configuration dicts


def ratio_split_tuple(split):
    split_s = sum([i for i in split if isinstance(i, (int, float))])
    new_split = tuple([i/split_s if isinstance(i, (int, float)) else i for i in split])
    return new_split


def param_combinations(param_dict):
    if isinstance(param_dict['split'], list):
        param_dict['split'] = list(map(ratio_split_tuple, param_dict['split']))
    else:
        param_dict['split'] = ratio_split_tuple(param_dict['split'])

    param_dict={i:[q] if type(q) is not list else q for (i,q) in param_dict.items()}
    keys = list(param_dict.keys())
    combinations = list(itertools.product(*param_dict.values()))
    result = [{keys[i]: combination[i] for i in range(len(keys))} for combination in combinations]
    return result

# # From csv to ds

# # # Dataset specific load to 2-col df ['text', label']

def load_PolarIs(path):
    df = pd.read_excel(path)
    df['label'] = df[['No_pathos', 'Positive', 'Negative']].idxmax(axis=1)
    df = df.rename(columns={'Sentence': 'text'})
    df = df[['text', 'label']]
    return df


def df_to_ds(df, split=(0.8, 0.1, 0.1), binary=False, balanced=False):
    """
    Converts a DataFrame into a Hugging Face DatasetDict, applying label mapping and optional dataset splitting and balancing.

    Args:
        df (pd.DataFrame): The DataFrame containing text data (column 'sentence') and labels (column 'label').
        split (tuple): A tuple of floats defining the train/test split ratios.
                      Example: [0.7, 0.3] for 70% train, 30% test. Can include an optional third element for validation.
        tg_map (dict): A dictionary mapping original labels to numerical indices.
        binary (bool, optional): If true, converts labels to binary
        balanced (bool, optional): If True, balances the training dataset by sampling with respect to class frequencies. Defaults to False.

    Returns:
        DatasetDict: A Hugging Face DatasetDict containing splits ('train', 'test', and optionally 'validate').

    """
    if binary:
        df['label'] = df['label'].apply(lambda x: to_binary_classification(x))

    # target map and reversed target map
    tg_map = {k: i for i, k in enumerate(df['label'].unique())}
    reversed_tg_map = {v: k for k, v in tg_map.items()}

    df['label'] = df['label'].map(tg_map)

    if balanced:
        train_test = [pd.Series(name='text'), pd.Series(name='text'), pd.Series(name='label'), pd.Series(name='label')]
        n_for_training = df.groupby('label').count()['text'].min()
        train_percentage = df.groupby('label').count()['text'].apply(
            lambda x: (split[0] * n_for_training) / x).to_dict()
        for label_name, group in df.groupby('label'):
            train_test_group = train_test_split(group['text'], group['label'],
                                                test_size=1 - train_percentage[label_name], random_state=42,
                                                shuffle=True)
            train_test = [pd.concat([i[0], i[1]], axis=0) for i in zip(train_test, train_test_group)]
    else:
        train_test = train_test_split(df['text'], df['label'], stratify=df['label'], test_size=split[1],
                                      random_state=42, shuffle=True)
    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(pd.concat([train_test[0], train_test[2]], axis=1))
    if len(split) == 2:
        ds['test'] = Dataset.from_pandas(pd.concat([train_test[1], train_test[3]], axis=1))
    elif len(split) == 3:
        val_ratio = split[2] / (split[1] + split[2])
        test_validate = train_test_split(train_test[1], train_test[3], stratify=train_test[3], test_size=val_ratio,
                                         random_state=42, shuffle=True)
        ds['test'] = Dataset.from_pandas(pd.concat([test_validate[0], test_validate[2]], axis=1))
        ds['validate'] = Dataset.from_pandas(pd.concat([test_validate[1], test_validate[3]], axis=1))

    return ds, tg_map, reversed_tg_map


# Train functions

def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}


def save_logs_from_training_run(trainer, params, timestamp, trained_model_path, colab, target_map, root_path):
    """
    Collects training logs, metadata, and performance metrics, saving them to a CSV file for tracking.

    Args:
        trainer (Trainer): The Hugging Face Trainer object.
        params (dict): A dictionary containing model and training parameters.
        timestamp (str): Datetime string for identifying the training run.
        trained_model_path (str): The output path where the trained model is saved.
        colab (bool): Indicates whether the code was run in Google Colab.
        target_map (dict): A mapping of original labels to numerical indices.
    """
    log_history = parse_log_history(trainer.state.log_history)
    log_df = pd.DataFrame(log_history[1])
    log_df.insert(0, 'model', params['model'])
    log_df.insert(0, 'timestamp', timestamp)
    log_df['binary'] = params['binary']
    log_df['balanced'] = params['balanced']
    log_df['split'] = str(params['split'])
    log_df['target_map'] = str(target_map)
    log_df['colab'] = colab
    log_df['model_path'] = trained_model_path
    log_df['samples_per_s'] = log_history[0]['train_samples_per_second']
    log_df['steps_per_s'] = log_history[0]['train_steps_per_second']
    log_df['per_device_train_batch_size'] = params['per_device_train_batch_size']
    log_df['per_device_eval_batch_size'] = params['per_device_eval_batch_size']

    float_cols = log_df.select_dtypes(include='float64')
    log_df[float_cols.columns] = float_cols.apply(lambda x: round(x, 3))

    log_path = os.path.join(root_path, 'output/training_logs.csv')
    log_df.to_csv(log_path, mode='a', header=not os.path.isfile(log_path), index=False)


def finetune(ds, params, target_map, reversed_target_map, save_logs, root_path, colab):
    """
    Fine-tunes a pre-trained language model for text classification. Handles tokenization, model loading, and training.

    Args:
        params (dict): A dictionary containing model and training parameters.
        ds (DatasetDict): A Hugging Face DatasetDict containing 'train', 'test', and optionally 'validate' splits.
        target_map (dict): A mapping of original labels to numerical indices.
        save_logs (bool): If True, saves training logs and metrics.
        reversed_target_map:
        root_path:
    """
    params_passed = {k: params[k] for k in params if k in ['num_train_epochs',
                                                           'save_strategy',
                                                           'per_device_train_batch_size',
                                                           'per_device_eval_batch_size']}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    trained_model_path = f"output/models/{params['model']}_{timestamp}"

    try:
        def tokenize_fn(batch):
            return tokenizer(batch['text'], truncation=True)

        # Tokenize dataset
        tokenizer = AutoTokenizer.from_pretrained(params['model'], trust_remote_code=True)
        # tokenizer.pad_token = tokenizer.eos_token
        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # tokenizer.add_special_tokens({'pad_token': '<pad>'})
            # model.resize_token_embeddings(len(tokenizer))
        tokenized_datasets = ds.map(tokenize_fn, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Change labels
        config = AutoConfig.from_pretrained(params['model'], trust_remote_code=True)
        config.vocab_size = tokenizer.vocab_size
        config.id2label = reversed_target_map
        config.label2id = target_map

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            params['model'], config=config, ignore_mismatched_sizes=True, trust_remote_code=True)

        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     model.resize_token_embeddings(len(tokenizer))

        training_args = TrainingArguments(
            output_dir=os.path.join(root_path, f'{trained_model_path}/checkpoints'),
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            use_cpu=False,
            **params_passed
        )

        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
        trainer.train()
        if save_logs:
            save_logs_from_training_run(trainer, params, timestamp, trained_model_path, colab, target_map, root_path)

    except Exception as e:
        print(f'Error: {e}')
        if save_logs:
            err_df = pd.DataFrame([[trained_model_path, str(e)]])
            err_path = os.path.join(root_path, 'output/error_logs.csv')
            err_df.to_csv(err_path, mode='a', header=not os.path.isfile(err_path), index=False)


# Validate functions

def get_log_for_val(checkpoint_path, logs_path, sort_col='Step'):
    """
    Retrieves the training log entry corresponding to a given checkpoint.

    Args:
        checkpoint_path (str): The path to the checkpoint directory.
        logs_path (str): The path to the CSV file containing the training logs.
        sort_col (str, optional): The column to sort by when searching for the latest checkpoint. Defaults to 'Step'.

    Returns:
        pd.Series: A Pandas Series representing a single row of the training logs.
    """
    training_logs = pd.read_csv(logs_path)
    if 'checkpoint-' in checkpoint_path:
        temp_path = checkpoint_path.rsplit('models/', 1)[-1]
        model_path, checkpoint_num = temp_path.rsplit('/checkpoints/checkpoint-')
        row = training_logs[(training_logs['model_path'].apply(lambda x: x.rsplit('models/', 1)[-1] == model_path)) & (
                    training_logs['Step'] == int(checkpoint_num))]
    else:
        row = training_logs[training_logs['model_path'] == checkpoint_path].sort_values(sort_col, ascending=False).head(
            1)
    return row.iloc[0]


def validate(row, ds):
    """
    Loads a trained model from a checkpoint and evaluates its performance on the validation set.

    Args:
        row (pd.Series): A single row from the training logs, containing checkpoint information.
        ds (DatasetDict): A Hugging Face DatasetDict containing a 'validate' split.

    Returns:
        tuple: A tuple containing:
            * predicted (list): List of predicted labels.
            * val_labels (list): List of true labels.
    """
    val_sentences = ds['validate']['sentence']
    val_labels = [reversed_target_map[i] for i in ds['validate']['label']]

    classifier = pipeline('text-classification',
                          model=os.path.join(row['model_path'], 'checkpoints', f"checkpoint-{row['Step']}"), device=0)
    predicted = [i['label'] for i in classifier(val_sentences)]
    return predicted, val_labels


def val_metrics(predicted, val_labels, target_map):
    """
    Calculates and displays validation metrics (accuracy, F1-score, confusion matrix).

    Args:
        predicted (list): List of predicted labels.
        val_labels (list): List of true labels.
        target_map (dict): A mapping of original labels to numerical indices.
    """
    print("acc:", accuracy_score(val_labels, predicted))
    print("f1:", f1_score(val_labels, predicted, average='macro'))

    cm = confusion_matrix(val_labels, predicted, normalize='true')
    plot_cm(cm, target_map)


def plot_cm(cm, target_map):
    classes = list(target_map.keys())
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    ax = sn.heatmap(df_cm, annot=True, fmt='.2g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")


def yeelight_eow_notification(bulb_ip):
    bulb = Bulb(bulb_ip)
    bulb.turn_on()
    bulb.set_rgb(0, 255, 0)
    bulb.set_brightness(100)
