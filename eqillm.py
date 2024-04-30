import os
import datetime

import evaluate
import numpy as np
import pandas as pd
import seaborn as sn
import itertools
import torch
import wandb

from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments, DataCollatorWithPadding, AutoTokenizer, pipeline, BitsAndBytesConfig
from transformers.modelcard import parse_log_history
from yeelight import Bulb


MAX_LEN = 256

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="fp4",
#     bnb_4bit_use_double_quant=False,
#     bnb_4bit_compute_dtype=torch.float16
# )

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

def load_PolarIs(path, binary=False):
    df = pd.read_excel(path)
    df['label'] = df[['No_pathos', 'Positive', 'Negative']].idxmax(axis=1)
    df = df.rename(columns={'Sentence': 'text'})
    df = df[['text', 'label']]
    if binary:
        df['label'] = df['label'].apply(lambda x: to_binary_classification(x))
    return df

def encode_labels(dataframe):
    encoder = LabelEncoder()
    dataframe['label'] = encoder.fit_transform(dataframe['label'])
    target_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    return dataframe, target_map


def split_ds(dataset, train_size=0.8, val_size=None):
    dataset = dataset.train_test_split(train_size=train_size, seed=42)
    if val_size is not None:
        val_ratio = 1 - (val_size/(1 - train_size))
        dataset2 = dataset['test'].train_test_split(train_size=val_ratio, seed=42)

        dataset['test'] = dataset2['train']
        dataset['val'] = dataset2['test']
    return dataset


# def df_to_ds(df, split=(0.8, 0.1, 0.1), balanced=False):
#     """
#     Converts a DataFrame into a Hugging Face DatasetDict, applying label mapping and optional dataset splitting and balancing.
#
#     Args:
#         df (pd.DataFrame): The DataFrame containing text data (column 'sentence') and labels (column 'label').
#         split (tuple): A tuple of floats defining the train/test split ratios.
#                       Example: [0.7, 0.3] for 70% train, 30% test. Can include an optional third element for validation.
#         tg_map (dict): A dictionary mapping original labels to numerical indices.
#         balanced (bool, optional): If True, balances the training dataset by sampling with respect to class frequencies. Defaults to False.
#
#     Returns:
#         DatasetDict: A Hugging Face DatasetDict containing splits ('train', 'test', and optionally 'validate').
#
#     """
#     # target map and reversed target map
#     tg_map = {k: i for i, k in enumerate(df['label'].unique())}
#
#     df['label'] = df['label'].map(tg_map)
#
#     if balanced:
#         train_test = [pd.Series(name='text'), pd.Series(name='text'), pd.Series(name='label'), pd.Series(name='label')]
#         n_for_training = df.groupby('label').count()['text'].min()
#         train_percentage = df.groupby('label').count()['text'].apply(
#             lambda x: (split[0] * n_for_training) / x).to_dict()
#         for label_name, group in df.groupby('label'):
#             train_test_group = train_test_split(group['text'], group['label'],
#                                                 test_size=1 - train_percentage[label_name], random_state=42,
#                                                 shuffle=True)
#             train_test = [pd.concat([i[0], i[1]], axis=0) for i in zip(train_test, train_test_group)]
#     else:
#         train_test = train_test_split(df['text'], df['label'], stratify=df['label'], test_size=split[1],
#                                       random_state=42, shuffle=True)
#     ds = DatasetDict()
#     ds['train'] = Dataset.from_pandas(pd.concat([train_test[0], train_test[2]], axis=1))
#     if len(split) == 2:
#         ds['test'] = Dataset.from_pandas(pd.concat([train_test[1], train_test[3]], axis=1))
#     elif len(split) == 3:
#         val_ratio = split[2] / (split[1] + split[2])
#         test_validate = train_test_split(train_test[1], train_test[3], stratify=train_test[3], test_size=val_ratio,
#                                          random_state=42, shuffle=True)
#         ds['test'] = Dataset.from_pandas(pd.concat([test_validate[0], test_validate[2]], axis=1))
#         ds['validate'] = Dataset.from_pandas(pd.concat([test_validate[1], test_validate[3]], axis=1))
#
#     return ds, tg_map


# Train functions

# def save_logs_from_training_run(trainer, params, timestamp, trained_model_path, colab, target_map):
#     """
#     Collects training logs, metadata, and performance metrics, saving them to a CSV file for tracking.
#
#     Args:
#         trainer (Trainer): The Hugging Face Trainer object.
#         params (dict): A dictionary containing model and training parameters.
#         timestamp (str): Datetime string for identifying the training run.
#         trained_model_path (str): The output path where the trained model is saved.
#         colab (bool): Indicates whether the code was run in Google Colab.
#         target_map (dict): A mapping of original labels to numerical indices.
#     """
#     log_history = parse_log_history(trainer.state.log_history)
#     log_df = pd.DataFrame(log_history[1])
#     log_df.insert(0, 'model_name', params['model_name'])
#     log_df.insert(0, 'timestamp', timestamp)
#     log_df['binary'] = params['binary']
#     log_df['balanced'] = params['balanced']
#     log_df['split'] = str(params['split'])
#     log_df['target_map'] = str(target_map)
#     log_df['colab'] = colab
#     log_df['model_path'] = trained_model_path
#     log_df['samples_per_s'] = log_history[0]['train_samples_per_second']
#     log_df['steps_per_s'] = log_history[0]['train_steps_per_second']
#     log_df['per_device_train_batch_size'] = params['per_device_train_batch_size']
#     log_df['per_device_eval_batch_size'] = params['per_device_eval_batch_size']
#
#     float_cols = log_df.select_dtypes(include='float64')
#     log_df[float_cols.columns] = float_cols.apply(lambda x: round(x, 3))
#
#     log_path = 'output/training_logs.csv'
#     log_df.to_csv(log_path, mode='a', header=not os.path.isfile(log_path), index=False)

def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}

class WeightedCELossTrainer(Trainer):
    def add_weights(self, weights):
        self.weights = weights
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights, device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def finetune(ds, params, target_map, colab):
    """
    Fine-tunes a pre-trained language model for text classification. Handles tokenization, model loading, and training.

    Args:
        params (dict): A dictionary containing model and training parameters.
        ds (DatasetDict): A Hugging Face DatasetDict containing 'train', 'test', and optionally 'validate' splits.
        target_map (dict): A mapping of original labels to numerical indices.
        save_logs (bool): If True, saves training logs and metrics.
        reversed_target_map:
    """
    params_passed = {k: params[k] for k in params if k not in ['model_name',
                                                               'split',
                                                               'binary',
                                                               'balanced']}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    trained_model_path = f"output/models/{params['model_name']}_{timestamp}"

    model, tokenizer, tokenized_datasets = init_model(params['model_name'], ds, target_map, True, True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")



    training_args = TrainingArguments(
        output_dir=f'{trained_model_path}/checkpoints',
        lr_scheduler_type= "constant",
        warmup_ratio= 0.1,
        max_grad_norm= 0.3,
        weight_decay=0.001,
        evaluation_strategy="epoch",
        report_to="wandb",
        run_name=params['model_name'],
        fp16=False,
        use_cpu=False,
        gradient_checkpointing=True,
        **params_passed
    )

    weights = [len(ds['train'].to_pandas()) / (2 * i) for i in ds['train'].to_pandas().label.value_counts()]


    trainer = WeightedCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.add_weights(weights)

    trainer.train()
    wandb.finish()


def init_model(model_checkpoint, ds, target_map, bnb=None, peft=None):

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

    add_pad_token = True if tokenizer.pad_token is None else False
    if add_pad_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    def token_preprocessing_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)

    # Apply the preprocessing function and remove the undesired columns
    tokenized_datasets = ds.map(token_preprocessing_function, batched=True)

    # Set to torch format
    tokenized_datasets.set_format("torch")

    # # Change labels
    #
    # config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # config.vocab_size = tokenizer.vocab_size
    # config.id2label = {v: k for k, v in target_map.items()}
    # config.label2id = target_map


    if bnb is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   num_labels=2,
                                                                   # config=config,
                                                                   ignore_mismatched_sizes=True,
                                                                   trust_remote_code=True,
                                                                   device_map='auto',
                                                                   # quantization_config=bnb_config,
                                                                   )
    else:
        compute_dtype = getattr(torch, "float16")
        print(compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   num_labels=2,
                                                                   # config=config,
                                                                   ignore_mismatched_sizes=True,
                                                                   trust_remote_code=True,
                                                                   device_map='auto',
                                                                   quantization_config=bnb_config,
                                                                   )



    if peft is not None:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=2,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            # target_modules='all-linear'
            # target_modules=[
            #     "q_proj",
            #     "v_proj",
            # ]
        )

        model = prepare_model_for_kbit_training(model)
        print('Model prepared')
        model = get_peft_model(model, peft_config)
        print('Model perfed')
        model.print_trainable_parameters()

    model = model.cuda()
    print(model.device)
    print('Model to cuda')

    # model.config.use_cache = False
    # model.config.pretraining_tp = 1

    if add_pad_token:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer, tokenized_datasets


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

#%%
