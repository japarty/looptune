import os
import datetime

import evaluate
import numpy as np
import pandas as pd
import seaborn as sn
import itertools
import torch
import wandb

from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments, DataCollatorWithPadding, AutoTokenizer, pipeline, BitsAndBytesConfig
from transformers.modelcard import parse_log_history
from yeelight import Bulb
from accelerate import init_empty_weights

MAX_LEN = 256

# Prepare functions

def to_binary_classification(x, convert_dict={"*": 'Pathos', 'No_pathos': 'No_pathos'}):
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
        if '*' in convert_dict.keys():
            return convert_dict['*']
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

def param_combinations_recursive(param_dict):
    for key in param_dict.keys():
        if isinstance(param_dict[key], dict):
            param_dict[key] = param_combinations_recursive(param_dict[key])

    param_dict={i:[q] if type(q) is not list else q for (i,q) in param_dict.items()}
    keys = list(param_dict.keys())
    combinations = list(itertools.product(*param_dict.values()))
    result = [{keys[i]: combination[i] for i in range(len(keys))} for combination in combinations]

    return result

# # From csv to ds

# # # Dataset specific load to 2-col df ['text', label']

def load_predefined_dataset(data_path, binarize):
    if data_path.endswith('polish_pathos_translated.xlsx'):
        return load_polish_pathos_translated(data_path, binarize)
    elif data_path.endswith('PolarIs-Pathos.xlsx'):
        return load_PolarIs(data_path, binarize)

def load_PolarIs(path, binarize=False):
    df = pd.read_excel(path)
    df['label'] = df[['No_pathos', 'Positive', 'Negative']].idxmax(axis=1)
    df = df.rename(columns={'Sentence': 'text'})
    df = df[['text', 'label']]

    if binarize:
        df['label'] = df['label'].apply(lambda x: to_binary_classification(x, {"*": 'Pathos', 'No_pathos': 'No_pathos'}))
    return df

def load_polish_pathos_translated(data_path, binarize=False):
    df = pd.read_excel(data_path)
    df['text'] = df['English']
    df['label'] = df['cleaned_pathos']
    df = df[['text', 'label']]

    if binarize:
        df['label'] = df['label'].apply(lambda x: to_binary_classification(x, {"*": 'Pathos', 'no pathos': 'No_pathos'}))
            
    return df

def encode_labels(dataframe):
    encoder = LabelEncoder()
    dataframe['label'] = encoder.fit_transform(dataframe['label'])
    target_map = dict(zip(encoder.classes_, map(int,encoder.transform(encoder.classes_))))

    return dataframe, target_map


def split_ds(dataset, train_size=0.8, val_size=None):
    dataset = dataset.train_test_split(train_size=train_size, seed=42)
    if val_size is not None:
        val_ratio = 1 - (val_size/(1 - train_size))
        dataset2 = dataset['test'].train_test_split(train_size=val_ratio, seed=42)

        dataset['test'] = dataset2['train']
        dataset['val'] = dataset2['test']
    return dataset




def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted', zero_division=0)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')["f1"]
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


def finetune(model, tokenizer, tokenized_datasets, ds, params, target_map):
    """
    Fine-tunes a pre-trained language model for text classification. Handles tokenization, model loading, and training.

    """
    cuda_flag = torch.cuda.is_available()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    trained_model_path = f"output/models/{model.config._name_or_path}_{timestamp}"
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    training_args = TrainingArguments(
        output_dir=f'{trained_model_path}/checkpoints',
        report_to="wandb",
        run_name=model.config._name_or_path,
        **params
    )

    trainer = WeightedCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    weights = [len(ds['train'].to_pandas()) / (len(ds['train'].to_pandas().label.value_counts()) * i) 
              for i in ds['train'].to_pandas().label.value_counts()]
    
    trainer.add_weights(weights)


    if cuda_flag:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
    
        wandb.log({'pre_gpu': {'model':gpu_stats.name, 'max_memory': max_memory, 'memory_reserved':start_gpu_memory}})

    trainer_stats = trainer.train()

    if cuda_flag:
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_training = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory         /max_memory*100, 3)
        lora_percentage = round(used_memory_for_training/max_memory*100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
        wandb.log({'post_gpu': {'peak_memory': used_memory, 'training_memory':used_memory_for_training}})
        
    predicted = trainer.predict(tokenized_datasets['test'])
    predicted_labels = [int(i.argmax()) for i in predicted[0]]
    true_labels = ds['test']['label']
    wandb.log({"cm_test" : wandb.plot.confusion_matrix(probs=None,
                                                    y_true=predicted_labels, 
                                                    preds=true_labels, class_names=list(target_map.keys()))
                                                                    })

    
    wandb.finish()

    return trainer

    # del model, tokenizer, tokenized_datasets, data_collator, trainer
    # import gc
    # torch.cuda.empty_cache()
    # gc.collect()
    
    # return model, trainer, tokenized_datasets
    


def init_model(model_checkpoint, ds, target_map, bnb_config=False, peft_config=False):
    cuda_flag = torch.cuda.is_available()
    
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
    
    # Change labels
    config = AutoConfig.from_pretrained(model_checkpoint, trust_remote_code=True)
    # config.vocab_size = tokenizer.vocab_size
    config.id2label = {v: k for k, v in target_map.items()}
    config.label2id = target_map

    if bnb_config == False:
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   # num_labels=2,
                                                                   config=config,
                                                                   ignore_mismatched_sizes=True,
                                                                   trust_remote_code=True,
                                                                   # device_map='auto',
                                                                   # quantization_config=bnb_config,
                                                                   )
    else:
        
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   # num_labels=2,
                                                                   config=config,
                                                                   ignore_mismatched_sizes=True,
                                                                   trust_remote_code=True,
                                                                   device_map='auto',
                                                                   quantization_config=bnb_config,
                                                                   )

    model.tie_weights()

    if peft_config != None and peft_config != False:
        model = prepare_model_for_kbit_training(model)
        print('Model prepared')
        model = get_peft_model(model, peft_config)
        print('Model perfed')
        model.print_trainable_parameters()

    if cuda_flag:
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

# def get_log_for_val(checkpoint_path, logs_path, sort_col='Step'):
#     """
#     Retrieves the training log entry corresponding to a given checkpoint.

#     Args:
#         checkpoint_path (str): The path to the checkpoint directory.
#         logs_path (str): The path to the CSV file containing the training logs.
#         sort_col (str, optional): The column to sort by when searching for the latest checkpoint. Defaults to 'Step'.

#     Returns:
#         pd.Series: A Pandas Series representing a single row of the training logs.
#     """
#     training_logs = pd.read_csv(logs_path)
#     if 'checkpoint-' in checkpoint_path:
#         temp_path = checkpoint_path.rsplit('models/', 1)[-1]
#         model_path, checkpoint_num = temp_path.rsplit('/checkpoints/checkpoint-')
#         row = training_logs[(training_logs['model_path'].apply(lambda x: x.rsplit('models/', 1)[-1] == model_path)) & (
#                     training_logs['Step'] == int(checkpoint_num))]
#     else:
#         row = training_logs[training_logs['model_path'] == checkpoint_path].sort_values(sort_col, ascending=False).head(
#             1)
#     return row.iloc[0]


# def validate(row, ds):
#     """
#     Loads a trained model from a checkpoint and evaluates its performance on the validation set.

#     Args:
#         row (pd.Series): A single row from the training logs, containing checkpoint information.
#         ds (DatasetDict): A Hugging Face DatasetDict containing a 'validate' split.

#     Returns:
#         tuple: A tuple containing:
#             * predicted (list): List of predicted labels.
#             * val_labels (list): List of true labels.
#     """
#     val_sentences = ds['validate']['sentence']
#     val_labels = [reversed_target_map[i] for i in ds['validate']['label']]

#     classifier = pipeline('text-classification',
#                           model=os.path.join(row['model_path'], 'checkpoints', f"checkpoint-{row['Step']}"), device=0)
#     predicted = [i['label'] for i in classifier(val_sentences)]
#     return predicted, val_labels


# def val_metrics(predicted, val_labels, target_map):
#     """
#     Calculates and displays validation metrics (accuracy, F1-score, confusion matrix).

#     Args:
#         predicted (list): List of predicted labels.
#         val_labels (list): List of true labels.
#         target_map (dict): A mapping of original labels to numerical indices.
#     """
#     print("acc:", accuracy_score(val_labels, predicted))
#     print("f1:", f1_score(val_labels, predicted, average='macro'))

#     cm = confusion_matrix(val_labels, predicted, normalize='true')
#     plot_cm(cm, target_map)


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
