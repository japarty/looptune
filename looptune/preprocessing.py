import pandas as pd
import itertools
import copy
import random

from datasets import concatenate_datasets
from collections import defaultdict
from datasets import Dataset, DatasetDict



# Dataset preaparation


def df_to_ds(df):
    print(df.groupby('label').count())
    ds = Dataset.from_pandas(df)
    ds = ds.class_encode_column('label')
    target_map = {i: ds.features["label"].str2int(i) for i in ds.features["label"].names}
    return ds, target_map


def reduce_labels(df, convert_dict):
        df['label'] = df['label'].apply(
            lambda x: to_binary_classification(x, convert_dict))
        return df

def to_binary_classification(x, convert_dict):
    """
    Converts labels to binary classification ('Pathos' or 'No_pathos').

    Args:
        x (str): The original label.
        convert_dict (dict, optional): A dictionary mapping original labels
            to their corresponding binary representation.

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


# def split_dataset(df, class_column):
#     min_samples = df[class_column].value_counts().min()
#     balanced_df = pd.DataFrame()
#     for cls in df[class_column].unique():
#         cls_samples = df[df[class_column] == cls].sample(n=min_samples)
#         balanced_df = pd.concat([balanced_df, cls_samples])
#     remaining_df = df.drop(balanced_df.index)
#     return balanced_df, remaining_df


def ratio_split_tuple(split):
    split_s = sum([i for i in split if isinstance(i, (int, float))])
    new_split = tuple([i / split_s if isinstance(i, (int, float)) else i for i in split])
    return new_split


def split_ds(dataset, train_size=0.8, val_size=None):
    dataset = dataset.train_test_split(train_size=train_size, seed=42, stratify_by_column='label')

    if val_size is not None:
        val_ratio = 1 - (val_size / (1 - train_size))
        dataset2 = dataset['test'].train_test_split(train_size=val_ratio, seed=42)

        dataset['test'] = dataset2['train']
        dataset['val'] = dataset2['test']

    return dataset


def balance_dataset(dataset, col_from, col_to=False):
    random.seed(42)

    samples_by_class = defaultdict(list)
    for sample in dataset[col_from]:
        samples_by_class[sample["label"]].append(sample)

    min_count = min(len(samples) for samples in samples_by_class.values())

    balanced_dataset = []
    leftovers = []
    for samples in samples_by_class.values():
        balanced_dataset.extend(samples[:min_count])
        leftovers.extend(samples[min_count:])

    random.shuffle(balanced_dataset)
    random.shuffle(leftovers)

    balanced_dataset = Dataset.from_list(balanced_dataset)
    balanced_dataset = balanced_dataset.cast_column('label', dataset[col_from].features['label'])
    dataset[col_from] = balanced_dataset

    if col_to != False:
        leftovers_ds = Dataset.from_list(leftovers)
        leftovers_ds = leftovers_ds.cast_column('label', dataset[col_to].features['label'])
        dataset[col_to] = concatenate_datasets([dataset[col_to], leftovers_ds])

    return dataset


# Create run configuration dicts

def prep_config_combinations(param_dict):
    for key in param_dict.keys():
        if isinstance(param_dict[key], dict):
            param_dict[key] = prep_config_combinations(param_dict[key])
    param_dict = {i: [q] if type(q) is not list else q for (i, q) in param_dict.items()}
    keys = list(param_dict.keys())
    combinations = list(itertools.product(*param_dict.values()))

    result = [{keys[i]: combination[i] for i in range(len(keys))} for combination in combinations]
    result = [copy.deepcopy(combination) for combination in result]
    return result