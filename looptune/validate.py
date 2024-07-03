import pandas as pd
import seaborn as sn


# Validate functions

def plot_cm(cm, target_map):
    classes = list(target_map.keys())
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    ax = sn.heatmap(df_cm, annot=True, fmt='.2g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")


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

# def encode_labels(dataframe):
#     encoder = LabelEncoder()
#     dataframe['label'] = encoder.fit_transform(dataframe['label'])
#     target_map = dict(zip(encoder.classes_, map(int,encoder.transform(encoder.classes_))))

#     return dataframe, target_map