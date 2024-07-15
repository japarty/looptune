import datetime
import evaluate
import numpy as np
import torch
import wandb

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from looptune.logging import get_cuda_memory
from transformers import EarlyStoppingCallback

# Trainer and metrics
def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='macro', zero_division=0)[
        "precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average='macro')["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]

    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}


class WeightedCELossTrainer(Trainer):
    # customized from source: https://huggingface.co/blog/Lora-for-sequence-classification-with-Roberta-Llama-Mistral
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


# Run

def finetune(model, tokenizer, tokenized_datasets, ds, target_map, training_arguments={}, trainer_arguments={}, report_to="none", log_memory=True):
    """
    Fine-tunes a pre-trained language model for text classification. Handles tokenization, model loading, and training.

    """
    if 'callbacks' in trainer_arguments:
        if not isinstance(trainer_arguments['callbacks'], list):
            if isinstance(trainer_arguments['callbacks'], tuple):
                trainer_arguments['callbacks'] = list(trainer_arguments['callbacks'])
            else:
                trainer_arguments['callbacks'] = [trainer_arguments['callbacks']]

    if report_to != "none":
        if 'wandb' in report_to:
            arg_report_to = 'wandb'
    else:
        arg_report_to = "none"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    trained_model_path = f"output/models/{model.config._name_or_path}_{timestamp}"

    if 'output_dir' not in training_arguments:
        training_arguments['output_dir'] = trained_model_path


    cuda_flag = torch.cuda.is_available()
    if log_memory and not cuda_flag:
        print("Log memory set to True, but CUDA is unavailable. Setting to False")
        log_memory = False

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    training_args = TrainingArguments(
        run_name=model.config._name_or_path,
        report_to=arg_report_to,
        **training_arguments
    )

    trainer = WeightedCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        **trainer_arguments,

    )

    weights = [len(ds['train'].to_pandas()) / (len(ds['train'].to_pandas().label.value_counts()) * i)
               for i in ds['train'].to_pandas().label.value_counts()]

    trainer.add_weights(weights)

    report_to = [report_to] if not isinstance(report_to, list) else report_to

    if log_memory:
        cuda_prestats = get_cuda_memory(0)
        start_gpu_memory = cuda_prestats['reserved']
        max_memory = cuda_prestats['total_memory']
        print(f"GPU = {cuda_prestats['name']}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
        if 'wandb' in report_to:
            print('am in wand')
            wandb.log({'pre_gpu': {'model': cuda_prestats['name'],
                                   'max_memory': max_memory,
                                   'memory_reserved': start_gpu_memory}})

    trainer_stats = trainer.train()

    if log_memory:
        cuda_poststats = get_cuda_memory(0)
        used_memory = cuda_poststats['reserved']
        used_memory_for_training = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_training / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        if 'wandb' in report_to:
            wandb.log({'post_gpu': {'peak_memory': used_memory, 'training_memory': used_memory_for_training}})

    predicted = trainer.predict(tokenized_datasets['test'])
    predicted_labels = [int(i.argmax()) for i in predicted[0]]
    true_labels = ds['test']['label']
    if 'wandb' in report_to:
        wandb.log({"cm_test": wandb.plot.confusion_matrix(probs=None,
                                                          y_true=true_labels,
                                                          preds=predicted_labels, class_names=list(target_map.keys()))
                   })

    return trainer, predicted
