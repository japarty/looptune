import datetime
import evaluate
import numpy as np
import torch
import wandb

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from looptune.logging import get_cuda_memory


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

def finetune(model, tokenizer, tokenized_datasets, ds, params, target_map, log_memory=False):
    """
    Fine-tunes a pre-trained language model for text classification. Handles tokenization, model loading, and training.

    """
    cuda_flag = torch.cuda.is_available()
    if log_memory:
        if 'wanb' in log_memory and cuda_flag == False:
            print("Log memory set to True, but CUDA is unavailable. Setting to False")
            log_memory = False

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    trained_model_path = f"output/models/{model.config._name_or_path}_{timestamp}"

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    training_args = TrainingArguments(
        output_dir=f'{trained_model_path}/checkpoints',
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

    if log_memory:
        cuda_prestats = get_cuda_memory(0)
        start_gpu_memory = cuda_prestats['reserved']
        max_memory = cuda_prestats['total_memory']
        print(f"GPU = {cuda_prestats['name']}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
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
        wandb.log({'post_gpu': {'peak_memory': used_memory, 'training_memory': used_memory_for_training}})

    predicted = trainer.predict(tokenized_datasets['test'])
    predicted_labels = [int(i.argmax()) for i in predicted[0]]
    true_labels = ds['test']['label']
    wandb.log({"cm_test": wandb.plot.confusion_matrix(probs=None,
                                                      y_true=true_labels,
                                                      preds=predicted_labels, class_names=list(target_map.keys()))
               })

    # clean_memory()

    return trainer, predicted
