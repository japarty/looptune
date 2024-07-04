import torch

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, BitsAndBytesConfig


def init_model(model_checkpoint, ds, target_map, bnb_config=False, peft_config=False, custom_loader=False):
    loader = AutoModelForSequenceClassification if custom_loader == False else custom_loader

    cuda_flag = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

    add_pad_token = True if tokenizer.pad_token is None else False
    if add_pad_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    def token_preprocessing_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=256)

    # Apply the preprocessing function and remove the undesired columns
    tokenized_datasets = ds.map(token_preprocessing_function, batched=True)

    # Set to torch format
    tokenized_datasets.set_format("torch")

    # Change labels
    config = AutoConfig.from_pretrained(model_checkpoint, trust_remote_code=True)
    # config.vocab_size = tokenizer.vocab_size
    config.id2label = {v: k for k, v in target_map.items()}
    config.label2id = target_map
    if not bnb_config:
        model = loader.from_pretrained(model_checkpoint,
                                                                   # num_labels=2,
                                                                   config=config,
                                                                   ignore_mismatched_sizes=True,
                                                                   trust_remote_code=True,
                                                                   # device_map='auto',
                                                                   # quantization_config=bnb_config,
                                                                   )
    else:
        bnb_config = BitsAndBytesConfig(**bnb_config)
        model = loader.from_pretrained(model_checkpoint,
                                                                   # num_labels=2,
                                                                   config=config,
                                                                   ignore_mismatched_sizes=True,
                                                                   trust_remote_code=True,
                                                                   device_map='auto',
                                                                   quantization_config=bnb_config,
                                                                   )

    model.tie_weights()
    if peft_config:
        peft_config = LoraConfig(**peft_config)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if cuda_flag:
        model = model.cuda()

    # model.config.use_cache = False
    # model.config.pretraining_tp = 1

    if add_pad_token:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer, tokenized_datasets

