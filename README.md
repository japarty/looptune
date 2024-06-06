# EQILLM
Project focusing on LLM finetuning analysis for emotion recognition. This project focuses on finetuning models in order to achieve quality of responses regarding pathos recognition observe in gpt-4. Additionally, I want to trial-and-error overall pipeline for interacting and improving LLM's.

## Pipelines
1. Finetuning transformers via Trainer - uses Huggingface's solutions to finetune transformers. Consists of 3 notebooks:
- 1.1.finetune-trainer - prepare multiple configuration params and run fined-tuning in looped fashion. For now it's prepared specifically for the currently used dataset, but it will be later polished to provide simple repeatable finetuning with multiple configurations to ease model comparison process.   
- 1.2.plot-finetuning - plot finetuning metrics 
- 1.3.validate-finetuning -  validate models saved as checkpoints. Note: by default, 1.1 is intended to run multiple times before deciding on specific model, hence most of runs don't require saving.
2. Prompt engineering
- 2.1.prompting - communicate with OpenAI-like API (for example, hosted by tools like LM-studio). Prepared for running looped
- 2.2.validate-prompting - check metrics of saved responses, plot confusion matrix

## Notes
- Training Loss = "No Log" the default logging_steps in TrainingArguments is set to 500 steps, so no loss is reported before 500 steps
  https://discuss.huggingface.co/t/sending-a-dataset-or-datasetdict-to-a-gpu/17208
## Issues
- RuntimeError: CUDA error: device-side assert triggered: occurs after adding `tokenizer.add_special_tokens({'pad_token': '[PAD]'})` as was proposed in response under models which require pad token (`tokenizer.pad_token = tokenizer.eos_token` seems to not work for some of them)
- not all models support 'device_map='auto'