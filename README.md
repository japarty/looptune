# EQILLM
Project focusing on LLM finetuning analysis for emotion recognition. This project focuses on finetuning models in order to achieve quality of responses regarding pathos recognition observe in gpt-4. Additionally, I want to trial-and-error overall pipeline for interacting and improving LLM's.

## Notes
For now, notebooks are prepared in such way to ease my jumping between local development and using Google Colab. For similar experience, you may keep repository copy synced on Google Drive


## Issues

- unresolved:
  - RuntimeError: CUDA error: device-side assert triggered: occurs after adding `tokenizer.add_special_tokens({'pad_token': '[PAD]'})` as was proposed in response under models which require pad token (`tokenizer.pad_token = tokenizer.eos_token` seems to not work for some of them)