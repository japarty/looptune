# EQILLM
Project focusing on LLM finetuning analysis for emotion recognition. This project focuses on finetuning models in order to achieve quality of responses regarding pathos recognition observe in gpt-4. Additionally, I want to trial-and-error overall pipeline for interacting and improving LLM's.

## Backend
Due to simplicity for now LM Studio is used for running server simulating OpenAI-aike API. For local use it's enough, and provides simplicity of running multiple models quantificatified with different methods




## Issues

- unresolved:
  - RuntimeError: CUDA error: device-side assert triggered: occurs after adding `tokenizer.add_special_tokens({'pad_token': '[PAD]'})` as was proposed in response under models which require pad token (`tokenizer.pad_token = tokenizer.eos_token` seems to not work for some of them)