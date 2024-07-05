# looptune
This package is prepared as a wrapper for a trainer finetuning pipeline, to simnplify proces of iterated finetuning even more.

# Installation:
1. Install repository: 
   - after downloading repository locally: `pip install -e <path to looptune repo main dir>/.` (for example, if running from a notebook in the `notebooks` dir that would be `../.`)
   - install from git `pip install git+https://github.com/japarty/looptune`
3. I you want to run with GPU (afaik only Nvidia is supported), install torch accordingly to: https://pytorch.org/

## Notebooks
1. finetune_example - simple finetuning of model for multilabel emotion classification, actively changed as it's also used to testing features to keep example up to date

## Notes
For now it's limited to sequence classification (or at least it was the only one tested). 

Ultimately, I want to provide it as a dockerized webapp.
## Issues
- not all models support 'device_map='auto'/are peftable. is_peftable() checking would be a TODO

- Using flash attentiom will need additional steps from the user. error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/ while `pip install flash-attn`, see https://github.com/huggingface/transformers/issues/30547