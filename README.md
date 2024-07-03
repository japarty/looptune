# looptune
This package is prepared as a wrapper for a trainer finetuning pipeline, to simnplify proces of iterated finetuning even more.

# How to use:
1. Activate your target env
2. install via `pip install -e <path to looptune repo main dir>/.` (for example, if running from a notebook in the `notebooks` dir that would be `../.`)
3. Install torch 
## Notebooks
1. finetune_example - simple finetuning of model for multilabel emotion classification with logging to wandb and peft support

## Notes
For now it's limited to sequence classification (or at least it was the only one tested). Ultimately, I want to provide it as a dockerized webapp.
## Issues
- not all models support 'device_map='auto'/are peftable
- 