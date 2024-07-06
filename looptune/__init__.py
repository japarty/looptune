import wandb

from .variable_manipulation import *
from .custom_datasets import *
from .finetune import *
from .model import *
from .optional_utils import *
from .preprocessing import *
from .prompting import *
from .validate import *
from .logging import *

import warnings


def single_run(run_params, df, to_return=None):
    wandb_log = False
    try:
        ds, target_map = df_to_ds(df)
        ds = split_ds(ds, train_size=run_params['split'][0])

        if 'balanced' in run_params:
            if run_params['balanced']:
                if all([isinstance(type(i), str) for i in run_params['balanced']]):
                    ds = balance_dataset(ds, *run_params['balanced'])
                else:
                    for i in run_params['balanced']:
                        if isinstance(i, str):
                            ds = balance_dataset(ds, i)
                        else:
                            ds = balance_dataset(ds, *i)

        warnings.filterwarnings('ignore')
        if 'report_to' in run_params:
            report_to = run_params['report_to']
            if isinstance(report_to, str):
                report_to = [report_to]
        else:
            report_to = "none"

        if wandb_log:
            print('wandb inited')
            wandb.init(name=run_params['model_name'], **run_params['wandb_init_params'])

        model, tokenizer, tokenized_datasets = init_model(
            run_params['model_name'],
            ds,
            target_map,
            run_params['bnb_config'] if 'bnb_config' in run_params else False,
            run_params['peft_config'] if 'peft_config' in run_params else False,
            run_params['custom_loader'] if 'custom_loader' in run_params else False,
        )

        trainer, predicted = finetune(model,
            tokenizer,
            tokenized_datasets,
            ds,
            target_map,
            run_params['training_arguments'] if 'training_arguments' in run_params else {},
            run_params['trainer'] if 'trainer' in run_params else {},
            report_to,
            run_params['log_memory'] if 'log_memory' in run_params else True,
            )

        warnings.filterwarnings('default')

        if to_return:
            return eval(f"{', '.join(to_return)}")

        # this runs if 'to_return' was not passed
        del model
        del trainer
        del tokenizer
        del tokenized_datasets
        clean_memory()
        wandb.finish()

    except Exception as exc:
        print(exc)

        if wandb_log:
            wandb.log({'error': str(exc)})
            wandb.finish(1)
