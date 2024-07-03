import wandb

from .variable_manipulation import *
from .custom_datasets import *
from .finetune import *
from .model import *
from .optional_utils import *
from .preprocessing import *
# from .prompting import *
from .validate import *
from .logging import *

def single_run(run_params, df, to_return=None, clean_at_end=True):
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


        if 'report_to' in run_params:
                report_to = run_params['report_to']
                if isinstance(report_to, str):
                    report_to = [report_to]
        else:
            report_to = False

        if wandb_log:
            wandb.init(name=run_params['model_name'], **run_params['wandb_init_params'])

        bnb_config = run_params['bnb_config'] if 'bnb_config' in run_params else False
        peft_config = run_params['peft_config'] if 'peft_config' in run_params else False

        model, tokenizer, tokenized_datasets = init_model(run_params['model_name'],
                                                          ds,
                                                          target_map,
                                                          bnb_config,
                                                          peft_config)
        trainer, predicted = finetune(model,
                           tokenizer,
                           tokenized_datasets,
                           ds,
                           run_params['training_arguments'],
                           target_map,
                           report_to)

        if to_return:
            return eval(f"{', '. join(to_return)}")

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





