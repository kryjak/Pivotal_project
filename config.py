import os
import torch as t
from attrdict import AttrDict

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
PATH_TO_REPOS = '/home/jkrys/gitrepos' # path to the directory containing the git repositories
PATH_TO_DATASETS = os.path.join(os.getcwd(), 'data_storage', 'datasets')

# Allowed models: 'DeepSeek-VL', 'LLaVa'
MODEL = 'DeepSeek-VL'

WANDB_PROJECT = "pivotal_adv_attacks"
WANDB_NAME = None
WANDB_KEY = os.getenv('WANDB_KEY')

HF_TOKEN = os.getenv('HF_TOKEN')


# different models have different tokenizers, so we can't use the same target across models
# that is, unless we find a group of n tokens which also constitute tokens in a different tokenizer
attack_config_DeepSeek = AttrDict(
    model_name = 'DeepSeek'
    # ,n_epochs = 4
    ,n_epochs = 250
    ,n_logs = 20
    ,eps = 8/255
    ,single_token_target = ["dog"]
    # ,multi_token_target = ['вки', 'deven', '的主要', 'ordin', ' kayaking', '网约车', ' indeed', '崎', '私家', ' startTime'] # random 10-token target
    ,multi_token_target = ['вки', 'deven', '的主要', 'ordin', ' kayaking'] # random 5-token target
    ,optimizer = t.optim.AdamW
    ,lr = 1e-1
    ,weight_decay=1e-2
)

attack_config_Llava = AttrDict(
    model_name = 'LLaVa'
    ,n_epochs = 250
    ,n_logs = 20
    ,eps = 8/255
    ,single_token_target = ["dog"]
    # ,multi_token_target = ['plots', 'authentic', 'отри', 'Im', 'Совет', '+=', 'civ', 'URLs', 'astero', 'éx'] # random 10-token target
    ,multi_token_target = ['plots', 'authentic', 'отри', 'Im', 'Совет'] # random 5-token target
    ,optimizer = t.optim.AdamW
    ,lr = 1e-1
    ,weight_decay=1e-2
)

wandb_config = AttrDict(
	wandb_project = WANDB_PROJECT
	,wandb_name = WANDB_NAME
)


jailbreak_config = AttrDict(
    model_name = 'DeepSeek'
    # ,n_epochs = 250
    ,n_epochs = 20
    ,n_logs = 10
    ,eps = 8/255
    # ,multi_token_target = ['plots', 'authentic', 'отри', 'Im', 'Совет', '+=', 'civ', 'URLs', 'astero', 'éx'] # random 10-token target
    ,jailbreak_target = ['plots', 'authentic', 'отри'] # random 3-token target
    ,optimizer = t.optim.AdamW
    ,lr = 1e-2
    ,weight_decay=1.0
    ,image_size = model.config.vision_config.params.image_size
)