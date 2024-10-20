import os
import torch as t
from utils import DotDict

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
PATH_TO_REPOS = os.path.split(os.getcwd())[0] # path to the directory containing the git repositories
PATH_TO_DATASETS = os.path.join(os.getcwd(), 'data_storage', 'datasets')
PATH_TO_TENSORS = os.path.join(os.getcwd(), 'data_storage', 'tensors')
# Allowed models: 'DeepSeek-VL', 'LLaVa'
MODEL = 'DeepSeek-VL'

WANDB_PROJECT = "pivotal_adv_attacks"
WANDB_NAME = None
WANDB_KEY = os.getenv('WANDB_KEY')

HF_TOKEN = os.getenv('HF_TOKEN')

attack_config_DeepSeek = DotDict(
    model_name = 'DeepSeek',
    n_epochs = 250,
    n_logs = 20,
    eps = 8/255,
    single_token_target = ["dog"],
    multi_token_target = ['вки', 'deven', '的主要', 'ordin', ' kayaking'],
    optimizer = t.optim.AdamW,
    lr = 1e-1,
    weight_decay=1e-2
)

attack_config_Llava = DotDict(
    model_name = 'LLaVa',
    n_epochs = 250,
    n_logs = 20,
    eps = 8/255,
    single_token_target = ["dog"],
    multi_token_target = ['plots', 'authentic', 'отри', 'Im', 'Совет'],
    optimizer = t.optim.AdamW,
    lr = 1e-1,
    weight_decay=1e-2
)

wandb_config = DotDict(
    wandb_project = WANDB_PROJECT,
    wandb_name = WANDB_NAME
)

jailbreak_config = DotDict(
    model_name = 'DeepSeek',
    n_epochs = 20,
    n_logs = 10,
    eps = 8/255,
    # ,multi_token_target = ['plots', 'authentic', 'отри', 'Im', 'Совет', '+=', 'civ', 'URLs', 'astero', 'éx'] # random 10-token target
    jailbreak_target = ['plots', 'authentic', 'отри'],
    optimizer = t.optim.AdamW,
    lr = 1e-2,
    weight_decay=1.0,
)
