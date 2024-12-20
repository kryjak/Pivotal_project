import os
import torch as t
from utils import DotDict

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
PATH_TO_REPOS = os.path.split(os.getcwd())[
    0
]  # path to the directory containing the git repositories
PATH_TO_DATASETS = os.path.join(os.getcwd(), "data_storage", "datasets")
PATH_TO_TENSORS = os.path.join(os.getcwd(), "data_storage", "tensors")

"""
Allowed models: 'DeepSeek-VL', 'LLaVa', 'QwenVL'
"""
MODEL = "LLaVa"

WANDB_PROJECT = "pivotal_adv_attacks"
WANDB_NAME = None
WANDB_KEY = os.getenv("WANDB_API_KEY")

HF_TOKEN = os.getenv("HF_TOKEN")

attack_config = DotDict(
    n_epochs=5000,
    n_logs=100,
    eps=255 / 255,
    optimizer=t.optim.SGD,
    lr=1e-2,
    weight_decay=0,
)

wandb_config = DotDict(wandb_project=WANDB_PROJECT, wandb_name=WANDB_NAME)

attack_config = attack_config + wandb_config
