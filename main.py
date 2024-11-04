import os

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
import wandb

from attacks import JailbreakAttack

# from data import load_dataset
from config import (
    DEVICE,
    MODEL,
    PATH_TO_DATASETS,
    WANDB_KEY,
    attack_config,
)
from custom_image_transforms import CustomTransforms
from models import load_model
from utils import open_image_from_url

wandb.login(key=WANDB_KEY)


## LOAD MODEL AND TOKENIZER
model, processor = load_model(MODEL)
# Put model in eval mode and switch off gradients:
model.eval()
if model.training:
    print("Model is in training mode")
else:
    print("Model is in eval mode")

# we're only interested in computing the gradients wrt the input images, not the internal parameters
for name, param in model.named_parameters():
    param.requires_grad = False

# model specific imports and configs
if MODEL == "DeepSeek-VL":
    from VLM_base_classes import DeepSeekVLBaseClass

    img_size = processor.image_processor.image_size

    base_model_class = DeepSeekVLBaseClass
elif MODEL == "LLaVa":
    from VLM_base_classes import LlavaBaseClass

    img_size = model.config.vision_config.image_size
    model.config.image_grid_pinpoints.append([img_size, img_size])
    print("LLaVa grid points: ", model.config.image_grid_pinpoints)

    base_model_class = LlavaBaseClass
else:
    raise NotImplementedError(f"Model {MODEL} not implemented yet.")

attack_config["image_size"] = img_size

# Instantiate the base class
# Crucially, this needs to be done *after* swtiching the internal gradients off.
# Otherwise, there will be a conflict with the optimiser, wihch only takes care of the perturbation mask
# And backpropagation will fail (on the second loss.backwards())
base_instance = base_model_class(model, processor)
print("Base instance created.")

# Define the processor so that we can pre-process the prompt+image into a format accepted by the model. This includes a normalisation step.
tokenizer = processor.tokenizer

processor_mean = t.tensor(processor.image_processor.image_mean).to(DEVICE)
processor_std = t.tensor(processor.image_processor.image_std).to(DEVICE)
print(f"{processor_mean=}")
print(f"{processor_std=}")

## LOAD IMAGE
img_url = "https://wp.inews.co.uk/wp-content/uploads/2023/03/SEI_149780351.jpg?crop=157px%2C0px%2C1537px%2C1537px&resize=640%2C640"
img_path = os.path.join(os.getcwd(), "test_image.jpg")
img = open_image_from_url(img_url)
# img_size = 100
img = img.resize((img_size, img_size))
img.save(img_path)

## ATTACKS
test_prompt = "What is shown in this image?"
test_target = "dog"

# # ## Single-token attack
# # In this attack, we are optimising using only a single target output token, e.g. 'dog'.
# print("Starting single-token attack.")
# single_token_attack = ControlSingleTokenAttack(
#     base_instance, attack_config, wandb_name="single_token"
# )
# init_image, delta, loss_train = single_token_attack.train_attack(test_prompt, img)

# # we can import saved data from wandb with:
# # api = wandb.Api()
# # wandb_run = api.run(run_path)
# # wandb_run.file('sample_data/' + 'init_image.pt' ).download(replace=True)
# # loaded_tensor = t.load("sample_data/init_image.pt")
# # assert t.equal(init_image, loaded_tensor) == True

# # The standard way to execute the attack is to suply the initial image and the perturbation `delta`. Both will be tensors: `image` is in the range $[0.0, 1.0]$, `delta` might spill outside of the range $[-1.0, 1.0]$ and is clamped within the function call.

# # 1. Use the default processing pipeline with autoregressive generation
# # max_new_tokens is set to 10
# output, answer = single_token_attack.execute_attack(
#     prompt=test_prompt,
#     image=init_image,
#     delta=delta,
#     max_new_tokens=10,
#     no_eos_token=True,
#     do_sample=True,
#     top_p=0.95,
#     top_k=20,
# )
# print(f"{answer=}")


# # 2. Alternatively, we can supply an adversarial image (inital + perturbation). This needs to be a tensor in the range $[0, 255.0]$ and the perturbation needs to be clamped manually.
# adversarial_image = (init_image + attack_config.eps * delta.clamp(-1, 1)).clamp(
#     0, 1
# ) * 255

# output_v2, answer_v2 = single_token_attack.execute_attack(
#     prompt=test_prompt,
#     adversarial_image=adversarial_image,
#     max_new_tokens=10,
#     no_eos_token=True,
#     do_sample=True,
#     top_p=0.95,
#     top_k=20,
# )
# print(f"{answer_v2=}")

# single_token_attack.finish_wandb_logging()
# print("Single-token attack finished.")

# # ## Multi-token attack
# # In this attack, we are optimising using multiple target output tokens, e.g. ['dog', 'on', 'a', 'bike']. For this reason, we introduce two training methods:
# # - 'autogregressive' -- using a prompt of length $n$, we generate the prediction for token in the $n+1$ position. This new token is appended to the original prompt, and now the new prompt of length $n+1$ is used to generate another token at position $n+2$. This proceeds recursively until we run out of target tokens.
# # - 'teacher_forcing' -- we append the actualy target tokens to the starting prompt, no matter if these were the actual tokens generated by the model or not. This method is supposed to smoothen out the loss curve by `keeping the model on track' despite prediction errors. The name ocmes from a scenario where a school teacher allows the student to view the correct answer in the first part of the test (even though the student was not able to solve this part), so that they can still attempt the next part.
# print("Starting multi-token attack.")
# multi_token_attack = ControlMultipleTokensAttack(
#     base_instance, attack_config, wandb_name="autoregressive"
# )

# multi_token_target=["вки", "deven", "的主要", "ordin", " kayaking"] # for DeepSeek
# multi_token_target=["plots", "authentic", "отри", "Im", "Совет"] # for LLaVa
# # # init_image, delta, loss_train = multi_token_attack.train_attack(prompt, img, training_method='teacher_forcing')
# init_image, delta, loss_train = multi_token_attack.train_attack(
#     test_prompt, img, multi_token_target, training_method="autoregressive"
# )

# # There are also two generation methods:
# # - 'automatic' -- we use the model's autoregressive generation method.
# # - 'manual' -- we manually generate the next token in the sequence using the model's logits.
# logits, answer_auto = multi_token_attack.execute_attack(
#     test_prompt,
#     init_image,
#     delta,
#     generation_method="automatic",
#     max_new_tokens=5,
#     no_eos_token=True,
#     do_sample=True,
#     top_p=0.95,
#     top_k=20,
# )
# print(f"{answer_auto=}")
# logits, answer_manual = multi_token_attack.execute_attack(
#     test_prompt,
#     init_image,
#     delta,
#     generation_method="manual",
#     max_new_tokens=5,
#     no_eos_token=True,
#     do_sample=True,
#     top_p=0.95,
#     top_k=20,
# )
# print(f"{answer_manual=}")

# plt.plot(loss_train)
# plt.show()

# multi_token_attack.finish_wandb()
# print("Multi-token attack finished.")

# ## Jailbreaking
print("Starting jailbreak attack.")
# for loading data from a previous run, supply the run name in the following format:
# wandb_run_name = "kryjak-None/pivotal_adv_attacks/nepwz08b"
# wandb_run_id = wandb_run_name.split("/")[-1]
# load_data_from_wandb(wandb_run_name)

# wandb_api = wandb.Api()
# run = wandb_api.run(wandb_run_name)
# print(run.summary)
# print(run.config)
# history = run.history()
# print(history.columns)

# artifact = wandb_api.artifact("kryjak-None/pivotal_adv_attacks/jailbreak_data:v3")
# artifact.download(path_prefix="jailbreak_completions")
# test_table = artifact.get("jailbreak_completions")
# df = pd.DataFrame(data=test_table.data, columns=test_table.columns)

df_train = pd.read_csv(
    os.path.join(PATH_TO_DATASETS, "advbench_mini_train.csv"), index_col=0
)
df_test = pd.read_csv(
    os.path.join(PATH_TO_DATASETS, "advbench_mini_test.csv"), index_col=0
)
df_test.head()

# prompts
train_goals = df_train["goal"].to_list()
test_goals = df_test["goal"].to_list()
train_goal_single = train_goals[:1]
test_goal_single = test_goals[:1]
# targets
train_targets = df_train["target"].to_list()
test_targets = df_test["target"].to_list()
# targets need to be tokenized: [[str1, str2,...], [str1, str2,...], ...]
train_targets_tokenized = [
    [
        tokenizer.decode(token)
        for token in tokenizer.encode(target, add_special_tokens=False)
    ]
    for target in train_targets
]
test_targets_tokenized = [
    [
        tokenizer.decode(token)
        for token in tokenizer.encode(target, add_special_tokens=False)
    ]
    for target in test_targets
]
train_target_tokenized_single = train_targets_tokenized[:1]
test_target_tokenized_single = test_targets_tokenized[:1]

jailbreak_attack = JailbreakAttack(
    base_instance, jailbreak_config + wandb_config, wandb_name="test"
)

delta, loss_train = jailbreak_attack.train(
    prompts=train_goal_single,
    #   prompts=train_goals,
    images=[img],
    targets=train_target_tokenized_single,
    #   targets=train_targets_tokenized,
    training_method="teacher_forcing",
    use_cache=False,
    batch_size=1,
)

max_memory_used = t.cuda.max_memory_allocated() / 1024**3
wandb.log({"max_memory_used": max_memory_used})
print(f"Max memory used during the run: {max_memory_used} GB")
t.cuda.reset_peak_memory_stats()

_, answer_auto_pil = jailbreak_attack.execute(
    prompt=test_goals[0],
    image=None,
    delta=delta,
    generation_method="automatic_with_pil",
    max_new_tokens=7,
    no_eos_token=False,
    do_sample=True,
    top_p=0.95,
    top_k=20,
)
_, answer_auto = jailbreak_attack.execute(
    prompt=test_goals[0],
    image=None,
    delta=delta,
    generation_method="automatic",
    max_new_tokens=7,
    no_eos_token=False,
    do_sample=True,
    top_p=0.95,
    top_k=20,
)
_, answer_manual = jailbreak_attack.execute(
    prompt=test_goals[0],
    image=None,
    delta=delta,
    generation_method="manual",
    max_new_tokens=7,
    no_eos_token=True,
    use_cache=False,
    do_sample=True,
    top_p=0.95,
    top_k=20,
)

print("Results without clamping:")
print(f"{answer_auto_pil=}")
print("----------------------")
print(f"{answer_auto=}")
print("----------------------")
print(f"{answer_manual=}")

_, answer_auto_pil = jailbreak_attack.execute_with_clamp(
    prompt=test_goals[1],
    image=None,
    delta=delta,
    generation_method="automatic_with_pil",
    max_new_tokens=500,
    no_eos_token=True,
    do_sample=True,
    top_k=2,
    # top_k=50,
    # top_p=0.95,
    no_repeat_ngram_size=6,
)

_, answer_auto = jailbreak_attack.execute_with_clamp(
    prompt=test_goals[1],
    image=None,
    delta=delta,
    generation_method="automatic",
    max_new_tokens=500,
    no_eos_token=True,
    do_sample=True,
    top_k=2,
    no_repeat_ngram_size=6,
)

print("Results with clamping:")
print(f"{answer_auto_pil=}")
print("----------------------")
print(f"{answer_auto=}")
print("----------------------")
jailbreak_attack.finish_wandb()
print("Jailbreak attack finished.")


# ## Augmentation attack
print("Starting augmentation attack.")
augmentations = {
    "noise_strength": 1e-1,
    "max_jitter_ratio": 0.01,
    "contrast_range": (0.9, 1.1),
    "sharpness_factor": "random",
    "grayscale_prob": 0.1,
    "seed": 42,
}

transform = CustomTransforms(**augmentations)

augmentation_attack = JailbreakAttack(
    base_instance, attack_config, wandb_name="augmentations"
)
delta, loss_train = augmentation_attack.train(
    prompts=train_goal_single,
    images=[img],
    targets=train_target_tokenized_single,
    training_method="teacher_forcing",
    use_cache=False,
    batch_size=1,
    augmentations=augmentations,
)
# augmentation_attack.test_dataset(delta=delta, df_test=df_test, max_new_tokens=500, no_eos_token=False, use_cache=False, do_sample=True, top_k=2, no_repeat_ngram_size=6)
# augmentation_attack.eval_dataset(max_new_tokens=100, no_eos_token=True, use_cache=False, do_sample=True, top_k=2, no_repeat_ngram_size=6)
plt.plot(loss_train)
plt.show()

max_memory_used = t.cuda.max_memory_allocated() / 1024**3
wandb.log({"max_memory_used": max_memory_used})
print(f"Max memory used during the run: {max_memory_used} GB")
wandb.log({"augmentations": augmentations})
augmentation_attack.finish_wandb()
print("Augmentation attack finished.")
