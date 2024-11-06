import os
import torch as t
import matplotlib.pyplot as plt
from torchvision import transforms

from config import (
    DEVICE,
    MODEL,
    PATH_TO_TENSORS,
    attack_config,
)

from attacks import JailbreakAttack
from models import load_model
from utils import open_image_from_url


model, processor = load_model(MODEL)
tokenizer = processor.tokenizer
processor_mean = t.tensor(processor.image_processor.image_mean).to(DEVICE)
processor_std = t.tensor(processor.image_processor.image_std).to(DEVICE)
print(f"{processor_mean=}")
print(f"{processor_std=}")

model.eval()
if model.training:
    print("Model is in training mode")
else:
    print("Model is in eval mode")

# we're only interested in computing the gradients wrt the input images, not the internal parameters
for name, param in model.named_parameters():
    param.requires_grad = False

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

base_instance = base_model_class(model, processor)
print("Base instance created.")

## LOAD IMAGE
img_url = "https://wp.inews.co.uk/wp-content/uploads/2023/03/SEI_149780351.jpg?crop=157px%2C0px%2C1537px%2C1537px&resize=640%2C640"
img_path = os.path.join(os.getcwd(), "test_image.jpg")
img = open_image_from_url(img_url)
# img_size = 100
img = img.resize((img_size, img_size))
img.save(img_path)

img_gray = 0.5 * t.ones((3, img_size, img_size)).to(DEVICE)
img_gray = transforms.ToPILImage()(img_gray)

## ATTACKS
test_prompt = ["What is shown in this image?"]
test_target = [["sun"]]

augmentations = {
    "contrast_range": (0.7, 1.5),
    "max_jitter_ratio": 0.1,
    "color_amount": 0.7,  # has to be between 0 and 1
    # "down_res": 224,
    # "up_res": img_size,
    # "down_noise": 0.05, # recommended <0.1
    # "up_noise": 0.0,
    "seed": 42,
}

# augmentations = {}

# multi_token_attack = ControlMultipleTokensAttack(
#     base_instance, attack_config, wandb_name="autoregressive"
# )

# multi_token_target=["plots", "authentic"] # for LLaVa
# # # init_image, delta, loss_train = multi_token_attack.train_attack(prompt, img, training_method='teacher_forcing')
# init_image, delta, loss_train = multi_token_attack.train_attack(
#     test_prompt[0], img, multi_token_target, training_method="autoregressive"
# )

augmentation_attack = JailbreakAttack(base_instance, attack_config, wandb_logging=True)

delta, loss_train = augmentation_attack.train(
    prompts=test_prompt,
    # images=[img_gray],
    images=None,
    targets=test_target,
    training_method="teacher_forcing",
    use_cache=False,
    batch_size=1,
    early_stop_loss=0.1,
    augmentations=augmentations,
)

plt.plot(loss_train)
plt.show()

# save the delta
t.save(delta, os.path.join(PATH_TO_TENSORS, "delta.pt"))

# display the delta
delta_pil = transforms.ToPILImage()(delta.clamp(0, 1).detach().cpu().float())
delta_pil.show()

# display the transformed image
transformed_image = (transforms.ToTensor()(img_gray).to(delta.device) + attack_config.eps * delta.clamp(-1, 1))
t.save(transformed_image, os.path.join(PATH_TO_TENSORS, f"transformed_image.pt"))
transformed_image_pil = transforms.ToPILImage()(transformed_image.clamp(0, 1).detach().cpu().numpy().transpose(1,2,0))
transformed_image_pil.show()
transformed_image_pil.save('transformed_image.png')
