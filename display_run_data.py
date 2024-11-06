import torch as t
import wandb
from torchvision import transforms

from utils import open_image_from_url

# api = wandb.Api()

# Get a specific run
# run = api.run("run_path")
# summary = run.summary
# print(summary.keys())  # see what metrics are available

# # The max number of samples fetched by 'history' is 500
# history_df = run.history()
# for idx, row in history_df.iterrows():
#     print(f"Step {idx}:")
#     print(f"Loss: {row['train_loss']}")
#     print(f"Predictions: {row['next_token_preds_batch']}")
#     print(f"Target probs: {row['target_token_probs_batch']}")

# # we can get more history data by using scan_history
# scan = run.scan_history()
# for idx, row in enumerate(scan):
#     print(f"Step {idx}:")
#     print(f"Batch id: {row['batch_idx']}")
#     print(f"Loss: {row['train_loss']}")
#     print(f"Predictions: {row['next_token_preds_batch']}")
#     print(f"Target probs: {row['target_token_probs_batch']}")

img_url = "https://wp.inews.co.uk/wp-content/uploads/2023/03/SEI_149780351.jpg?crop=157px%2C0px%2C1537px%2C1537px&resize=640%2C640"
img = open_image_from_url(img_url)
# img = img.resize((384, 384)) # for DeepSeek
img = img.resize((336, 336))  # for LLaVa
init_image = transforms.ToTensor()(img)
print(init_image.max())

delta = (
    t.load(
        "data_storage/tensors/delta_DeepSeek_noimage_augment_cat.pt",
        map_location="cpu",
    )
    .detach()
    .float()
)

use_init_image = False

if use_init_image:
    eps = 8 / 255
    adv_image = (init_image + eps * delta.clamp(-1, 1)).clamp(0, 1)
    print(adv_image.max())
    transform = transforms.ToPILImage()(adv_image)
    transform.show()
else:
    transform = transforms.ToPILImage()(delta.clamp(0, 1))
    transform.show()
