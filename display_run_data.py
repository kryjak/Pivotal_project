import torch as t
import wandb
from torchvision import transforms
from utils import open_image_from_url

# api = wandb.Api()

# Get a specific run
# run = api.run("run_path")

# Method 1: Get all history (returns pandas DataFrame)
# history_df = run.history(samples=None)
# print(history_df["target_token_probs_batch"])  # access specific column
# print(history_df["train_loss"])  # access specific column

# # Method 2: Get summary statistics
# summary = run.summary
# print(summary.keys())  # see what metrics are available

# print(list(run.scan_history())[0])
# Method 3: Scan through all metrics
# for idx, row in enumerate(run.scan_history()):
#     # row will be a dict with your logged metrics
#     print(f"idx: {idx}")
#     print(row["next_token_preds_batch"])
#     print(row["target_token_probs_batch"])


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
