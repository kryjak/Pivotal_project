from torch.utils.data import Dataset
import torch as t
from typing import Union, List


class VLMJailbreakDataset(Dataset):
    def __init__(
        self,
        prompts: List,
        images: Union[List[t.Tensor], None],
        targets: List[List[str]],
    ):
        self.prompts = prompts
        self.images = images
        self.targets = targets

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        # image = self.images[idx] if self.images else None
        image = self.images[idx] if self.images is not None else None
        target = self.targets[idx]
        return prompt, image, target


def custom_collate_fn(batch):
    prompts, images, targets = zip(*batch)

    # Stack images if they exist, otherwise return a list of Nones
    images = t.stack(images) if images[0] is not None else None

    return list(prompts), images, list(targets)


### USAGE ###
# dataset = VLMJailbreakDataset(prompts=prompts, images=init_images, targets=targets)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
# for batch in dataloader:
#     prompts_batch, images_batch, targets_batch = batch
#     print(prompts_batch)  # List of prompts in the batch
#     print(images_batch)   # Should be None
#     print(targets_batch)  # List of targets in the batch
#     break
