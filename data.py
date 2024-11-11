from torch.utils.data import Dataset
import torch as t
from typing import Union, List


class VLMJailbreakDataset(Dataset):
    """A PyTorch Dataset for handling VLM (Vision Language Model) data.

    This dataset manages prompts, optional images, and their corresponding targets.
    It supports both text-only and multimodal (text + image) data formats.

    Args:
        prompts (List): A list of text prompts.
        images (Union[List[t.Tensor], None]): A list of image tensors or None if no images.
        targets (List[List[str]]): A list of target responses, where each target is a list of strings (tokens).

    Attributes:
        prompts (List): Stored text prompts.
        images (Union[List[t.Tensor], None]): Stored image tensors or None.
        targets (List[List[str]]): Stored target responses.
    """
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
    """Custom collation function for DataLoader to handle batches with optional images.

    This function processes batches by separating prompts, images, and targets.
    Images are stacked into a single tensor if present, otherwise returns None.

    Args:
        batch: A list of tuples, where each tuple contains (prompt, image, target).

    Returns:
        tuple: Contains:
            - list: Batch of prompts
            - Union[torch.Tensor, None]: Stacked images or None if no images
            - list: Batch of targets
    """
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
