import requests
import wandb
import os
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import torch as t
import einops
from typing import Union, List


### IMAGE UTILS ###
def open_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            raise Exception(
                f"Failed to retrieve image. Status code: {response.status_code}"
            )
    except Exception as e:
        print(e)
        raise


def _denormalise(
    image: t.Tensor, mean: Union[float, List[float]], std: Union[float, List[float]]
) -> t.Tensor:
    return (image * std) + mean


# function to display the image processed with LlavaNextImageProcessor
def display_processed_tiles(
    pixel_values: t.Tensor,
    mean: Union[float, List[float]],
    std: Union[float, List[float]],
) -> Image.Image:
    """
    Function to display the tiles produced from an image by the LlavaNextImageProcessor.

    Args:
        pixel_values (t.Tensor): output from the LlavaNextImageProcessor
        mean (float or list of floats): mean used to normalise the image
        std (float or list of floats): standard deviation used to normalise the image

    Returns:
        canvas (Image): PIL image with the tiles displayed in a grid.
    """
    processed_image = einops.rearrange(pixel_values, "b t c h w -> b t h w c")
    print(f"{processed_image.shape=}")

    # processed_image needs to be the output of the processor
    if processed_image.shape[0] == 1:
        processed_image = processed_image.squeeze(0)
    else:
        raise NotImplementedError("Batch size > 1 not implemented yet")

    # Calculate the number of tiles per row
    num_tiles = processed_image.shape[0]
    # tiles_per_row = 4  # Adjust this value to change the number of tiles per row
    tiles_per_row = num_tiles  # Adjust this value to change the number of tiles per row
    num_rows = (num_tiles + tiles_per_row - 1) // tiles_per_row

    # Create a single canvas to hold the stacked images
    max_width = max(processed_image[i].shape[1] for i in range(num_tiles))
    max_height = max(processed_image[i].shape[0] for i in range(num_tiles))
    canvas_width = max_width * tiles_per_row + (tiles_per_row + 1) * 2
    canvas_height = max_height * num_rows + (num_rows + 1) * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))

    # Draw the tiles on the canvas with a red frame
    for i in range(num_tiles):
        row = i // tiles_per_row
        col = i % tiles_per_row
        x = (
            col * max_width + (col + 0) * 2
        )  # Add 2 pixels for the frame around each tile
        y = (
            row * max_height + (row + 0) * 2
        )  # Add 2 pixels for the frame around each tile
        image = _denormalise(processed_image[i], mean, std)
        image_pil = Image.fromarray(image.cpu().numpy().astype(np.uint8))

        # Add a red frame around the image
        frame_width = 2  # Adjust the frame width as desired
        frame_color = (255, 0, 0)  # Red color
        framed_image = Image.new(
            "RGB",
            (image_pil.width + frame_width * 2, image_pil.height + frame_width * 2),
            frame_color,
        )
        framed_image.paste(image_pil, (frame_width, frame_width))

        # Draw the frame around the image
        draw = ImageDraw.Draw(framed_image)
        draw.rectangle(
            [(0, 0), (framed_image.width - 1, framed_image.height - 1)],
            outline=frame_color,
            width=frame_width,
        )

        canvas.paste(framed_image, (x, y))

    return canvas


### WANDB UTILS ###
def load_data_from_wandb(wandb_run_name: str) -> None:
    wandb_api = wandb.Api()
    run = wandb_api.run(wandb_run_name)
    files = [f.name for f in run.files() if f.name.endswith(".pt")]

    path_to_wandb_tensors = "data_storage/tensors"
    for tensor_name in ["init_image", "delta"]:
        tensor_file = os.path.join(path_to_wandb_tensors, tensor_name + ".pt")
        if tensor_file in files:
            run.file(tensor_file).download(replace=True)
            globals()[tensor_name] = t.load(tensor_file)
        else:
            globals()[tensor_name] = None

    print(f"Loaded data from wandb run {wandb_run_name}")


class DotDict(dict):
    """A dictionary that supports dot notation as well as dictionary access notation"""

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def __add__(self, other):
        if isinstance(other, dict):
            new = DotDict(self)
            new.update(other)
            return new
        return NotImplemented
