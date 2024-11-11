import torch as t
import numpy as np
from torchvision import transforms
from torch.nn.functional import grid_sample, interpolate


class CustomTransforms:
    """
    A class that applies various image transformations to torch tensor images.

    The transformations can include:
    - Contrast adjustment
    - Spatial jittering
    - Color to grayscale shifting
    - Resolution downsampling with noise
    - Resolution upsampling with noise

    Parameters
    ----------
    **kwargs : dict
        contrast_range : tuple(float, float), optional
            Range for random contrast adjustment (min, max). The contrast factor is sampled uniformly from the range.
        max_jitter_ratio : float, optional
            Maximum ratio of image dimensions for random spatial shifts. The shift is sampled uniformly from the range [-max_jitter_ratio, max_jitter_ratio].
        color_amount : float, optional
            Value between 0 and 1 controlling color preservation (0=grayscale, 1=full color).
        down_res : int, optional
            Target resolution for downsampling
        down_noise : float, optional
            Noise strength to add after downsampling. Noise is sampled uniformly from the range [-down_noise/2, down_noise/2].
        up_res : int, optional
            Target resolution for upsampling
        up_noise : float, optional
            Noise strength to add after upsampling. Noise is sampled uniformly from the range [-up_noise/2, up_noise/2].

    Notes
    -----
    - Input images should be torch.Tensors in the range [0.0, 255.0]
    - The transformations are deterministically random (seed is set to 42)
    - All transformations preserve the input tensor's device and dtype
    - In order to add downsampled/upsampled noise, both (down_res, down_noise)/(up_res, up_noise) must be specified
    """

    def __init__(self, **kwargs):
        self.params = {**kwargs}
        self.seed = self.params.get("seed", 42)
        # set seed to make the transforms deterministically random
        t.manual_seed(self.seed)

    def __call__(self, image: t.Tensor) -> t.Tensor:
        """
        Apply a series of image transformations to the input image.
        Expects the image to be a torch.Tensor in the range [0.0, 255.0].

        Args:
            image (t.Tensor): The input image tensor of shape [3, H, W].

        Returns:
            t.Tensor: The transformed image tensor of shape [3, H, W].
        """
        device = image.device
        _, h, w = image.shape
        dtype = image.dtype

        # changing contrast
        if self.params.get("contrast_range") is not None:
            assert (
                len(self.params["contrast_range"]) == 2
            ), "contrast_range must be a tuple of two numbers"
            assert (
                self.params["contrast_range"][0] < self.params["contrast_range"][1]
            ), "contrast_range[0] must be less than contrast_range[1]"

            lower, upper = self.params["contrast_range"]
            contrast_factor = (
                t.rand(1, device=device, dtype=dtype) * (upper - lower) + lower
            )

            # image = transforms.functional.adjust_contrast(image, contrast_factor)
            mean = image.mean(dim=[-1, -2], keepdim=True)
            image = (image - mean) * contrast_factor + mean

        # shift the result in x and y
        if self.params.get("max_jitter_ratio") is not None:
            assert (
                self.params["max_jitter_ratio"] > 0
            ), "max_jitter_ratio must be positive"

            max_height_jitter = int(h * self.params["max_jitter_ratio"])
            max_width_jitter = int(w * self.params["max_jitter_ratio"])
            jit_x = np.random.randint(-max_height_jitter, max_height_jitter + 1)
            jit_y = np.random.randint(-max_width_jitter, max_width_jitter + 1)

            image = t.roll(image, shifts=(jit_x, jit_y), dims=(-2, -1))

        # shifting in the color <-> grayscale axis
        if self.params.get("color_amount") is not None:
            assert (
                0 <= self.params["color_amount"] <= 1
            ), "color_amount must be between 0 and 1"
            color_amount = self.params["color_amount"]
            image = color_amount * image + t.mean(image, axis=0, keepdims=True) * (
                1 - color_amount
            )

        # descrease the resolution
        if (
            self.params.get("down_res") is not None
            and self.params.get("down_noise") is not None
        ):
            down_res = self.params["down_res"]
            down_noise = self.params["down_noise"]
            image = interpolate(
                image.unsqueeze(0), size=(down_res, down_res), mode="bicubic"
            ).squeeze(0)

            # random uniform is recommended over random normal
            # this is because uniform is bouned by [0, 1), so does not suffer from outliers outside of 1 sigma, 2 sigma, etc.
            noise = down_noise * 255.0 * (t.rand_like(image) - 0.5)
            # noise = down_noise * 255.0/2.0 * t.randn_like(image)
            image = image + noise

        # increase the resolution
        if (
            self.params.get("up_res") is not None
            and self.params.get("up_noise") is not None
        ):
            up_res = self.params["up_res"]
            up_noise = self.params["up_noise"]
            image = interpolate(
                image.unsqueeze(0), size=(up_res, up_res), mode="bicubic"
            ).squeeze(0)

            noise = up_noise * 255.0 * (t.rand_like(image) - 0.5)
            # noise = up_noise * 255.0/2.0 * t.randn_like(image)
            image = image + noise

        # clipping to the right range of values
        image = t.clamp(image, 0, 255)

        return image


class CustomTransforms_old:
    def __init__(self, **kwargs):
        self.params = {**kwargs}
        self.seed = self.params.get("seed", 42)
        # set seed to make the transforms deterministically random
        t.manual_seed(self.seed)

    def __call__(self, image: t.Tensor) -> t.Tensor:
        device = image.device
        _, h, w = image.shape
        dtype = image.dtype

        # 1. Random noise
        if self.params.get("noise_strength") is not None:
            assert self.params["noise_strength"] > 0, "noise_strength must be positive"
            noise = t.randn_like(image) * self.params["noise_strength"]
            image = image + noise
            # image = t.clamp(image, 0, 1) <-- Clamping messes something up

        # 2. Random jitter
        if self.params.get("max_jitter_ratio") is not None:
            assert (
                self.params["max_jitter_ratio"] > 0
            ), "max_jitter_ratio must be positive"
            mean_side = (h + w) / 2
            max_jitter_size = mean_side * self.params["max_jitter_ratio"]

            jitter_x = t.randn((1,), device=device, dtype=dtype)
            jitter_y = t.randn((1,), device=device, dtype=dtype)

            jitter_x = t.tanh(jitter_x) * max_jitter_size / w
            jitter_y = t.tanh(jitter_y) * max_jitter_size / h

            grid_x = t.linspace(-1, 1, w, device=device, dtype=dtype)
            grid_y = t.linspace(-1, 1, h, device=device, dtype=dtype)
            grid_x, grid_y = t.meshgrid(grid_x, grid_y, indexing="xy")
            base_grid = t.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

            jitter_offset = t.stack(
                [jitter_x.expand(h, w), jitter_y.expand(h, w)], dim=-1
            ).unsqueeze(0)
            grid = base_grid + jitter_offset

            image = grid_sample(image.unsqueeze(0), grid, align_corners=True).squeeze(0)

        # 3. Random contrast
        if self.params.get("contrast_range") is not None:
            assert (
                len(self.params["contrast_range"]) == 2
            ), "contrast_range must be a tuple of two numbers"
            assert (
                self.params["contrast_range"][0] < self.params["contrast_range"][1]
            ), "contrast_range[0] must be less than contrast_range[1]"
            contrast_factor = (
                t.randn(1, device=device, dtype=dtype)
                * (self.params["contrast_range"][1] - self.params["contrast_range"][0])
                + self.params["contrast_range"][0]
            )
            mean = image.mean(dim=[-1, -2], keepdim=True)
            image = (image - mean) * contrast_factor + mean

        # 4. Random color-grayscale shift
        # I don't get what 'a small, random color-grayscale SHIFT.' means
        if self.params.get("grayscale_prob") is not None:
            assert (
                0 <= self.params["grayscale_prob"] <= 1
            ), "grayscale_prob must be between 0 and 1"
            gray_shift = transforms.RandomGrayscale(p=self.params["grayscale_prob"])
            image = gray_shift(image)

        return image
