import torch as t
from torchvision import transforms
from torch.nn.functional import grid_sample


class CustomTransforms:
    def __init__(self, **kwargs):
        self.params = {**kwargs}
        t.manual_seed(42)  # set seed to make the transforms deterministically random

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
