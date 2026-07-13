from __future__ import annotations

from io import BytesIO
import random

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF

from config import AttackHyperParams


class AttackAugmentationPipeline:
    def __init__(self, config: AttackHyperParams, image_size: int):
        self.config = config
        self.image_size = image_size

    def __call__(self, images: torch.Tensor, epsilon: float, *, noise: torch.Tensor | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        x = images
        mode = self.config.noise_mode
        if noise is not None:
            x = (x + noise).clamp(0.0, 1.0)
        elif mode in {"legacy", "paper_gaussian_single_sample"} and self.config.enable_gaussian and random.random() < self.config.gaussian_prob:
            x = self.apply_gaussian_noise(x, epsilon)
        if self.config.geometry_mode not in {"legacy", "none"}:
            x = self.apply_explicit_geometry(x, self.config.geometry_mode, generator=generator)
        else:
            if self.config.geometry_mode == "legacy" and self.config.enable_crop and random.random() < self.config.crop_prob:
                x = self.apply_crop(x)
            x = self.apply_pad_and_resize(x)
        if self.config.enable_jpeg and random.random() < self.config.jpeg_prob:
            x = self.apply_diff_jpeg(x)
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _uniform(low: float, high: float, generator: torch.Generator | None, device: torch.device) -> float:
        return float((torch.rand((), generator=generator, device=device) * (high - low) + low).item())

    def apply_explicit_geometry(self, images: torch.Tensor, mode: str,
                                *, generator: torch.Generator | None = None) -> torch.Tensor:
        if mode == "geometric_mixture":
            choices = ("translation", "resize_pad", "scale")
            index = int(torch.randint(0, len(choices), (), generator=generator, device=images.device).item())
            mode = choices[index]
        outputs = []
        for image in images:
            _, height, width = image.shape
            if mode == "translation":
                max_y = int(round(height * self.config.translation_fraction))
                max_x = int(round(width * self.config.translation_fraction))
                dy = int(torch.randint(-max_y, max_y + 1, (), generator=generator, device=image.device).item()) if max_y else 0
                dx = int(torch.randint(-max_x, max_x + 1, (), generator=generator, device=image.device).item()) if max_x else 0
                padded = F.pad(image, (max_x, max_x, max_y, max_y), mode="reflect")
                image = padded[:, max_y - dy:max_y - dy + height, max_x - dx:max_x - dx + width]
            elif mode in {"resize_pad", "scale"}:
                scale = self._uniform(self.config.geometry_scale_min, self.config.geometry_scale_max, generator, image.device)
                small_h, small_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
                small = F.interpolate(image.unsqueeze(0), size=(small_h, small_w), mode="bilinear", align_corners=False, antialias=True)[0]
                if mode == "resize_pad":
                    top = int(torch.randint(0, height - small_h + 1, (), generator=generator, device=image.device).item())
                    left = int(torch.randint(0, width - small_w + 1, (), generator=generator, device=image.device).item())
                    image = F.pad(small, (left, width - small_w - left, top, height - small_h - top), mode="constant", value=0.0)
                else:
                    image = F.interpolate(small.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False, antialias=True)[0]
            else:
                raise ValueError(f"Unsupported explicit geometry mode: {mode!r}")
            outputs.append(TF.resize(image, [self.image_size, self.image_size], antialias=True))
        return torch.stack(outputs)

    def apply_gaussian_noise(self, images: torch.Tensor, epsilon: float) -> torch.Tensor:
        sigma = epsilon * self.config.gaussian_scale_multiplier
        return (images + sigma * torch.randn_like(images)).clamp(0.0, 1.0)

    def apply_crop(self, images: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for image in images:
            top, left, crop_h, crop_w = RandomResizedCrop.get_params(
                image,
                scale=(self.config.crop_scale_min, self.config.crop_scale_max),
                ratio=(self.config.crop_ratio_min, self.config.crop_ratio_max),
            )
            outputs.append(TF.crop(image, top, left, crop_h, crop_w))
        return outputs

    def apply_pad_and_resize(self, images: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            image_list = [image for image in images]
        else:
            image_list = images
        outputs = []
        for image in image_list:
            _, height, width = image.shape
            if self.config.enable_pad and random.random() < self.config.pad_prob:
                pad_h = max(0, self.image_size - height)
                pad_w = max(0, self.image_size - width)
                if pad_h > 0 or pad_w > 0:
                    pad_top = random.randint(0, pad_h) if pad_h > 0 else 0
                    pad_bottom = pad_h - pad_top
                    pad_left = random.randint(0, pad_w) if pad_w > 0 else 0
                    pad_right = pad_w - pad_left
                    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
            image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
            outputs.append(image)
        return torch.stack(outputs, dim=0)

    def apply_diff_jpeg(self, images: torch.Tensor) -> torch.Tensor:
        backend = self.config.jpeg_backend.strip().lower()
        if backend == "tensor":
            return self.apply_tensor_jpeg(images)
        if backend == "pil":
            return self.apply_pil_jpeg(images)
        raise ValueError(f"Unsupported jpeg_backend: {self.config.jpeg_backend}")

    def apply_tensor_jpeg(self, images: torch.Tensor) -> torch.Tensor:
        quality = random.uniform(self.config.jpeg_quality_min, self.config.jpeg_quality_max)
        levels = max(8, int(16 + (quality * 239)))
        compressed = torch.round(images * levels) / levels

        if quality < 0.95:
            _, _, height, width = images.shape
            scale = max(0.5, min(1.0, 0.5 + (0.5 * quality)))
            down_h = max(1, int(round(height * scale)))
            down_w = max(1, int(round(width * scale)))
            compressed = F.interpolate(
                compressed,
                size=(down_h, down_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            compressed = F.interpolate(
                compressed,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        compressed = compressed.clamp(0.0, 1.0)
        return images + (compressed - images).detach()

    def apply_pil_jpeg(self, images: torch.Tensor) -> torch.Tensor:
        quality = random.uniform(self.config.jpeg_quality_min, self.config.jpeg_quality_max)
        jpeg_images = []
        for image in images:
            pil = TF.to_pil_image(image.detach().cpu().clamp(0.0, 1.0))
            buffer = BytesIO()
            pil.save(buffer, format="JPEG", quality=max(1, min(100, int(quality * 100))))
            buffer.seek(0)
            jpeg_pil = Image.open(buffer).convert("RGB")
            jpeg_images.append(TF.to_tensor(jpeg_pil))
        jpeg_tensor = torch.stack(jpeg_images, dim=0).to(images.device)
        return images + (jpeg_tensor - images).detach()


def effective_sigma(config: AttackHyperParams, epsilon: float, progress: float = 0.0) -> float:
    sigma = float(config.noise_sigma) if config.noise_sigma is not None else epsilon * config.gaussian_scale_multiplier
    progress = max(0.0, min(1.0, progress))
    if config.noise_schedule == "linear_decay":
        sigma *= 1.0 - progress
    elif config.noise_schedule == "cosine_decay":
        sigma *= 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
    return sigma


def sample_noise(mode: str, reference: torch.Tensor, sigma: float, *, generator: torch.Generator | None = None,
                 antithetic_base: torch.Tensor | None = None, antithetic_sign: int = 1) -> torch.Tensor:
    if mode in {"none", "legacy", "paper_gaussian_single_sample"} or sigma == 0:
        return torch.zeros_like(reference)
    if mode in {"gaussian_eot", "antithetic_gaussian_eot"}:
        base = antithetic_base if antithetic_base is not None else torch.randn(reference.shape, device=reference.device, dtype=reference.dtype, generator=generator)
        return base * sigma * antithetic_sign
    if mode == "uniform_eot":
        bound = (3.0 ** 0.5) * sigma
        return torch.empty_like(reference).uniform_(-bound, bound, generator=generator)
    if mode == "rademacher_eot":
        values = torch.randint(0, 2, reference.shape, device=reference.device, generator=generator)
        return (values.to(reference.dtype) * 2.0 - 1.0) * sigma
    raise ValueError(f"Unsupported explicit noise mode: {mode!r}")
