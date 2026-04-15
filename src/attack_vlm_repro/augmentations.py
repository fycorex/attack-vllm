from __future__ import annotations

from io import BytesIO
import random

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF

from .config import AttackHyperParams


class AttackAugmentationPipeline:
    def __init__(self, config: AttackHyperParams, image_size: int):
        self.config = config
        self.image_size = image_size

    def __call__(self, images: torch.Tensor, epsilon: float) -> torch.Tensor:
        x = images
        if self.config.enable_gaussian and random.random() < self.config.gaussian_prob:
            x = self.apply_gaussian_noise(x, epsilon)
        if self.config.enable_crop and random.random() < self.config.crop_prob:
            x = self.apply_crop(x)
        x = self.apply_pad_and_resize(x)
        if self.config.enable_jpeg and random.random() < self.config.jpeg_prob:
            x = self.apply_diff_jpeg(x)
        return x.clamp(0.0, 1.0)

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
