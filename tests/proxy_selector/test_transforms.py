import torch

from proxy_selector.transforms import eot_transform


def test_eot_shape_and_gradient() -> None:
    image = torch.rand(1, 3, 32, 40, requires_grad=True)
    transformed = eot_transform(image, translation_pixels=4, resize_min=.95, resize_max=1.05, generator=torch.Generator().manual_seed(42))
    assert transformed.shape == image.shape
    transformed.square().mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
