from proxy_selector.adapters.qwen35 import qwen_smart_resize


def test_qwen_smart_resize_matches_grid_constraints() -> None:
    assert qwen_smart_resize(224, 224, factor=32, min_pixels=65536, max_pixels=16777216) == (256, 256)
    height, width = qwen_smart_resize(1000, 2000, factor=32, min_pixels=65536, max_pixels=65536)
    assert height % 32 == width % 32 == 0
    assert height * width <= 65536
