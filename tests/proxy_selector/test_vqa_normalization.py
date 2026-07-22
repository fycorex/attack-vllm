from proxy_selector.vqa_normalization import normalize_answer


def test_vqa_normalization_handles_articles_numbers_and_punctuation() -> None:
    assert normalize_answer("The, Two dogs!") == "2 dogs"
    assert normalize_answer("  An   apple. ") == "apple"
