import numpy as np
import pytest

from formcoreset.redundancy import (
    JointRedundancyConfig,
    cosine_similarity_matrix,
    greedy_select,
    hybrid_caption_similarity,
    jaccard_similarity,
    marginal_information,
    pair_redundancy,
    redundancy_to_set,
    thresholded_similarity,
)


def test_thresholded_similarity() -> None:
    assert thresholded_similarity(0.8, 0.8) == 0.0
    assert thresholded_similarity(0.9, 0.8) == pytest.approx(0.5)
    assert thresholded_similarity(1.0, 0.8) == pytest.approx(1.0)
    assert thresholded_similarity(0.5, 0.8) == 0.0


def test_pair_redundancy_pure_and() -> None:
    config = JointRedundancyConfig(
        form_threshold=0.5,
        caption_threshold=0.5,
        form_floor=0.0,
        caption_floor=0.0,
    )

    both = pair_redundancy(1.0, 1.0, config)
    form_only = pair_redundancy(1.0, 0.0, config)
    caption_only = pair_redundancy(0.0, 1.0, config)

    assert both.joint_redundancy == pytest.approx(1.0)
    assert both.dominant == "joint"
    assert form_only.joint_redundancy == pytest.approx(0.0)
    assert form_only.dominant == "form"
    assert caption_only.joint_redundancy == pytest.approx(0.0)
    assert caption_only.dominant == "caption"


def test_pair_redundancy_default_floors() -> None:
    form_only = pair_redundancy(1.0, 0.0)
    caption_only = pair_redundancy(0.0, 1.0)
    both = pair_redundancy(1.0, 1.0)

    assert form_only.joint_redundancy == pytest.approx(0.30)
    assert caption_only.joint_redundancy == pytest.approx(0.10)
    assert both.joint_redundancy == pytest.approx(1.0)


def test_pair_redundancy_without_caption() -> None:
    result = pair_redundancy(0.9, None)

    assert result.form_redundancy == pytest.approx(0.5)
    assert result.caption_redundancy == 0.0
    assert result.joint_redundancy == pytest.approx(0.5)
    assert result.dominant == "form"


def test_pair_redundancy_bounds() -> None:
    for form in (0.0, 0.25, 0.5, 0.8, 1.0):
        for caption in (0.0, 0.25, 0.5, 0.8, 1.0):
            result = pair_redundancy(form, caption)
            assert 0.0 <= result.joint_redundancy <= 1.0


def test_cosine_similarity_matrix() -> None:
    matrix = cosine_similarity_matrix([[1.0, 0.0], [0.0, 1.0]])

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[0, 1] == pytest.approx(0.0)


def test_redundancy_to_set() -> None:
    form = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    caption = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    same = redundancy_to_set(form, caption, 1, [0])
    different_form_same_caption = redundancy_to_set(form, caption, 2, [0])

    assert same.joint_redundancy == pytest.approx(1.0)
    assert same.dominant == "joint"
    assert different_form_same_caption.joint_redundancy == pytest.approx(0.10)
    assert different_form_same_caption.dominant == "caption"


def test_marginal_information_prefers_new_information() -> None:
    form = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    caption = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    duplicate = marginal_information(form, caption, 1, [0], quality_score=1.0)
    novel = marginal_information(form, caption, 2, [0], quality_score=1.0)

    assert duplicate.form_novelty == pytest.approx(0.0)
    assert duplicate.caption_novelty == pytest.approx(0.0)
    assert duplicate.novelty == pytest.approx(0.0)
    assert duplicate.redundancy.joint_redundancy == pytest.approx(1.0)
    assert novel.novelty == pytest.approx(1.0)
    assert novel.score > duplicate.score


def test_greedy_select_respects_per_form_cap() -> None:
    form = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    caption = np.eye(4, dtype=np.float32)
    quality = [1.0, 1.0, 1.0, 1.0]

    selected = greedy_select(
        form,
        caption,
        quality,
        max_items=4,
        form_ids=[0, 0, 0, 1],
        per_form_cap=1,
    )

    assert len(selected) == 2
    assert set(selected) == {0, 3}


def test_jaccard_similarity() -> None:
    assert jaccard_similarity("1girl, solo", "1girl, smile") == pytest.approx(1 / 3)
    assert jaccard_similarity(
        "1girl, solo",
        "1girl, smile",
        stop_tags={"1girl"},
    ) == pytest.approx(0.0)


def test_hybrid_caption_similarity() -> None:
    assert hybrid_caption_similarity(0.0, 1.0, tag_weight=0.5) == pytest.approx(0.5)
    assert hybrid_caption_similarity(1.0, 1.0, tag_weight=0.5) == pytest.approx(1.0)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        JointRedundancyConfig(form_threshold=1.5)
    with pytest.raises(ValueError):
        JointRedundancyConfig(top_k=0)
