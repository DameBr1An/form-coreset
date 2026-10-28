"""Toy demo for the form-caption joint redundancy metric."""

import numpy as np

from formcoreset.redundancy import (
    JointRedundancyConfig,
    greedy_select,
    marginal_information,
    pair_redundancy,
)


def main() -> None:
    config = JointRedundancyConfig()

    print("Pairwise breakdowns")
    cases = [
        ("same form, same caption", 0.95, 0.92),
        ("same form, different caption", 0.95, 0.20),
        ("different form, same caption", 0.20, 0.92),
        ("different form, different caption", 0.20, 0.20),
    ]
    for label, form_similarity, caption_similarity in cases:
        breakdown = pair_redundancy(form_similarity, caption_similarity, config)
        print(
            f"- {label}: joint={breakdown.joint_redundancy:.3f} "
            f"(form={breakdown.form_redundancy:.3f}, "
            f"caption={breakdown.caption_redundancy:.3f}, "
            f"dominant={breakdown.dominant})"
        )

    form = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    caption = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    print("\nMarginal information relative to item 0")
    for index in (1, 2):
        info = marginal_information(
            form,
            caption,
            candidate_index=index,
            selected_indices=[0],
            quality_score=0.9,
            config=config,
        )
        print(
            f"- item {index}: novelty={info.novelty:.3f}, "
            f"redundancy={info.redundancy.joint_redundancy:.3f}, "
            f"score={info.score:.3f}"
        )

    selected = greedy_select(
        form,
        caption,
        quality_scores=[0.9, 0.8, 0.85],
        max_items=3,
        form_ids=[0, 0, 1],
        per_form_cap=1,
        config=config,
    )
    print(f"\nGreedy selection with per-form cap 1: {selected}")


if __name__ == "__main__":
    main()
