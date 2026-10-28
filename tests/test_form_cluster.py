import numpy as np
import pytest

from formcoreset.cluster import cluster_features


def test_cluster_features_separates_groups() -> None:
    features = np.asarray(
        [[1.0, 0.0, 0.0, 0.0]] * 4
        + [[0.0, 1.0, 0.0, 0.0]] * 4
        + [[0.0, 0.0, 1.0, 0.0]] * 4,
        dtype=np.float32,
    )

    result = cluster_features(features, min_cluster_size=2, min_samples=1)

    assert result.labels.shape == (12,)
    assert result.n_clusters >= 2
    assert result.params["min_cluster_size"] >= 2


def test_cluster_features_rejects_1d() -> None:
    with pytest.raises(ValueError):
        cluster_features(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_cluster_features_requires_two_rows() -> None:
    with pytest.raises(ValueError):
        cluster_features(np.asarray([[1.0, 0.0]], dtype=np.float32))
