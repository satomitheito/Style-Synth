"""
Pytest tests for the Fashion Recommendation Engine.

Run with:
    pytest tests/ -v
    pytest tests/ -v -m "not integration"  # Skip integration tests
"""

import numpy as np
import pytest
import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation_engine import FashionRecommendationEngine


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_embeddings():
    """Create small sample embeddings for unit tests."""
    np.random.seed(42)
    return np.random.randn(100, 512).astype('float32')


@pytest.fixture
def sample_labels():
    """Create sample labels for unit tests."""
    np.random.seed(42)
    return np.random.randint(0, 10, size=100)


@pytest.fixture
def sample_class_names():
    """Sample Fashion-MNIST class names."""
    return [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]


@pytest.fixture
def engine_with_index(sample_embeddings, sample_labels, sample_class_names):
    """Create an engine with a built index."""
    engine = FashionRecommendationEngine(
        embeddings=sample_embeddings,
        labels=sample_labels,
        class_names=sample_class_names,
        n_components=64,
        use_pca=True,
        index_type="L2"
    )
    engine.build_index()
    return engine


@pytest.fixture
def real_data():
    """Load real embeddings and labels for integration tests."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    cv_path = os.path.join(base_path, '..', '..', 'ComputerVisionFiles')

    embeddings_path = os.path.join(cv_path, 'fashion_mnist_resnet50_embeddings.npy')
    labels_path = os.path.join(cv_path, 'fashion_mnist_labels.npy')
    classes_path = os.path.join(cv_path, 'fashion_mnist_classes.txt')

    if not all(os.path.exists(p) for p in [embeddings_path, labels_path, classes_path]):
        pytest.skip("Data files not available. Run 'git lfs pull' to download.")

    try:
        embeddings = np.load(embeddings_path, allow_pickle=True)
        labels = np.load(labels_path, allow_pickle=True)
        # Check if files are valid (LFS pointer files will be small/invalid)
        if embeddings.size < 1000:
            pytest.skip("LFS files not pulled. Run 'git lfs pull' to download.")
    except Exception as e:
        pytest.skip(f"Could not load data files: {e}")

    with open(classes_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    return embeddings, labels, class_names


# =============================================================================
# Unit Tests
# =============================================================================

class TestInit:
    """Tests for engine initialization."""

    def test_init_with_valid_params(self, sample_embeddings, sample_labels, sample_class_names):
        """Test engine initializes with valid parameters."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            n_components=64
        )

        assert engine.n_samples == 100
        assert engine.original_dim == 512
        assert engine.n_components == 64
        assert engine.index is None  # Not built yet


class TestBuildIndex:
    """Tests for index building."""

    @pytest.mark.parametrize("index_type", ["L2", "cosine"])
    def test_build_index(self, sample_embeddings, sample_labels, sample_class_names, index_type):
        """Test building index with L2 and cosine."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            n_components=64,
            index_type=index_type
        )
        engine.build_index()

        assert engine.index is not None
        assert engine.index.ntotal == 100
        assert engine.reduced_embeddings.shape == (100, 64)

    def test_build_index_with_metadata(self, sample_embeddings, sample_labels, sample_class_names):
        """Test building index with metadata."""
        metadata = {'item_ids': list(range(100))}

        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            n_components=64
        )
        engine.build_index(metadata=metadata)

        assert engine.item_metadata == metadata

    def test_invalid_index_type_raises_error(self, sample_embeddings, sample_labels, sample_class_names):
        """Test that invalid index type raises error."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            use_pca=False,
            index_type="invalid"
        )

        with pytest.raises(ValueError, match="Unknown index_type"):
            engine.build_index()


class TestSearch:
    """Tests for search functionality."""

    def test_search_returns_k_results(self, engine_with_index, sample_embeddings):
        """Test search returns requested number of results."""
        query = sample_embeddings[0]
        distances, indices = engine_with_index.search(query, k=5)

        assert len(indices[0]) == 5
        assert len(distances[0]) == 5

    def test_search_without_index_raises_error(self, sample_embeddings, sample_labels, sample_class_names):
        """Test search without built index raises error."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names
        )

        with pytest.raises(ValueError, match="Index not built"):
            engine.search(sample_embeddings[0], k=5)

    def test_search_self_is_most_similar(self, engine_with_index, sample_embeddings):
        """Test that an item is most similar to itself."""
        query = sample_embeddings[0]
        distances, indices = engine_with_index.search(query, k=1)

        assert indices[0][0] == 0
        assert distances[0][0] < 0.001


class TestFiltering:
    """Tests for filtering functionality."""

    def test_filter_by_class(self, engine_with_index, sample_embeddings, sample_labels):
        """Test filtering results by class."""
        query = sample_embeddings[0]
        filter_classes = [0, 1]

        _, indices = engine_with_index.search(query, k=10, filter_by_class=filter_classes)

        for idx in indices[0]:
            assert sample_labels[idx] in filter_classes

    def test_exclude_indices(self, engine_with_index, sample_embeddings):
        """Test excluding specific indices from results."""
        query = sample_embeddings[0]
        exclude = [0, 1, 2]

        _, indices = engine_with_index.search(query, k=10, exclude_indices=exclude)

        for idx in indices[0]:
            assert idx not in exclude


class TestRecommend:
    """Tests for the recommend method."""

    def test_recommend_returns_formatted_results(self, engine_with_index, sample_embeddings):
        """Test recommend returns properly formatted results."""
        query = sample_embeddings[0]
        recommendations = engine_with_index.recommend(query, k=5)

        assert isinstance(recommendations, list)
        assert len(recommendations) == 5

        for rec in recommendations:
            assert 'index' in rec
            assert 'distance' in rec
            assert 'label' in rec
            assert 'class_name' in rec
            assert 'rank' in rec

    def test_recommend_with_metadata(self, sample_embeddings, sample_labels, sample_class_names):
        """Test recommend returns metadata when requested."""
        metadata = {'item_ids': list(range(100))}

        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            n_components=64
        )
        engine.build_index(metadata=metadata)

        recommendations = engine.recommend(sample_embeddings[0], k=5, return_metadata=True)

        for rec in recommendations:
            assert 'item_ids' in rec


class TestSaveLoad:
    """Tests for save and load functionality."""

    def test_save_and_load(self, engine_with_index, sample_embeddings):
        """Test saving and loading the engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_engine.pkl')

            engine_with_index.save(save_path)

            assert os.path.exists(save_path)
            assert os.path.exists(save_path.replace('.pkl', '_index.faiss'))

            loaded_engine = FashionRecommendationEngine.load(save_path)

            query = sample_embeddings[0]
            original_recs = engine_with_index.recommend(query, k=5)
            loaded_recs = loaded_engine.recommend(query, k=5)

            assert len(original_recs) == len(loaded_recs)
            for orig, loaded in zip(original_recs, loaded_recs):
                assert orig['index'] == loaded['index']


# =============================================================================
# Integration Test
# =============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration test with real Fashion-MNIST data."""

    def test_full_workflow_with_real_data(self, real_data):
        """Test complete workflow with real embeddings."""
        embeddings, labels, class_names = real_data

        engine = FashionRecommendationEngine(
            embeddings=embeddings,
            labels=labels,
            class_names=class_names,
            n_components=128,
            use_pca=True,
            index_type="L2"
        )
        engine.build_index()

        # Test search
        query_idx = 0
        recommendations = engine.recommend(embeddings[query_idx], k=5)

        assert len(recommendations) == 5
        assert recommendations[0]['index'] == query_idx  # Most similar to itself

        # Test filtering
        filter_classes = [0, 1]
        filtered_recs = engine.recommend(
            embeddings[query_idx], k=10, filter_by_class=filter_classes
        )
        for rec in filtered_recs:
            assert rec['label'] in filter_classes
