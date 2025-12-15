"""
Pytest tests for the Fashion Recommendation Engine.

This module contains both unit tests and integration tests for the
recommendation engine. Integration tests require actual data files.

Run from RecommendationFiles directory with:
    pytest tests/ -v
    pytest tests/ -v -m "not integration"  # Skip integration tests
    pytest tests/ -v -m "integration"      # Only integration tests
"""

import numpy as np
import pytest
import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation_engine import (
    FashionRecommendationEngine,
    benchmark_search_speed,
    analyze_memory_usage,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_embeddings():
    """Create small sample embeddings for unit tests."""
    np.random.seed(42)
    n_samples = 100
    embedding_dim = 512
    return np.random.randn(n_samples, embedding_dim).astype('float32')


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
def real_data_path():
    """Path to real data files."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    cv_path = os.path.join(base_path, '..', '..', 'ComputerVisionFiles')
    return cv_path


@pytest.fixture
def real_data(real_data_path):
    """
    Load real embeddings and labels for integration tests.
    Skip if data files not available.
    """
    embeddings_path = os.path.join(real_data_path, 'fashion_mnist_resnet50_embeddings.npy')
    labels_path = os.path.join(real_data_path, 'fashion_mnist_labels.npy')
    classes_path = os.path.join(real_data_path, 'fashion_mnist_classes.txt')

    if not all(os.path.exists(p) for p in [embeddings_path, labels_path, classes_path]):
        pytest.skip("Data files not available. Run 'git lfs pull' to download.")

    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    with open(classes_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    return embeddings, labels, class_names


# =============================================================================
# Unit Tests
# =============================================================================

class TestFashionRecommendationEngineInit:
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
        assert engine.use_pca is True
        assert engine.index_type == "L2"
        assert engine.index is None  # Not built yet

    def test_init_without_pca(self, sample_embeddings, sample_labels, sample_class_names):
        """Test engine can be initialized without PCA."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            use_pca=False
        )

        assert engine.use_pca is False

    def test_init_with_cosine_similarity(self, sample_embeddings, sample_labels, sample_class_names):
        """Test engine can use cosine similarity."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            index_type="cosine"
        )

        assert engine.index_type == "cosine"


class TestBuildIndex:
    """Tests for index building."""

    def test_build_index_l2(self, sample_embeddings, sample_labels, sample_class_names):
        """Test building L2 index."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            n_components=64,
            index_type="L2"
        )
        engine.build_index()

        assert engine.index is not None
        assert engine.index.ntotal == 100
        assert engine.reduced_embeddings is not None
        assert engine.reduced_embeddings.shape == (100, 64)

    def test_build_index_cosine(self, sample_embeddings, sample_labels, sample_class_names):
        """Test building cosine similarity index."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            n_components=64,
            index_type="cosine"
        )
        engine.build_index()

        assert engine.index is not None
        assert engine.index.ntotal == 100

    def test_build_index_without_pca(self, sample_embeddings, sample_labels, sample_class_names):
        """Test building index without PCA reduction."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            use_pca=False
        )
        engine.build_index()

        assert engine.index is not None
        assert engine.reduced_embeddings.shape == (100, 512)  # Original dimension

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


class TestSearch:
    """Tests for the search functionality."""

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

        # First result should be the query item itself
        assert indices[0][0] == 0
        # Distance to itself should be very small (close to 0)
        assert distances[0][0] < 0.001


class TestFiltering:
    """Tests for filtering functionality."""

    def test_filter_by_class(self, engine_with_index, sample_embeddings, sample_labels):
        """Test filtering results by class."""
        query = sample_embeddings[0]
        filter_classes = [0, 1]

        _, indices = engine_with_index.search(
            query, k=10, filter_by_class=filter_classes
        )

        for idx in indices[0]:
            assert sample_labels[idx] in filter_classes

    def test_exclude_indices(self, engine_with_index, sample_embeddings):
        """Test excluding specific indices from results."""
        query = sample_embeddings[0]
        exclude = [0, 1, 2]

        _, indices = engine_with_index.search(
            query, k=10, exclude_indices=exclude
        )

        for idx in indices[0]:
            assert idx not in exclude

    def test_combined_filter_and_exclude(self, engine_with_index, sample_embeddings, sample_labels):
        """Test combining class filter and index exclusion."""
        query = sample_embeddings[0]
        filter_classes = [0, 1, 2]
        exclude = [0, 1]

        _, indices = engine_with_index.search(
            query, k=5, filter_by_class=filter_classes, exclude_indices=exclude
        )

        for idx in indices[0]:
            assert sample_labels[idx] in filter_classes
            assert idx not in exclude


class TestRecommend:
    """Tests for the recommend method."""

    def test_recommend_returns_list_of_dicts(self, engine_with_index, sample_embeddings):
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

    def test_recommend_ranks_are_sequential(self, engine_with_index, sample_embeddings):
        """Test that ranks are sequential starting from 1."""
        query = sample_embeddings[0]
        recommendations = engine_with_index.recommend(query, k=5)

        ranks = [rec['rank'] for rec in recommendations]
        assert ranks == [1, 2, 3, 4, 5]

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

        recommendations = engine.recommend(
            sample_embeddings[0], k=5, return_metadata=True
        )

        for rec in recommendations:
            assert 'item_ids' in rec


class TestSaveLoad:
    """Tests for save and load functionality."""

    def test_save_and_load(self, engine_with_index, sample_embeddings):
        """Test saving and loading the engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_engine.pkl')

            # Save
            engine_with_index.save(save_path)

            # Verify files created
            assert os.path.exists(save_path)
            assert os.path.exists(save_path.replace('.pkl', '_index.faiss'))

            # Load
            loaded_engine = FashionRecommendationEngine.load(save_path)

            # Verify loaded engine works
            query = sample_embeddings[0]
            original_recs = engine_with_index.recommend(query, k=5)
            loaded_recs = loaded_engine.recommend(query, k=5)

            assert len(original_recs) == len(loaded_recs)
            for orig, loaded in zip(original_recs, loaded_recs):
                assert orig['index'] == loaded['index']


class TestNormalization:
    """Tests for embedding normalization."""

    def test_l2_index_does_not_normalize(self, sample_embeddings, sample_labels, sample_class_names):
        """Test L2 index doesn't normalize embeddings."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            index_type="L2"
        )

        # _normalize_embeddings should return unchanged for L2
        result = engine._normalize_embeddings(sample_embeddings)
        np.testing.assert_array_equal(result, sample_embeddings)

    def test_cosine_index_normalizes(self, sample_embeddings, sample_labels, sample_class_names):
        """Test cosine index normalizes embeddings."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            index_type="cosine"
        )

        result = engine._normalize_embeddings(sample_embeddings)

        # Check that all vectors are unit length
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(sample_embeddings)), decimal=5)


class TestHelperFunctions:
    """Tests for standalone helper functions."""

    def test_benchmark_search_speed(self, engine_with_index):
        """Test benchmark function returns expected metrics."""
        results = benchmark_search_speed(engine_with_index, n_queries=10, k=5)

        assert 'total_time' in results
        assert 'avg_time_per_query' in results
        assert 'queries_per_second' in results
        assert results['n_queries'] == 10
        assert results['k'] == 5

    def test_analyze_memory_usage(self, engine_with_index):
        """Test memory analysis function returns expected metrics."""
        results = analyze_memory_usage(engine_with_index)

        assert 'original_embeddings_size_mb' in results
        assert 'reduced_embeddings_size_mb' in results
        assert 'compression_ratio' in results
        assert 'n_samples' in results


# =============================================================================
# Integration Tests (require real data files)
# =============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests that use real Fashion-MNIST data."""

    def test_basic_functionality_with_real_data(self, real_data):
        """Test basic functionality with real embeddings."""
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
        query_embedding = embeddings[query_idx]
        recommendations = engine.recommend(query_embedding, k=5)

        assert len(recommendations) == 5
        assert recommendations[0]['rank'] == 1
        assert recommendations[0]['index'] == query_idx  # Most similar to itself

    def test_filtering_with_real_data(self, real_data):
        """Test filtering with real data."""
        embeddings, labels, class_names = real_data

        engine = FashionRecommendationEngine(
            embeddings=embeddings,
            labels=labels,
            class_names=class_names,
            n_components=128
        )
        engine.build_index()

        query_embedding = embeddings[0]

        # Test class filtering
        filter_classes = [0, 1]
        recommendations = engine.recommend(
            query_embedding, k=10, filter_by_class=filter_classes
        )

        for rec in recommendations:
            assert rec['label'] in filter_classes

        # Test exclusion
        exclude = [0, 1, 2]
        recommendations = engine.recommend(
            query_embedding, k=10, exclude_indices=exclude
        )

        for rec in recommendations:
            assert rec['index'] not in exclude

    def test_save_load_with_real_data(self, real_data):
        """Test save/load with real data."""
        embeddings, labels, class_names = real_data

        engine = FashionRecommendationEngine(
            embeddings=embeddings,
            labels=labels,
            class_names=class_names,
            n_components=128
        )
        engine.build_index()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'integration_test_engine.pkl')
            engine.save(save_path)

            loaded_engine = FashionRecommendationEngine.load(save_path)

            query = embeddings[0]
            orig_recs = engine.recommend(query, k=5)
            loaded_recs = loaded_engine.recommend(query, k=5)

            for r1, r2 in zip(orig_recs, loaded_recs):
                assert r1['index'] == r2['index']


# =============================================================================
# Parametrized Tests
# =============================================================================

class TestParametrized:
    """Parametrized tests for various configurations."""

    @pytest.mark.parametrize("n_components", [16, 32, 64])  # Keep <= min(n_samples, n_features)
    def test_different_pca_dimensions(self, sample_embeddings, sample_labels, sample_class_names, n_components):
        """Test engine works with different PCA dimensions."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            n_components=n_components
        )
        engine.build_index()

        assert engine.reduced_embeddings.shape[1] == n_components

        recommendations = engine.recommend(sample_embeddings[0], k=5)
        assert len(recommendations) == 5

    @pytest.mark.parametrize("index_type", ["L2", "cosine"])
    def test_different_index_types(self, sample_embeddings, sample_labels, sample_class_names, index_type):
        """Test engine works with different index types."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            n_components=64,
            index_type=index_type
        )
        engine.build_index()

        recommendations = engine.recommend(sample_embeddings[0], k=5)
        assert len(recommendations) == 5

    @pytest.mark.parametrize("k", [1, 5, 10, 20])
    def test_different_k_values(self, engine_with_index, sample_embeddings, k):
        """Test different numbers of recommendations."""
        recommendations = engine_with_index.recommend(sample_embeddings[0], k=k)
        assert len(recommendations) == k


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_index_type_raises_error(self, sample_embeddings, sample_labels, sample_class_names):
        """Test that invalid index type raises error."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            use_pca=False,  # Disable PCA to test index type validation
            index_type="invalid"
        )

        with pytest.raises(ValueError, match="Unknown index_type"):
            engine.build_index()

    def test_n_components_larger_than_samples_raises_error(self, sample_labels, sample_class_names):
        """Test that n_components > min(n_samples, n_features) raises error from sklearn."""
        # Create embeddings with small dimension
        small_embeddings = np.random.randn(100, 32).astype('float32')

        engine = FashionRecommendationEngine(
            embeddings=small_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            n_components=64  # Larger than embedding dim (32)
        )

        # sklearn PCA raises ValueError when n_components > min(n_samples, n_features)
        with pytest.raises(ValueError, match="n_components"):
            engine.build_index()

    def test_engine_works_without_pca(self, sample_embeddings, sample_labels, sample_class_names):
        """Test that engine works correctly when PCA is disabled."""
        engine = FashionRecommendationEngine(
            embeddings=sample_embeddings,
            labels=sample_labels,
            class_names=sample_class_names,
            use_pca=False
        )
        engine.build_index()

        # Should use original embedding dimension
        assert engine.reduced_embeddings.shape[1] == sample_embeddings.shape[1]

        recommendations = engine.recommend(sample_embeddings[0], k=5)
        assert len(recommendations) == 5
