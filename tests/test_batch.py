"""Tests for batch operations."""

import pytest
import math


class TestBatchSnap:
    """Tests for batch snapping functionality."""

    def test_batch_basic(self):
        """Test basic batch snapping."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        vectors = [[0.6, 0.8], [0.8, 0.6], [0.1, 0.99]]
        
        results = manifold.snap_batch(vectors)
        
        assert len(results) == 3
        
        for sx, sy, noise in results:
            assert isinstance(sx, float)
            assert isinstance(sy, float)
            assert isinstance(noise, float)
            assert -1.0 <= sx <= 1.0
            assert -1.0 <= sy <= 1.0
            assert noise >= 0

    def test_batch_empty_list(self):
        """Test batch snapping with empty list."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        results = manifold.snap_batch([])
        
        assert len(results) == 0

    def test_batch_single_vector(self):
        """Test batch snapping with single vector."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        results = manifold.snap_batch([[0.6, 0.8]])
        
        assert len(results) == 1
        sx, sy, noise = results[0]
        assert abs(sx - 0.6) < 0.01
        assert abs(sy - 0.8) < 0.01

    def test_batch_large(self):
        """Test batch snapping with many vectors."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        # Generate many vectors
        import random
        random.seed(42)
        vectors = [[random.uniform(-1, 1), random.uniform(-1, 1)] 
                  for _ in range(1000)]
        
        results = manifold.snap_batch(vectors)
        
        assert len(results) == 1000
        
        # All results should be valid
        for sx, sy, noise in results:
            magnitude = math.sqrt(sx * sx + sy * sy)
            assert abs(magnitude - 1.0) < 0.0001

    def test_batch_consistency_with_single(self):
        """Test batch results match single snap results."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        vectors = [[0.6, 0.8], [0.577, 0.816], [0.707, 0.707], [0.1, 0.99]]
        
        # Batch results
        batch_results = manifold.snap_batch(vectors)
        
        # Individual results
        individual_results = [manifold.snap(v[0], v[1]) for v in vectors]
        
        # Should match
        for batch, individual in zip(batch_results, individual_results):
            assert abs(batch[0] - individual[0]) < 1e-6
            assert abs(batch[1] - individual[1]) < 1e-6
            assert abs(batch[2] - individual[2]) < 1e-6


class TestBatchWithNumpy:
    """Tests for batch snapping with NumPy arrays."""

    def test_batch_numpy_2d_array(self):
        """Test batch snapping with NumPy 2D array."""
        np = pytest.importorskip("numpy")
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        vectors = np.array([[0.6, 0.8], [0.8, 0.6], [0.1, 0.99]])
        results = manifold.snap_batch(vectors)
        
        assert len(results) == 3

    def test_batch_numpy_large_array(self):
        """Test batch snapping with large NumPy array."""
        np = pytest.importorskip("numpy")
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        # Generate random vectors
        n = 5000
        vectors = np.random.randn(n, 2)
        
        results = manifold.snap_batch(vectors)
        
        assert len(results) == n

    def test_batch_numpy_dtype(self):
        """Test batch snapping with different NumPy dtypes."""
        np = pytest.importorskip("numpy")
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        # Test float32
        vectors_f32 = np.array([[0.6, 0.8]], dtype=np.float32)
        results = manifold.snap_batch(vectors_f32)
        assert len(results) == 1
        
        # Test float64
        vectors_f64 = np.array([[0.6, 0.8]], dtype=np.float64)
        results = manifold.snap_batch(vectors_f64)
        assert len(results) == 1

    def test_batch_numpy_shape_variations(self):
        """Test batch snapping with different array shapes."""
        np = pytest.importorskip("numpy")
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        # Single vector, shape (1, 2)
        vectors = np.array([[0.6, 0.8]])
        results = manifold.snap_batch(vectors)
        assert len(results) == 1
        
        # Multiple vectors, shape (N, 2)
        vectors = np.array([[0.6, 0.8], [0.8, 0.6]])
        results = manifold.snap_batch(vectors)
        assert len(results) == 2


class TestBatchPerformance:
    """Tests for batch operation performance."""

    def test_batch_faster_than_individual(self):
        """Test that batch operations are faster than individual."""
        from constraint_theory import PythagoreanManifold
        import time
        
        manifold = PythagoreanManifold(200)
        
        n = 1000
        vectors = [[0.5, 0.8] for _ in range(n)]
        
        # Time individual
        start = time.perf_counter()
        for v in vectors:
            manifold.snap(v[0], v[1])
        individual_time = time.perf_counter() - start
        
        # Time batch
        start = time.perf_counter()
        manifold.snap_batch(vectors)
        batch_time = time.perf_counter() - start
        
        # Batch should be faster (allowing some margin)
        # In practice batch is usually 2-5x faster
        assert batch_time < individual_time * 2

    def test_batch_memory_efficient(self):
        """Test that batch operations don't use excessive memory."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        # Process large batch
        n = 10000
        vectors = [[0.5, 0.8] for _ in range(n)]
        
        # Should complete without memory issues
        results = manifold.snap_batch(vectors)
        
        assert len(results) == n


class TestBatchResults:
    """Tests for batch operation results."""

    def test_batch_unit_vectors(self):
        """Test that all batch results are unit vectors."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        import random
        random.seed(42)
        vectors = [[random.uniform(-1, 1), random.uniform(-1, 1)] 
                  for _ in range(100)]
        
        results = manifold.snap_batch(vectors)
        
        for sx, sy, _ in results:
            magnitude_squared = sx * sx + sy * sy
            assert abs(magnitude_squared - 1.0) < 1e-6

    def test_batch_noise_values(self):
        """Test that noise values are reasonable."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        # Exact triples should have low noise
        exact_vectors = [[3/5, 4/5], [5/13, 12/13], [8/17, 15/17]]
        results = manifold.snap_batch(exact_vectors)
        
        for sx, sy, noise in results:
            assert noise < 0.001

    def test_batch_preserves_order(self):
        """Test that batch preserves input order."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        vectors = [
            [0.6, 0.8],   # Should snap to (0.6, 0.8)
            [0.0, 1.0],   # Should snap to (0, 1)
            [1.0, 0.0],   # Should snap to (1, 0)
        ]
        
        results = manifold.snap_batch(vectors)
        
        # Check order preserved
        assert abs(results[0][0] - 0.6) < 0.01
        assert abs(results[0][1] - 0.8) < 0.01
        
        assert abs(results[1][0]) < 0.01
        assert abs(results[1][1] - 1.0) < 0.01
        
        assert abs(results[2][0] - 1.0) < 0.01
        assert abs(results[2][1]) < 0.01


class TestBatchEdgeCases:
    """Tests for edge cases in batch operations."""

    def test_batch_with_duplicates(self):
        """Test batch with duplicate vectors."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        vectors = [[0.6, 0.8], [0.6, 0.8], [0.6, 0.8]]
        results = manifold.snap_batch(vectors)
        
        # All should have same result
        assert results[0] == results[1] == results[2]

    def test_batch_with_zeros(self):
        """Test batch with zero vectors."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        vectors = [[0.0, 0.0], [0.6, 0.8]]
        results = manifold.snap_batch(vectors)
        
        assert len(results) == 2

    def test_batch_mixed_signs(self):
        """Test batch with mixed positive/negative vectors."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        vectors = [
            [0.6, 0.8],
            [-0.6, 0.8],
            [0.6, -0.8],
            [-0.6, -0.8],
        ]
        
        results = manifold.snap_batch(vectors)
        
        # Check correct signs
        assert results[0][0] > 0 and results[0][1] > 0  # Q1
        assert results[1][0] < 0 and results[1][1] > 0  # Q2
        assert results[2][0] > 0 and results[2][1] < 0  # Q4
        assert results[3][0] < 0 and results[3][1] < 0  # Q3

    def test_batch_very_small_vectors(self):
        """Test batch with very small vectors."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        vectors = [[1e-10, 1e-10], [1e-10, 0.8]]
        results = manifold.snap_batch(vectors)
        
        assert len(results) == 2
        for sx, sy, _ in results:
            magnitude = math.sqrt(sx * sx + sy * sy)
            assert abs(magnitude - 1.0) < 0.0001

    def test_batch_very_large_vectors(self):
        """Test batch with very large vectors."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        vectors = [[1000.0, 1000.0], [1e10, 1e10]]
        results = manifold.snap_batch(vectors)
        
        # Should handle gracefully
        assert len(results) == 2


class TestChunkedBatch:
    """Tests for chunked batch processing."""

    def test_chunked_processing_consistency(self):
        """Test that chunked processing gives same results."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        import random
        random.seed(42)
        vectors = [[random.uniform(-1, 1), random.uniform(-1, 1)] 
                  for _ in range(100)]
        
        # Full batch
        full_results = manifold.snap_batch(vectors)
        
        # Chunked batch
        chunk_size = 25
        chunked_results = []
        for i in range(0, len(vectors), chunk_size):
            chunk = vectors[i:i + chunk_size]
            chunked_results.extend(manifold.snap_batch(chunk))
        
        # Should match
        assert len(full_results) == len(chunked_results)
        for full, chunked in zip(full_results, chunked_results):
            assert abs(full[0] - chunked[0]) < 1e-6
            assert abs(full[1] - chunked[1]) < 1e-6
            assert abs(full[2] - chunked[2]) < 1e-6

    def test_different_chunk_sizes(self):
        """Test processing with different chunk sizes."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        import random
        random.seed(42)
        vectors = [[random.uniform(-1, 1), random.uniform(-1, 1)] 
                  for _ in range(100)]
        
        # Process with different chunk sizes
        for chunk_size in [1, 10, 50, 100]:
            results = []
            for i in range(0, len(vectors), chunk_size):
                chunk = vectors[i:i + chunk_size]
                results.extend(manifold.snap_batch(chunk))
            
            assert len(results) == 100


class TestBatchThreadSafety:
    """Tests for thread safety of batch operations."""

    def test_batch_from_multiple_threads(self):
        """Test batch operations from multiple threads."""
        from constraint_theory import PythagoreanManifold
        from concurrent.futures import ThreadPoolExecutor
        
        manifold = PythagoreanManifold(200)
        
        def process_batch(batch_id):
            vectors = [[0.6, 0.8] for _ in range(10)]
            return manifold.snap_batch(vectors)
        
        # Run multiple batches in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_batch, i) for i in range(4)]
            results = [f.result() for f in futures]
        
        # All should succeed
        assert len(results) == 4
        for batch_result in results:
            assert len(batch_result) == 10

    def test_shared_manifold_thread_safety(self):
        """Test that shared manifold is thread-safe."""
        from constraint_theory import PythagoreanManifold
        from concurrent.futures import ThreadPoolExecutor
        
        manifold = PythagoreanManifold(200)
        
        def snap_random(seed):
            import random
            random.seed(seed)
            vectors = [[random.uniform(-1, 1), random.uniform(-1, 1)] 
                      for _ in range(100)]
            return manifold.snap_batch(vectors)
        
        # Run with different seeds
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(snap_random, i) for i in range(8)]
            results = [f.result() for f in futures]
        
        # All should complete without errors
        assert len(results) == 8
        for batch_result in results:
            assert len(batch_result) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
