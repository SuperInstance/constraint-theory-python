#!/usr/bin/env python3
"""
Batch processing example for Constraint Theory Python bindings.

This example demonstrates efficient batch processing techniques for
processing large numbers of vectors.
"""

import time
from constraint_theory import PythagoreanManifold, generate_triples


def benchmark_snap_methods():
    """Compare individual snap vs batch processing."""
    print("=" * 60)
    print("Benchmark: Individual vs Batch Processing")
    print("=" * 60)
    
    manifold = PythagoreanManifold(density=200)
    
    # Generate test vectors
    test_vectors = []
    for _ in range(1000):
        # Generate random unit vectors
        import random
        import math
        angle = random.uniform(0, 2 * math.pi)
        test_vectors.append([math.cos(angle), math.sin(angle)])
    
    print(f"\n   Testing with {len(test_vectors)} vectors\n")
    
    # Method 1: Individual snaps
    start = time.time()
    individual_results = []
    for x, y in test_vectors:
        result = manifold.snap(x, y)
        individual_results.append(result)
    individual_time = time.time() - start
    
    # Method 2: Batch processing
    start = time.time()
    batch_results = manifold.snap_batch(test_vectors)
    batch_time = time.time() - start
    
    # Verify results are identical
    for i, (ind, batch) in enumerate(zip(individual_results, batch_results)):
        assert abs(ind[0] - batch[0]) < 1e-6, f"X mismatch at {i}"
        assert abs(ind[1] - batch[1]) < 1e-6, f"Y mismatch at {i}"
        assert abs(ind[2] - batch[2]) < 1e-6, f"Noise mismatch at {i}"
    
    print(f"   {'Method':<25} {'Time (ms)':>12} {'Vectors/sec':>15}")
    print(f"   {'-'*25} {'-'*12} {'-'*15}")
    print(f"   {'Individual snaps':<25} {individual_time*1000:>12.2f} "
          f"{len(test_vectors)/individual_time:>15,.0f}")
    print(f"   {'Batch processing':<25} {batch_time*1000:>12.2f} "
          f"{len(test_vectors)/batch_time:>15,.0f}")
    print(f"\n   Speedup: {individual_time/batch_time:.2f}x faster with batch processing")


def process_large_dataset():
    """Demonstrate processing a large dataset."""
    print("\n" + "=" * 60)
    print("Processing Large Dataset")
    print("=" * 60)
    
    manifold = PythagoreanManifold(density=500)
    
    # Simulate a large dataset
    dataset_size = 100000
    print(f"\n   Generating {dataset_size:,} vectors...")
    
    import random
    import math
    vectors = []
    for _ in range(dataset_size):
        angle = random.uniform(0, 2 * math.pi)
        vectors.append([math.cos(angle), math.sin(angle)])
    
    print(f"   Processing {dataset_size:,} vectors...\n")
    
    # Process in chunks for memory efficiency
    chunk_size = 10000
    all_results = []
    total_time = 0
    
    for i in range(0, len(vectors), chunk_size):
        chunk = vectors[i:i + chunk_size]
        
        start = time.time()
        results = manifold.snap_batch(chunk)
        chunk_time = time.time() - start
        total_time += chunk_time
        
        all_results.extend(results)
        
        if (i // chunk_size) % 2 == 0 or i + chunk_size >= len(vectors):
            print(f"   Processed {min(i + chunk_size, len(vectors)):>7,} / {dataset_size:,} "
                  f"({(i + len(chunk))/dataset_size*100:.1f}%) - "
                  f"Chunk time: {chunk_time*1000:.1f}ms")
    
    print(f"\n   Total processing time: {total_time:.2f}s")
    print(f"   Throughput: {dataset_size/total_time:,.0f} vectors/second")
    
    # Analyze results
    noise_values = [noise for _, _, noise in all_results]
    avg_noise = sum(noise_values) / len(noise_values)
    min_noise = min(noise_values)
    max_noise = max(noise_values)
    
    print(f"\n   Results summary:")
    print(f"     Average noise: {avg_noise:.6f}")
    print(f"     Min noise:     {min_noise:.6f}")
    print(f"     Max noise:     {max_noise:.6f}")


def streaming_processing():
    """Demonstrate streaming/chunked processing."""
    print("\n" + "=" * 60)
    print("Streaming/Chunked Processing")
    print("=" * 60)
    
    manifold = PythagoreanManifold(density=200)
    
    def vector_stream(n_chunks, chunk_size):
        """Simulate a stream of vectors."""
        import random
        import math
        for _ in range(n_chunks):
            chunk = []
            for _ in range(chunk_size):
                angle = random.uniform(0, 2 * math.pi)
                chunk.append([math.cos(angle), math.sin(angle)])
            yield chunk
    
    n_chunks = 10
    chunk_size = 1000
    
    print(f"\n   Processing {n_chunks} chunks of {chunk_size} vectors each\n")
    
    all_results = []
    for i, chunk in enumerate(vector_stream(n_chunks, chunk_size)):
        results = manifold.snap_batch(chunk)
        all_results.extend(results)
        print(f"   Chunk {i+1}/{n_chunks}: processed {len(chunk)} vectors")
    
    print(f"\n   Total vectors processed: {len(all_results)}")


def parallel_processing_simulation():
    """Demonstrate how to structure code for parallel processing."""
    print("\n" + "=" * 60)
    print("Parallel Processing Structure (Simulation)")
    print("=" * 60)
    
    # Note: Actual parallel processing would use multiprocessing or concurrent.futures
    # This example shows the structure
    
    import random
    import math
    
    manifold = PythagoreanManifold(density=200)
    
    def process_batch(vectors, batch_id):
        """Process a batch of vectors."""
        results = manifold.snap_batch(vectors)
        return batch_id, results
    
    # Simulate parallel workloads
    batches = []
    for i in range(4):
        batch = []
        for _ in range(500):
            angle = random.uniform(0, 2 * math.pi)
            batch.append([math.cos(angle), math.sin(angle)])
        batches.append((batch, i))
    
    print(f"\n   Simulating parallel processing of {len(batches)} batches\n")
    
    # Sequential simulation (in real code, use multiprocessing.Pool or similar)
    start = time.time()
    all_results = {}
    for batch, batch_id in batches:
        _, results = process_batch(batch, batch_id)
        all_results[batch_id] = results
        print(f"   Processed batch {batch_id}: {len(results)} vectors")
    
    total_time = time.time() - start
    total_vectors = sum(len(r) for r in all_results.values())
    
    print(f"\n   Total: {total_vectors} vectors in {total_time*1000:.2f}ms")
    print("""
   For actual parallel processing, consider:
   
   from concurrent.futures import ProcessPoolExecutor
   
   with ProcessPoolExecutor(max_workers=4) as executor:
       futures = [executor.submit(process_batch, batch, i) 
                  for batch, i in batches]
       results = [f.result() for f in futures]
   
   Note: Each process needs its own manifold instance.
   """)


def memory_efficient_processing():
    """Demonstrate memory-efficient processing techniques."""
    print("\n" + "=" * 60)
    print("Memory-Efficient Processing")
    print("=" * 60)
    
    manifold = PythagoreanManifold(density=200)
    
    # Generator for vectors
    def generate_vectors(count):
        """Generate vectors on-demand."""
        import random
        import math
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            yield [math.cos(angle), math.sin(angle)]
    
    # Process with minimal memory footprint
    total_vectors = 50000
    chunk_size = 5000
    
    print(f"\n   Processing {total_vectors:,} vectors in chunks of {chunk_size:,}\n")
    
    processed = 0
    total_noise = 0
    exact_matches = 0
    
    # Process chunk by chunk
    current_chunk = []
    for vector in generate_vectors(total_vectors):
        current_chunk.append(vector)
        
        if len(current_chunk) >= chunk_size:
            results = manifold.snap_batch(current_chunk)
            
            for _, _, noise in results:
                total_noise += noise
                if noise < 0.001:
                    exact_matches += 1
            
            processed += len(current_chunk)
            print(f"   Processed: {processed:>7,} / {total_vectors:,}")
            current_chunk = []
    
    # Process remaining vectors
    if current_chunk:
        results = manifold.snap_batch(current_chunk)
        for _, _, noise in results:
            total_noise += noise
            if noise < 0.001:
                exact_matches += 1
        processed += len(current_chunk)
    
    print(f"\n   Results:")
    print(f"     Total processed: {processed:,}")
    print(f"     Average noise:   {total_noise/processed:.6f}")
    print(f"     Exact matches:   {exact_matches:,} ({exact_matches/processed*100:.1f}%)")


def main():
    print("=" * 60)
    print("Constraint Theory - Batch Processing Examples")
    print("=" * 60)
    
    benchmark_snap_methods()
    process_large_dataset()
    streaming_processing()
    parallel_processing_simulation()
    memory_efficient_processing()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Batch Processing Best Practices:
1. Always use snap_batch() for multiple vectors - 2-5x faster
2. Process in chunks for large datasets (memory efficiency)
3. Use streaming/generators for very large datasets
4. Consider parallel processing for CPU-bound workloads
5. Profile with realistic data sizes before optimizing
    """)
    
    print("Done!")


if __name__ == "__main__":
    main()
