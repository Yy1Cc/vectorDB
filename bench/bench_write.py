#!/usr/bin/env python3
"""Benchmark write throughput: single upsert vs batch upsert vs insert, FLAT vs HNSW."""
import asyncio
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    base_url, gen_vector, gen_vectors, make_single_upsert, make_batch_upsert,
    make_single_insert, run_concurrent, run_sequential, BenchResult, print_result, DEFAULT_DIM
)

URL = base_url()


async def bench_single_upsert_concurrent(num=1000, dim=DEFAULT_DIM, index_type="FLAT", concurrency=32, start_id=1):
    """Single upsert with concurrent requests."""
    bodies = []
    for i in range(num):
        bodies.append(make_single_upsert(start_id + i, gen_vector(dim, seed=i), index_type, {"int_field": i}))
    result = await run_concurrent(f"{URL}/UserService/upsert", bodies, concurrency)
    result.name = f"single_upsert_concurrent [{index_type}] n={num} c={concurrency} dim={dim}"
    return result


async def bench_batch_upsert(num=10000, batch_size=100, dim=DEFAULT_DIM, index_type="FLAT", concurrency=8, start_id=1):
    """Batch upsert with concurrent requests."""
    total_batches = num // batch_size
    bodies = []
    for b in range(total_batches):
        sid = start_id + b * batch_size
        vecs = gen_vectors(batch_size, dim, seed=b)
        vectors_list = [vecs[i].tolist() for i in range(batch_size)]
        bodies.append(make_batch_upsert(sid, vectors_list, index_type,
                                        extra_fields_fn=lambda x: {"int_field": x}))
    result = await run_concurrent(f"{URL}/UserService/upsert", bodies, concurrency)
    result.name = f"batch_upsert_concurrent [{index_type}] n={num} batch={batch_size} c={concurrency} dim={dim}"
    result.extra["batch_size"] = batch_size
    result.extra["total_batches"] = total_batches
    return result


async def bench_batch_upsert_sequential(num=10000, batch_size=100, dim=DEFAULT_DIM, index_type="FLAT", start_id=1):
    """Batch upsert sequential (no concurrency)."""
    total_batches = num // batch_size
    bodies = []
    for b in range(total_batches):
        sid = start_id + b * batch_size
        vecs = gen_vectors(batch_size, dim, seed=b)
        vectors_list = [vecs[i].tolist() for i in range(batch_size)]
        bodies.append(make_batch_upsert(sid, vectors_list, index_type,
                                        extra_fields_fn=lambda x: {"int_field": x}))
    result = await run_sequential(f"{URL}/UserService/upsert", bodies)
    result.name = f"batch_upsert_sequential [{index_type}] n={num} batch={batch_size} dim={dim}"
    result.extra["batch_size"] = batch_size
    result.extra["total_batches"] = total_batches
    return result


async def bench_insert_concurrent(num=1000, dim=DEFAULT_DIM, index_type="FLAT", concurrency=64, start_id=1):
    """Direct insert (no Raft/WAL, pure index performance)."""
    bodies = []
    for i in range(num):
        bodies.append(make_single_insert(start_id + i, gen_vector(dim, seed=i), index_type))
    result = await run_concurrent(f"{URL}/UserService/insert", bodies, concurrency)
    result.name = f"insert_concurrent [{index_type}] n={num} c={concurrency} dim={dim} (no Raft/WAL)"
    return result


async def main():
    results = []
    dim = DEFAULT_DIM

    print("\n" + "="*70)
    print("  WRITE THROUGHPUT BENCHMARKS")
    print("="*70)

    # --- FLAT index (IDs: 1 - 20000) ---
    print("\n>>> Testing FLAT single upsert (n=500, c=32)...")
    r = await bench_single_upsert_concurrent(500, dim, "FLAT", 32, start_id=1)
    print_result(r)
    results.append(r)

    print("\n>>> Testing FLAT batch upsert concurrent (n=10000, batch=100, c=8)...")
    r = await bench_batch_upsert(10000, 100, dim, "FLAT", 8, start_id=10000)
    print_result(r)
    results.append(r)

    print("\n>>> Testing FLAT batch upsert sequential (n=10000, batch=100)...")
    r = await bench_batch_upsert_sequential(10000, 100, dim, "FLAT", start_id=20000)
    print_result(r)
    results.append(r)

    # --- HNSW index (IDs: 100000 - 140000, distinct from FLAT) ---
    print("\n>>> Testing HNSW single upsert (n=500, c=32)...")
    r = await bench_single_upsert_concurrent(500, dim, "HNSW", 32, start_id=100000)
    print_result(r)
    results.append(r)

    print("\n>>> Testing HNSW batch upsert concurrent (n=10000, batch=100, c=8)...")
    r = await bench_batch_upsert(10000, 100, dim, "HNSW", 8, start_id=110000)
    print_result(r)
    results.append(r)

    print("\n>>> Testing HNSW batch upsert sequential (n=10000, batch=100)...")
    r = await bench_batch_upsert_sequential(10000, 100, dim, "HNSW", start_id=120000)
    print_result(r)
    results.append(r)

    # --- Insert (no Raft, IDs: 200000+, separate index space) ---
    print("\n>>> Testing FLAT insert (no Raft, n=1000, c=64)...")
    r = await bench_insert_concurrent(1000, dim, "FLAT", 64, start_id=200000)
    print_result(r)
    results.append(r)

    print("\n>>> Testing HNSW insert (no Raft, n=1000, c=64)...")
    r = await bench_insert_concurrent(1000, dim, "HNSW", 64, start_id=300000)
    print_result(r)
    results.append(r)

    # Save results
    output = [r.summary() for r in results]
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_write.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
