#!/usr/bin/env python3
"""Benchmark search latency and QPS: FLAT vs HNSW, different k, filter vs no-filter."""
import asyncio
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    base_url, gen_vector, gen_vectors, make_search, make_batch_upsert,
    run_concurrent, BenchResult, print_result, send_request, DEFAULT_DIM
)
import aiohttp

URL = base_url()


async def load_data(num=10000, dim=DEFAULT_DIM, index_type="FLAT", start_id=1):
    """Load data via batch upsert before search benchmarks."""
    print(f"  Loading {num} vectors [{index_type}] dim={dim}...")
    batch_size = 200
    total_batches = num // batch_size
    async with aiohttp.ClientSession() as session:
        for b in range(total_batches):
            sid = start_id + b * batch_size
            vecs = gen_vectors(batch_size, dim, seed=b + 10000)
            vectors_list = [vecs[i].tolist() for i in range(batch_size)]
            body = make_batch_upsert(sid, vectors_list, index_type,
                                     extra_fields_fn=lambda x: {"int_field": x % 100})
            ok, _, _ = await send_request(session, f"{URL}/UserService/upsert", body, timeout=120)
            if not ok:
                print(f"  WARNING: batch {b} failed")
    print(f"  Data loading complete.")


async def bench_search_concurrent(num_queries=1000, k=10, dim=DEFAULT_DIM, index_type="FLAT",
                                  concurrency=64, use_filter=False):
    """Search with concurrent requests."""
    bodies = []
    for i in range(num_queries):
        vec = gen_vector(dim, seed=i + 50000)
        if use_filter:
            bodies.append(make_search(vec, k, index_type, "int_field", "=", i % 100))
        else:
            bodies.append(make_search(vec, k, index_type))
    filter_tag = " +filter" if use_filter else ""
    result = await run_concurrent(f"{URL}/UserService/search", bodies, concurrency)
    result.name = f"search_concurrent [{index_type}] n={num_queries} k={k} c={concurrency}{filter_tag}"
    return result


async def main():
    results = []
    dim = DEFAULT_DIM
    num_data = 10000

    print("\n" + "="*70)
    print("  SEARCH LATENCY & QPS BENCHMARKS")
    print("="*70)

    # --- FLAT search (IDs: 1-10000) ---
    await load_data(num_data, dim, "FLAT", start_id=1)

    for k in [1, 10, 50]:
        print(f"\n>>> Testing FLAT search (n=1000, k={k}, c=64)...")
        r = await bench_search_concurrent(1000, k, dim, "FLAT", 64)
        print_result(r)
        results.append(r)

    print(f"\n>>> Testing FLAT search with filter (n=1000, k=10, c=64)...")
    r = await bench_search_concurrent(1000, 10, dim, "FLAT", 64, use_filter=True)
    print_result(r)
    results.append(r)

    # --- HNSW search (IDs: 100000-110000, distinct from FLAT) ---
    await load_data(num_data, dim, "HNSW", start_id=100000)

    for k in [1, 10, 50]:
        print(f"\n>>> Testing HNSW search (n=1000, k={k}, c=64)...")
        r = await bench_search_concurrent(1000, k, dim, "HNSW", 64)
        print_result(r)
        results.append(r)

    print(f"\n>>> Testing HNSW search with filter (n=1000, k=10, c=64)...")
    r = await bench_search_concurrent(1000, 10, dim, "HNSW", 64, use_filter=True)
    print_result(r)
    results.append(r)

    # Save results
    output = [r.summary() for r in results]
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_search.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
