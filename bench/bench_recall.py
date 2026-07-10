#!/usr/bin/env python3
"""Benchmark HNSW recall rate vs FLAT (ground truth) for different k values.

Uses FLAT search results as ground truth (exact search) and compares HNSW
approximate search results against them. Both indices search the same data.
"""
import asyncio
import sys
import os
import json
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    base_url, gen_vector, gen_vectors, make_batch_upsert,
    make_search, send_request, BenchResult, print_result, DEFAULT_DIM
)
import aiohttp

URL = base_url()


async def load_data_via_insert(num=5000, dim=DEFAULT_DIM, index_type="FLAT", start_id=1):
    """Load data via insert API (direct index, no Raft/WAL/scalar_storage)."""
    print(f"  Loading {num} vectors [{index_type}] dim={dim} via insert API...")
    batch_size = 100
    total_batches = num // batch_size
    async with aiohttp.ClientSession() as session:
        for b in range(total_batches):
            sid = start_id + b * batch_size
            vecs = gen_vectors(batch_size, dim, seed=b + 30000)
            for i in range(batch_size):
                vid = sid + i
                vec = vecs[i].tolist()
                body = {"vectors": vec, "id": vid, "indexType": index_type}
                await send_request(session, f"{URL}/UserService/insert", body, timeout=30)
    print(f"  Data loading complete: {num} vectors loaded.")


def compute_recall(ground_truth_ids, approx_ids):
    """Compute recall rate: |truth ∩ approx| / |truth|."""
    if not ground_truth_ids:
        return 0.0
    truth_set = set(ground_truth_ids)
    approx_set = set(approx_ids)
    intersection = truth_set & approx_set
    return len(intersection) / len(truth_set)


async def main():
    dim = DEFAULT_DIM
    num_data = 5000
    num_queries = 200
    k_values = [1, 10, 50, 100]
    start_id = 600000  # High ID range to minimize interference

    print("\n" + "="*70)
    print("  RECALL RATE BENCHMARKS (HNSW vs FLAT)")
    print("="*70)

    # Load the SAME vectors into both FLAT and HNSW indices
    await load_data_via_insert(num_data, dim, "FLAT", start_id=start_id)
    await load_data_via_insert(num_data, dim, "HNSW", start_id=start_id)

    # Generate query vectors (use some of the loaded vectors as queries)
    query_vectors = []
    for i in np.random.choice(num_data, num_queries, replace=False):
        vecs = gen_vectors(1, dim, seed=int(i // 100 + 30000))
        query_vectors.append(vecs[0].tolist())

    results = {}
    async with aiohttp.ClientSession() as session:
        for k in k_values:
            flat_recalls = []
            hnsw_recalls = []
            flat_latencies = []
            hnsw_latencies = []

            for qv in query_vectors:
                # FLAT search = ground truth (exact search)
                body_flat = make_search(qv, k, "FLAT")
                t0 = time.perf_counter()
                ok_flat, _, resp_flat = await send_request(session, f"{URL}/UserService/search", body_flat)
                flat_lat = (time.perf_counter() - t0) * 1000
                flat_ids = resp_flat.get("vectors", []) if ok_flat else []
                flat_latencies.append(flat_lat)

                # HNSW search = approximate
                body_hnsw = make_search(qv, k, "HNSW")
                t0 = time.perf_counter()
                ok_hnsw, _, resp_hnsw = await send_request(session, f"{URL}/UserService/search", body_hnsw)
                hnsw_lat = (time.perf_counter() - t0) * 1000
                hnsw_ids = resp_hnsw.get("vectors", []) if ok_hnsw else []
                hnsw_latencies.append(hnsw_lat)

                # Recall = how many HNSW results match FLAT results
                hnsw_recalls.append(compute_recall(flat_ids, hnsw_ids))
                # FLAT recall vs itself should be 100%
                flat_recalls.append(compute_recall(flat_ids, flat_ids))

            avg_flat_recall = np.mean(flat_recalls)
            avg_hnsw_recall = np.mean(hnsw_recalls)
            avg_flat_lat = np.mean(flat_latencies)
            avg_hnsw_lat = np.mean(hnsw_latencies)

            print(f"\n  k={k}:")
            print(f"    FLAT recall: {avg_flat_recall:.4f} ({avg_flat_recall*100:.2f}%) | avg latency: {avg_flat_lat:.3f} ms")
            print(f"    HNSW recall: {avg_hnsw_recall:.4f} ({avg_hnsw_recall*100:.2f}%) | avg latency: {avg_hnsw_lat:.3f} ms")
            if avg_hnsw_lat > 0:
                print(f"    Speedup (FLAT/HNSW): {avg_flat_lat/avg_hnsw_lat:.2f}x")

            results[f"k={k}"] = {
                "k": k,
                "flat_recall": round(float(avg_flat_recall), 4),
                "hnsw_recall": round(float(avg_hnsw_recall), 4),
                "flat_avg_latency_ms": round(float(avg_flat_lat), 3),
                "hnsw_avg_latency_ms": round(float(avg_hnsw_lat), 3),
                "speedup": round(float(avg_flat_lat / avg_hnsw_lat), 2) if avg_hnsw_lat > 0 else 0,
                "num_queries": num_queries,
                "num_data": num_data,
            }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_recall.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
