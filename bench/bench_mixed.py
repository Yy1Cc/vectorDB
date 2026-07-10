#!/usr/bin/env python3
"""Benchmark mixed read/write workload: 80% search / 20% upsert."""
import asyncio
import sys
import os
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    base_url, gen_vector, gen_vectors, make_search, make_batch_upsert,
    make_single_upsert, send_request, BenchResult, print_result, DEFAULT_DIM
)
import aiohttp

URL = base_url()


async def load_initial_data(num=5000, dim=DEFAULT_DIM, index_type="FLAT", start_id=1):
    """Load initial data before mixed workload."""
    print(f"  Loading {num} initial vectors [{index_type}]...")
    batch_size = 200
    total_batches = num // batch_size
    async with aiohttp.ClientSession() as session:
        for b in range(total_batches):
            sid = start_id + b * batch_size
            vecs = gen_vectors(batch_size, dim, seed=b + 20000)
            vectors_list = [vecs[i].tolist() for i in range(batch_size)]
            body = make_batch_upsert(sid, vectors_list, index_type,
                                     extra_fields_fn=lambda x: {"int_field": x % 100})
            await send_request(session, f"{URL}/UserService/upsert", body, timeout=120)
    print(f"  Initial data loaded.")


async def bench_mixed(total_ops=5000, read_ratio=0.8, dim=DEFAULT_DIM,
                      index_type="FLAT", concurrency=64, existing_ids=5000, start_write_id=1):
    """Run mixed read/write workload.
    
    Args:
        total_ops: total number of operations
        read_ratio: fraction of operations that are searches
        existing_ids: number of existing IDs to search against
        start_write_id: starting ID for new writes
    """
    semaphore = asyncio.Semaphore(concurrency)
    latencies_read = []
    latencies_write = []
    read_ok = 0
    read_fail = 0
    write_ok = 0
    write_fail = 0
    next_write_id = start_write_id

    async with aiohttp.ClientSession() as session:
        async def worker(op_type, body):
            nonlocal read_ok, read_fail, write_ok, write_fail
            async with semaphore:
                ok, lat, _ = await send_request(session, f"{URL}/UserService/{op_type}", body)
                if op_type == "search":
                    latencies_read.append(lat)
                    if ok:
                        read_ok += 1
                    else:
                        read_fail += 1
                else:
                    latencies_write.append(lat)
                    if ok:
                        write_ok += 1
                    else:
                        write_fail += 1

        tasks = []
        for i in range(total_ops):
            if random.random() < read_ratio:
                # Search: use a random existing vector
                vid = random.randint(1, existing_ids)
                vec = gen_vector(dim, seed=vid)
                body = make_search(vec, k=10, index_type=index_type)
                tasks.append(asyncio.create_task(worker("search", body)))
            else:
                # Upsert: new single vector
                vec = gen_vector(dim, seed=next_write_id)
                body = make_single_upsert(next_write_id, vec, index_type, {"int_field": next_write_id % 100})
                next_write_id += 1
                tasks.append(asyncio.create_task(worker("upsert", body)))

        start = asyncio.get_event_loop().time()
        await asyncio.gather(*tasks)
        total_time = asyncio.get_event_loop().time() - start

    result = BenchResult(
        name=f"mixed_read_write [{index_type}] ops={total_ops} read={int(read_ratio*100)}% c={concurrency}",
        total_requests=total_ops,
        success_count=read_ok + write_ok,
        fail_count=read_fail + write_fail,
        total_time_s=total_time,
    )
    result.extra = {
        "read_ops": read_ok + read_fail,
        "write_ops": write_ok + write_fail,
        "read_qps": round((read_ok + read_fail) / total_time, 1),
        "write_qps": round((write_ok + write_fail) / total_time, 1),
        "read_latency_avg_ms": round(sum(latencies_read)/len(latencies_read), 3) if latencies_read else 0,
        "read_latency_p50_ms": round(sorted(latencies_read)[len(latencies_read)//2], 3) if latencies_read else 0,
        "read_latency_p95_ms": round(sorted(latencies_read)[int(len(latencies_read)*0.95)], 3) if latencies_read else 0,
        "write_latency_avg_ms": round(sum(latencies_write)/len(latencies_write), 3) if latencies_write else 0,
        "write_latency_p50_ms": round(sorted(latencies_write)[len(latencies_write)//2], 3) if latencies_write else 0,
        "write_latency_p95_ms": round(sorted(latencies_write)[int(len(latencies_write)*0.95)], 3) if latencies_write else 0,
    }
    return result


async def main():
    results = []
    dim = DEFAULT_DIM

    print("\n" + "="*70)
    print("  MIXED READ/WRITE BENCHMARKS")
    print("="*70)

    # FLAT mixed workload (IDs: 1-5000 existing)
    await load_initial_data(5000, dim, "FLAT", start_id=1)

    print("\n>>> Testing FLAT mixed 80% read / 20% write (ops=5000, c=64)...")
    r = await bench_mixed(5000, 0.8, dim, "FLAT", 64, 5000, start_write_id=10000)
    print_result(r)
    results.append(r)

    print("\n>>> Testing FLAT mixed 50% read / 50% write (ops=5000, c=64)...")
    r = await bench_mixed(5000, 0.5, dim, "FLAT", 64, 5000, start_write_id=15000)
    print_result(r)
    results.append(r)

    # HNSW mixed workload (IDs: 100000-105000 existing, distinct from FLAT)
    await load_initial_data(5000, dim, "HNSW", start_id=100000)

    print("\n>>> Testing HNSW mixed 80% read / 20% write (ops=5000, c=64)...")
    r = await bench_mixed(5000, 0.8, dim, "HNSW", 64, 5000, start_write_id=110000)
    print_result(r)
    results.append(r)

    print("\n>>> Testing HNSW mixed 50% read / 50% write (ops=5000, c=64)...")
    r = await bench_mixed(5000, 0.5, dim, "HNSW", 64, 5000, start_write_id=115000)
    print_result(r)
    results.append(r)

    # Save results
    output = [r.summary() for r in results]
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_mixed.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
