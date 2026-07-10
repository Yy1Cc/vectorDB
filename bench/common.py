#!/usr/bin/env python3
"""Common utilities for vectorDB benchmarking."""
import asyncio
import aiohttp
import numpy as np
import time
import json
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7781
DEFAULT_DIM = 128


def base_url(host=DEFAULT_HOST, port=DEFAULT_PORT):
    return f"http://{host}:{port}"


def gen_vector(dim=DEFAULT_DIM, seed=None):
    """Generate a random float vector."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.rand(dim).astype(np.float32).tolist()
    return np.random.rand(dim).astype(np.float32).tolist()


def gen_vectors(n, dim=DEFAULT_DIM, seed=None):
    """Generate n random vectors as a numpy array."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.rand(n, dim).astype(np.float32)
    return np.random.rand(n, dim).astype(np.float32)


def make_single_upsert(vid, vector, index_type="FLAT", extra_fields=None):
    """Build a single upsert request body."""
    body = {"vectors": vector, "id": vid, "indexType": index_type}
    if extra_fields:
        body.update(extra_fields)
    return body


def make_batch_upsert(start_id, vectors, index_type="FLAT", extra_fields_fn=None):
    """Build a batch upsert request body.
    
    Args:
        start_id: starting ID for the batch
        vectors: list of vector arrays
        index_type: FLAT or HNSW
        extra_fields_fn: optional callable(id) -> dict of extra fields
    """
    items = []
    for i, vec in enumerate(vectors):
        item = {"id": start_id + i, "vectors": vec if isinstance(vec, list) else vec.tolist()}
        if extra_fields_fn:
            item.update(extra_fields_fn(start_id + i))
        items.append(item)
    return {"operationType": "batch_upsert", "indexType": index_type, "items": items}


def make_single_insert(vid, vector, index_type="FLAT"):
    """Build a single insert request body (direct index, no Raft/WAL)."""
    return {"vectors": vector, "id": vid, "indexType": index_type}


def make_search(vector, k=10, index_type="FLAT", filter_field=None, filter_op=None, filter_value=None):
    """Build a search request body."""
    body = {"vectors": vector, "k": k, "indexType": index_type}
    if filter_field and filter_op is not None and filter_value is not None:
        body["filter"] = {"fieldName": filter_field, "op": filter_op, "value": filter_value}
    return body


def make_query(vid):
    """Build a query request body."""
    return {"id": vid}


@dataclass
class BenchResult:
    name: str
    total_requests: int
    success_count: int
    fail_count: int
    latencies_ms: List[float] = field(default_factory=list)
    total_time_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def qps(self):
        return self.total_requests / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def throughput_ops_s(self):
        return self.success_count / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def p50(self):
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p95(self):
        if not self.latencies_ms:
            return 0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def p99(self):
        if not self.latencies_ms:
            return 0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.99)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def avg_ms(self):
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    @property
    def min_ms(self):
        return min(self.latencies_ms) if self.latencies_ms else 0

    @property
    def max_ms(self):
        return max(self.latencies_ms) if self.latencies_ms else 0

    def summary(self):
        return {
            "name": self.name,
            "total_requests": self.total_requests,
            "success": self.success_count,
            "fail": self.fail_count,
            "total_time_s": round(self.total_time_s, 3),
            "qps": round(self.qps, 1),
            "throughput_ops_s": round(self.throughput_ops_s, 1),
            "latency_avg_ms": round(self.avg_ms, 3),
            "latency_p50_ms": round(self.p50, 3),
            "latency_p95_ms": round(self.p95, 3),
            "latency_p99_ms": round(self.p99, 3),
            "latency_min_ms": round(self.min_ms, 3),
            "latency_max_ms": round(self.max_ms, 3),
            "extra": self.extra,
        }


async def send_request(session, url, body, timeout=30):
    """Send a single POST request and return (success, latency_ms, response_json)."""
    start = time.perf_counter()
    try:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            latency_ms = (time.perf_counter() - start) * 1000
            if resp.status == 200:
                data = await resp.json()
                if data.get("retCode") == 0:
                    return True, latency_ms, data
                else:
                    return False, latency_ms, data
            else:
                text = await resp.text()
                return False, latency_ms, {"error": text, "status": resp.status}
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, {"error": str(e)}


async def run_concurrent(url, bodies, concurrency=64):
    """Run requests concurrently with a semaphore-limited concurrency.
    
    Returns BenchResult.
    """
    result = BenchResult(
        name="concurrent",
        total_requests=len(bodies),
        success_count=0,
        fail_count=0,
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(session, body):
        async with semaphore:
            ok, lat, _ = await send_request(session, url, body)
            result.latencies_ms.append(lat)
            if ok:
                result.success_count += 1
            else:
                result.fail_count += 1

    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(worker(session, b)) for b in bodies]
        await asyncio.gather(*tasks)
    result.total_time_s = time.perf_counter() - start
    return result


async def run_sequential(url, bodies):
    """Run requests sequentially (one at a time).
    
    Returns BenchResult.
    """
    result = BenchResult(
        name="sequential",
        total_requests=len(bodies),
        success_count=0,
        fail_count=0,
    )
    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        for body in bodies:
            ok, lat, _ = await send_request(session, url, body)
            result.latencies_ms.append(lat)
            if ok:
                result.success_count += 1
            else:
                result.fail_count += 1
    result.total_time_s = time.perf_counter() - start
    return result


async def wait_for_service(host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=10):
    """Wait until the service is responding."""
    url = f"http://{host}:{port}/AdminService/GetNode"
    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        while time.perf_counter() - start < timeout:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return False


def print_result(result: BenchResult):
    """Print a formatted result summary."""
    s = result.summary()
    print(f"\n{'='*60}")
    print(f"  {s['name']}")
    print(f"{'='*60}")
    print(f"  Requests:    {s['total_requests']} (success={s['success']}, fail={s['fail']})")
    print(f"  Total time:  {s['total_time_s']}s")
    print(f"  QPS:         {s['qps']}")
    print(f"  Throughput:  {s['throughput_ops_s']} ops/s")
    print(f"  Latency avg: {s['latency_avg_ms']} ms")
    print(f"  Latency p50: {s['latency_p50_ms']} ms")
    print(f"  Latency p95: {s['latency_p95_ms']} ms")
    print(f"  Latency p99: {s['latency_p99_ms']} ms")
    print(f"  Latency min: {s['latency_min_ms']} ms")
    print(f"  Latency max: {s['latency_max_ms']} ms")
    if s['extra']:
        for k, v in s['extra'].items():
            print(f"  {k}: {v}")
    print(f"{'='*60}")
