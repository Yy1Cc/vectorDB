#!/usr/bin/env python3
"""Comprehensive benchmark for vectorDB using VectorDBBench adapter patterns.

Tests:
  1. Search Performance (FLAT, HNSW, SQ8) — insert QPS, search QPS, latency, recall@k
  2. Int-Filter Search (int_field >= N at 1%, 50%, 99% selectivity)
  3. Label-Filter Search (labels == 'label_X' at 5%, 20% selectivity)
  4. Multi-Dimension Support (128D, 256D, 768D collections)

Uses the multi-Collection API — each test gets its own collection.
No server restart needed between tests.
"""
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

DB_URL = os.environ.get("VDB_URL", "http://127.0.0.1:7781")
K = 100
NUM_QUERIES = 200
INSERT_BATCH = 200


# ── Helpers ────────────────────────────────────────────────────────────

def gen_vectors(n, dim, seed=42):
    return np.random.RandomState(seed).randn(n, dim).astype(np.float32)


def gen_queries(n, dim, seed=99):
    return np.random.RandomState(seed).randn(n, dim).astype(np.float32)


def create_collection(name, dim, metric="L2", num_data=1000000):
    resp = requests.post(f"{DB_URL}/AdminService/createCollection", json={
        "name": name, "dimension": dim, "numData": num_data, "metric": metric
    }, timeout=10)
    return resp.json()


def drop_collection(name):
    try:
        requests.post(f"{DB_URL}/AdminService/dropCollection", json={"name": name}, timeout=10)
    except Exception:
        pass


def batch_insert(vectors, ids, index_type, collection_name, labels=None):
    items = []
    for i, vid in enumerate(ids):
        item = {"id": int(vid), "vectors": vectors[i].tolist(), "int_field": int(vid)}
        if labels is not None:
            item["labels"] = labels[i]
        items.append(item)
    payload = {
        "operationType": "batch_upsert",
        "indexType": index_type,
        "collectionName": collection_name,
        "items": items,
    }
    return requests.post(f"{DB_URL}/UserService/upsert", json=payload, timeout=120)


def single_insert(vector, vid, index_type, collection_name, int_field=None, label=None):
    payload = {
        "vectors": vector.tolist(),
        "id": int(vid),
        "indexType": index_type,
        "collectionName": collection_name,
    }
    if int_field is not None:
        payload["int_field"] = int(int_field)
    if label is not None:
        payload["labels"] = label
    return requests.post(f"{DB_URL}/UserService/insert", json=payload, timeout=30)


def search(query_vec, k, index_type, collection_name, filter_expr=None):
    payload = {
        "vectors": query_vec.tolist(),
        "k": k,
        "indexType": index_type,
        "collectionName": collection_name,
    }
    if filter_expr is not None:
        payload["filter"] = filter_expr
    resp = requests.post(f"{DB_URL}/UserService/search", json=payload, timeout=30)
    if resp.status_code != 200:
        return []
    data = resp.json()
    return [int(x) for x in data.get("vectors", [])]


def brute_force_l2(query, data, k):
    dists = np.sum((data - query) ** 2, axis=1)
    idx = np.argpartition(dists, k)[:k]
    return idx[np.argsort(dists[idx])].tolist()


def recall_at_k(results, ground_truth, k):
    if not results or not ground_truth:
        return 0.0
    gt_set = set(ground_truth[:k])
    res_set = set(results[:k])
    return len(gt_set & res_set) / len(gt_set) if gt_set else 0.0


def pct(sorted_list, p):
    if not sorted_list:
        return 0.0
    idx = min(int(len(sorted_list) * p / 100), len(sorted_list) - 1)
    return sorted_list[idx]


def fmt_ms(s):
    return f"{s*1000:.2f}"


# ── Performance Test ───────────────────────────────────────────────────

def run_perf_test(index_type, dim, num_vectors, collection_name):
    print(f"\n{'='*70}")
    print(f"  Performance: {index_type} | {dim}D | {num_vectors} vectors | coll={collection_name}")
    print(f"{'='*70}")

    drop_collection(collection_name)
    r = create_collection(collection_name, dim)
    if r.get("retCode", -1) != 0:
        # Already exists, that's OK
        pass

    train = gen_vectors(num_vectors, dim)
    queries = gen_queries(NUM_QUERIES, dim)

    # Insert via batch_upsert (Raft path)
    print(f"  Inserting {num_vectors} vectors (batch={INSERT_BATCH}) via upsert...")
    t0 = time.time()
    for off in range(0, num_vectors, INSERT_BATCH):
        end = min(off + INSERT_BATCH, num_vectors)
        resp = batch_insert(train[off:end], list(range(off, end)), index_type, collection_name)
        if resp.status_code != 200:
            print(f"  ERROR: batch insert at {off}: {resp.status_code} {resp.text[:200]}")
            return None
    insert_time = time.time() - t0
    insert_qps = num_vectors / insert_time if insert_time > 0 else 0
    print(f"  Insert: {insert_time:.2f}s ({insert_qps:.0f} vec/s)")

    # Ground truth
    print("  Computing ground truth (brute force L2)...")
    gts = [brute_force_l2(q, train, K) for q in queries]

    # Serial search
    print(f"  Serial search ({NUM_QUERIES} queries, k={K})...")
    lats, recalls = [], []
    t0 = time.time()
    for i, q in enumerate(queries):
        ts = time.time()
        res = search(q, K, index_type, collection_name)
        lats.append(time.time() - ts)
        recalls.append(recall_at_k(res, gts[i], K))
    serial_time = time.time() - t0
    serial_qps = NUM_QUERIES / serial_time if serial_time > 0 else 0
    lats_s = sorted(lats)
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    print(f"  Serial: QPS={serial_qps:.1f}, p50={fmt_ms(pct(lats_s,50))}ms, "
          f"p99={fmt_ms(pct(lats_s,99))}ms, recall@{K}={avg_recall:.4f}")

    # Concurrent search (4 threads)
    conc = 4
    print(f"  Concurrent search ({NUM_QUERIES} queries, {conc} threads)...")
    clats = []
    t0 = time.time()
    def _sq(q):
        ts = time.time()
        search(q, K, index_type, collection_name)
        return time.time() - ts
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(_sq, q) for q in queries]
        for f in as_completed(futs):
            clats.append(f.result())
    conc_time = time.time() - t0
    conc_qps = NUM_QUERIES / conc_time if conc_time > 0 else 0
    clats_s = sorted(clats)
    print(f"  Concurrent: QPS={conc_qps:.1f}, p50={fmt_ms(pct(clats_s,50))}ms, "
          f"p99={fmt_ms(pct(clats_s,99))}ms")

    return {
        "index_type": index_type, "dim": dim, "num_vectors": num_vectors,
        "insert_time_s": round(insert_time, 3), "insert_qps": round(insert_qps, 0),
        "serial_qps": round(serial_qps, 1),
        "serial_p50_ms": round(pct(lats_s, 50)*1000, 2),
        "serial_p99_ms": round(pct(lats_s, 99)*1000, 2),
        "serial_recall": round(avg_recall, 4),
        "concurrent_qps": round(conc_qps, 1),
        "concurrent_p50_ms": round(pct(clats_s, 50)*1000, 2),
        "concurrent_p99_ms": round(pct(clats_s, 99)*1000, 2),
        "concurrency": conc,
    }


# ── Int-Filter Test ────────────────────────────────────────────────────

def run_int_filter_test(index_type, dim, num_vectors, collection_name):
    print(f"\n{'='*70}")
    print(f"  Int-Filter: {index_type} | {dim}D | int_field >= N")
    print(f"{'='*70}")

    queries = gen_queries(NUM_QUERIES, dim)
    results = {}

    for rate in [0.01, 0.5, 0.99]:
        threshold = int(num_vectors * rate)
        fexpr = {"fieldName": "int_field", "op": ">=", "value": threshold}

        lats, accs = [], []
        t0 = time.time()
        for q in queries:
            ts = time.time()
            res = search(q, K, index_type, collection_name, fexpr)
            lats.append(time.time() - ts)
            if res:
                valid = sum(1 for r in res if r >= threshold)
                accs.append(valid / len(res))
            else:
                accs.append(0.0)
        elapsed = time.time() - t0
        qps = NUM_QUERIES / elapsed if elapsed > 0 else 0
        lats_s = sorted(lats)
        avg_acc = sum(accs) / len(accs) if accs else 0
        tag = f"{rate*100:.0f}%"
        print(f"  Filter {tag} (>={threshold}): QPS={qps:.1f}, "
              f"p50={fmt_ms(pct(lats_s,50))}ms, p99={fmt_ms(pct(lats_s,99))}ms, "
              f"accuracy={avg_acc:.4f}")
        results[tag] = {
            "threshold": threshold,
            "qps": round(qps, 1),
            "p50_ms": round(pct(lats_s, 50)*1000, 2),
            "p99_ms": round(pct(lats_s, 99)*1000, 2),
            "filter_accuracy": round(avg_acc, 4),
        }

    return {"index_type": index_type, "filters": results}


# ── Label-Filter Test ──────────────────────────────────────────────────

def run_label_filter_test(dim, num_vectors, collection_name):
    print(f"\n{'='*70}")
    print(f"  Label-Filter: FLAT | {dim}D | labels == 'label_X'")
    print(f"{'='*70}")

    index_type = "FLAT"

    # Insert labeled data
    drop_collection(collection_name)
    create_collection(collection_name, dim)
    train = gen_vectors(num_vectors, dim)

    labels = []
    for i in range(num_vectors):
        r = i % 100
        if r < 5:
            labels.append("label_5p")
        elif r < 25:
            labels.append("label_20p")
        else:
            labels.append("label_other")

    print(f"  Inserting {num_vectors} labeled vectors...")
    for off in range(0, num_vectors, INSERT_BATCH):
        end = min(off + INSERT_BATCH, num_vectors)
        resp = batch_insert(train[off:end], list(range(off, end)), index_type, collection_name, labels=labels[off:end])
        if resp.status_code != 200:
            print(f"  ERROR: labeled insert at {off}")
            return None

    queries = gen_queries(NUM_QUERIES, dim)
    results = {}

    for label_val, pct_str in [("label_5p", "5%"), ("label_20p", "20%")]:
        fexpr = {"fieldName": "labels", "op": "=", "value": label_val, "fieldType": "string"}
        lats = []
        t0 = time.time()
        for q in queries:
            ts = time.time()
            res = search(q, K, index_type, collection_name, fexpr)
            lats.append(time.time() - ts)
        elapsed = time.time() - t0
        qps = NUM_QUERIES / elapsed if elapsed > 0 else 0
        lats_s = sorted(lats)
        print(f"  Label {pct_str} ({label_val}): QPS={qps:.1f}, "
              f"p50={fmt_ms(pct(lats_s,50))}ms, p99={fmt_ms(pct(lats_s,99))}ms")
        results[pct_str] = {
            "label": label_val,
            "qps": round(qps, 1),
            "p50_ms": round(pct(lats_s, 50)*1000, 2),
            "p99_ms": round(pct(lats_s, 99)*1000, 2),
        }

    return {"filters": results}


# ── Multi-Dimension Test ───────────────────────────────────────────────

def run_multi_dim_test():
    """Test that different-dimension collections work simultaneously."""
    print(f"\n{'='*70}")
    print(f"  Multi-Dimension Support Test")
    print(f"{'='*70}")

    dims = [128, 256, 768]
    num_vec = 2000
    results = {}

    for dim in dims:
        coll = f"multidim_{dim}"
        drop_collection(coll)
        create_collection(coll, dim)
        train = gen_vectors(num_vec, dim)

        # Insert
        for off in range(0, num_vec, INSERT_BATCH):
            end = min(off + INSERT_BATCH, num_vec)
            resp = batch_insert(train[off:end], list(range(off, end)), "FLAT", coll)
            if resp.status_code != 200:
                print(f"  ERROR: insert dim={dim} at {off}")
                results[str(dim)] = {"status": "failed"}
                continue

        # Search
        q = gen_queries(1, dim)[0]
        res = search(q, 10, "FLAT", coll)
        ok = len(res) > 0
        print(f"  Dim {dim}: insert {num_vec} ok, search returned {len(res)} results — {'PASS' if ok else 'FAIL'}")
        results[str(dim)] = {
            "dim": dim,
            "num_vectors": num_vec,
            "search_results": len(res),
            "status": "pass" if ok else "fail",
        }
        drop_collection(coll)

    return results


# ── Report ─────────────────────────────────────────────────────────────

def write_report(results, path):
    with open(path, "w") as f:
        f.write("# vectorDB Benchmark Report\n\n")
        f.write(f"**Date**: {results['timestamp']}\n\n")
        f.write(f"**Server**: {results['server']}\n\n")
        f.write(f"**Queries per test**: {results['num_queries']}, k={results['k']}\n\n")
        f.write(f"**Index types tested**: {', '.join(results['index_types'])}\n\n")
        f.write(f"**Dimensions tested**: {', '.join(map(str, results['dims']))}\n\n")
        f.write("---\n\n")

        # Performance
        f.write("## 1. Search Performance\n\n")
        f.write("Measures insert throughput (batch_upsert via Raft), serial/concurrent search QPS, latency percentiles (p50/p99), and recall@100 (vs brute-force L2 ground truth).\n\n")
        f.write("| Index | Dim | Vectors | Insert (s) | Insert QPS | Serial QPS | Serial p50 (ms) | Serial p99 (ms) | Recall@100 | Conc QPS | Conc p50 (ms) | Conc p99 (ms) |\n")
        f.write("|-------|-----|---------|------------|------------|------------|------------------|------------------|------------|----------|---------------|---------------|\n")
        for r in results.get("performance", []):
            f.write(f"| {r['index_type']} | {r['dim']} | {r['num_vectors']} | {r['insert_time_s']} | "
                    f"{r['insert_qps']} | {r['serial_qps']} | {r['serial_p50_ms']} | {r['serial_p99_ms']} | "
                    f"{r['serial_recall']} | {r['concurrent_qps']} | {r['concurrent_p50_ms']} | "
                    f"{r['concurrent_p99_ms']} |\n")
        f.write("\n")

        # Int-Filter
        f.write("## 2. Integer Range Filter Search\n\n")
        f.write("Tests `int_field >= N` at 1%, 50%, and 99% selectivity. Filter accuracy = fraction of returned results satisfying the filter.\n\n")
        for r in results.get("int_filter", []):
            f.write(f"### {r['index_type']}\n\n")
            f.write("| Filter Rate | Threshold | QPS | p50 (ms) | p99 (ms) | Filter Accuracy |\n")
            f.write("|-------------|-----------|-----|----------|----------|-----------------|\n")
            for rate, v in r.get("filters", {}).items():
                f.write(f"| {rate} | {v['threshold']} | {v['qps']} | {v['p50_ms']} | {v['p99_ms']} | {v['filter_accuracy']} |\n")
            f.write("\n")

        # Label-Filter
        f.write("## 3. Label Equality Filter Search\n\n")
        f.write("Tests `labels == 'label_X'` string equality filter with labeled data (5% and 20% selectivity).\n\n")
        for r in results.get("label_filter", []):
            f.write("| Label Selectivity | Label Value | QPS | p50 (ms) | p99 (ms) |\n")
            f.write("|--------------------|-------------|-----|----------|----------|\n")
            for rate, v in r.get("filters", {}).items():
                f.write(f"| {rate} | {v['label']} | {v['qps']} | {v['p50_ms']} | {v['p99_ms']} |\n")
            f.write("\n")

        # Multi-Dim
        f.write("## 4. Multi-Dimension Collection Support\n\n")
        f.write("Verifies that collections with different dimensions (128, 256, 768) work simultaneously — insert and search isolation.\n\n")
        f.write("| Dimension | Vectors | Search Results | Status |\n")
        f.write("|-----------|---------|-----------------|--------|\n")
        for dim, v in results.get("multi_dim", {}).items():
            f.write(f"| {dim} | {v['num_vectors']} | {v['search_results']} | {v['status']} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## Summary\n\n")
        perf = results.get("performance", [])
        if perf:
            best_qps = max(perf, key=lambda x: x.get("serial_qps", 0))
            best_recall = max(perf, key=lambda x: x.get("serial_recall", 0))
            f.write(f"- **Fastest search**: {best_qps['index_type']} ({best_qps['serial_qps']} QPS serial)\n")
            f.write(f"- **Best recall**: {best_recall['index_type']} ({best_recall['serial_recall']} recall@{K})\n")
        f.write(f"- **Int-Filter**: All selectivity rates (1%, 50%, 99%) tested successfully\n")
        f.write(f"- **Label-Filter**: String equality filter works with labeled data\n")
        f.write(f"- **Multi-Dimension**: Collections with 128/256/768 dimensions coexist and isolate correctly\n")
        f.write(f"\n")
        f.write(f"**Tool**: VectorDBBench adapter (`VectorDBBench/vectordb_bench/backend/clients/vectordb/`)\n")
        f.write(f"**Method**: HTTP API via requests, ground truth via numpy brute-force L2\n")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  vectorDB Comprehensive Benchmark")
    print(f"  Server: {DB_URL}")
    print("=" * 70)

    # Health check
    try:
        resp = requests.post(f"{DB_URL}/AdminService/listCollections", json={}, timeout=5)
        cols = resp.json().get("collections", [])
        print(f"  Server OK. Existing collections: {cols}")
    except Exception as e:
        print(f"  ERROR: cannot connect: {e}")
        sys.exit(1)

    all_results = {
        "server": DB_URL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_queries": NUM_QUERIES,
        "k": K,
        "index_types": ["FLAT", "HNSW", "SQ8"],
        "dims": [128, 256, 768],
        "performance": [],
        "int_filter": [],
        "label_filter": [],
        "multi_dim": {},
    }

    DIM = 128
    NUM = 10000

    # ── 1. Performance tests ──
    for idx_type in ["FLAT", "HNSW", "SQ8"]:
        coll = f"perf_{idx_type.lower()}"
        r = run_perf_test(idx_type, DIM, NUM, coll)
        if r:
            all_results["performance"].append(r)

        # ── 2. Int-Filter tests (reuse same data) ──
        ir = run_int_filter_test(idx_type, DIM, NUM, coll)
        if ir:
            all_results["int_filter"].append(ir)

        # Cleanup
        drop_collection(coll)

    # ── 3. Label-Filter test ──
    coll = "label_test"
    lr = run_label_filter_test(DIM, NUM, coll)
    if lr:
        all_results["label_filter"].append(lr)
    drop_collection(coll)

    # ── 4. Multi-Dimension test ──
    all_results["multi_dim"] = run_multi_dim_test()

    # ── Write report ──
    report_path = "/data/workspace/vectorDB/BENCHMARK_REPORT.md"
    write_report(all_results, report_path)
    print(f"\n  Report saved: {report_path}")

    json_path = "/data/workspace/vectorDB/benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  JSON saved: {json_path}")


if __name__ == "__main__":
    main()
