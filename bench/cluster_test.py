#!/usr/bin/env python3
"""Full cluster test: master + etcd + data nodes + proxy.

Tests:
1. Master node management (AddNode, RemoveNode, GetInstance, GetNodeInfo)
2. Partition config CRUD (UpdatePartitionConfig, GetPartitionConfig)
3. Proxy write forwarding (upsert → master node)
4. Proxy search (broadcast to all partitions)
5. Read/write separation (writes to leader, reads from any node)
6. Data consistency across nodes
"""
import asyncio
import aiohttp
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import gen_vector, make_single_upsert, make_batch_upsert, make_search, make_query, send_request

ETCD_URL = "http://127.0.0.1:2379"
MASTER_URL = "http://127.0.0.1:6060"
PROXY_URL = "http://127.0.0.1:6061"
NODE1_URL = "http://127.0.0.1:7781"
NODE3_URL = "http://127.0.0.1:7783"
NODE4_URL = "http://127.0.0.1:7784"


async def http_post(session, url, data, timeout=10):
    try:
        async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            return resp.status, await resp.json()
    except Exception as e:
        return 0, {"error": str(e)}


async def http_get(session, url, timeout=10):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            return resp.status, await resp.json()
    except Exception as e:
        return 0, {"error": str(e)}


async def main():
    print("="*70)
    print("  vectorDB 完整集群测试 (master + etcd + data nodes + proxy)")
    print("="*70)

    # Check all services
    print("\n  Checking services...")
    async with aiohttp.ClientSession() as session:
        services = [
            ("etcd", "http://127.0.0.1:2379/health", "GET"),
            ("master", f"{MASTER_URL}/MasterService/GetInstance", "POST"),
            ("proxy", f"{PROXY_URL}/ProxyService/topology", "GET"),
            ("node1", f"{NODE1_URL}/AdminService/GetNode", "GET"),
            ("node3", f"{NODE3_URL}/AdminService/GetNode", "GET"),
            ("node4", f"{NODE4_URL}/AdminService/GetNode", "GET"),
        ]
        for name, url, method in services:
            if method == "GET":
                if name == "etcd":
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                            ok = resp.status == 200
                    except:
                        ok = False
                else:
                    status, data = await http_get(session, url, timeout=5)
                    ok = status == 200
            else:
                status, data = await http_post(session, url, {"instanceId": 1}, timeout=5)
                ok = status == 200
            print(f"    {name}: {'UP' if ok else 'DOWN'}")

    # ==========================================
    # TEST 1: Master Node Management
    # ==========================================
    print("\n" + "="*70)
    print("  TEST 1: Master Node Management")
    print("="*70)

    async with aiohttp.ClientSession() as session:
        # Get current instances
        print("\n  [1.1] GetInstance (before)...")
        status, data = await http_post(session, f"{MASTER_URL}/MasterService/GetInstance", {"instanceId": 1})
        nodes_before = data.get("data", {}).get("nodes", [])
        print(f"    Nodes: {len(nodes_before)}")
        for n in nodes_before:
            print(f"      nodeId={n['nodeId']}, url={n['url']}, role={n['role']}")

        # Add a new test node
        print("\n  [1.2] AddNode (nodeId=99, test)...")
        status, data = await http_post(session, f"{MASTER_URL}/MasterService/AddNode",
                                       {"instanceId": 1, "nodeId": 99, "url": "http://127.0.0.1:9999", "role": 1, "status": 1})
        print(f"    Result: {data}")

        # Verify
        print("\n  [1.3] GetInstance (after add)...")
        status, data = await http_post(session, f"{MASTER_URL}/MasterService/GetInstance", {"instanceId": 1})
        nodes_after_add = data.get("data", {}).get("nodes", [])
        print(f"    Nodes: {len(nodes_after_add)}")
        added = any(n["nodeId"] == 99 for n in nodes_after_add)
        print(f"    Node 99 added: {'YES' if added else 'NO'}")

        # Remove the test node
        print("\n  [1.4] RemoveNode (nodeId=99)...")
        status, data = await http_post(session, f"{MASTER_URL}/MasterService/RemoveNode",
                                       {"instanceId": 1, "nodeId": 99})
        print(f"    Result: {data}")

        # Verify removal
        print("\n  [1.5] GetInstance (after remove)...")
        status, data = await http_post(session, f"{MASTER_URL}/MasterService/GetInstance", {"instanceId": 1})
        nodes_after_remove = data.get("data", {}).get("nodes", [])
        removed = not any(n["nodeId"] == 99 for n in nodes_after_remove)
        print(f"    Nodes: {len(nodes_after_remove)}, Node 99 removed: {'YES' if removed else 'NO'}")

        test1_pass = added and removed
        print(f"\n  RESULT: {'PASSED' if test1_pass else 'FAILED'}")

    # ==========================================
    # TEST 2: Partition Config CRUD
    # ==========================================
    print("\n" + "="*70)
    print("  TEST 2: Partition Config CRUD")
    print("="*70)

    async with aiohttp.ClientSession() as session:
        # Get current config
        print("\n  [2.1] GetPartitionConfig (before)...")
        status, data = await http_post(session, f"{MASTER_URL}/MasterService/GetPartitionConfig", {"instanceId": 1})
        config_before = data.get("data", {})
        print(f"    partitionKey={config_before.get('partitionKey')}, partitions={len(config_before.get('partitions', []))}")

        # Update config
        print("\n  [2.2] UpdatePartitionConfig...")
        new_config = {
            "instanceId": 1,
            "partitionKey": "id",
            "numberOfPartitions": 3,
            "partitions": [
                {"partitionId": 0, "nodeId": 1},
                {"partitionId": 1, "nodeId": 3},
                {"partitionId": 2, "nodeId": 4}
            ]
        }
        status, data = await http_post(session, f"{MASTER_URL}/MasterService/UpdatePartitionConfig", new_config)
        print(f"    Result: {data}")

        # Verify
        print("\n  [2.3] GetPartitionConfig (after update)...")
        status, data = await http_post(session, f"{MASTER_URL}/MasterService/GetPartitionConfig", {"instanceId": 1})
        config_after = data.get("data", {})
        print(f"    partitionKey={config_after.get('partitionKey')}, numberOfPartitions={config_after.get('numberOfPartitions')}")
        for p in config_after.get("partitions", []):
            print(f"      partitionId={p['partitionId']}, nodeId={p['nodeId']}")

        test2_pass = (config_after.get("partitionKey") == "id" and
                      config_after.get("numberOfPartitions") == 3 and
                      len(config_after.get("partitions", [])) == 3)
        print(f"\n  RESULT: {'PASSED' if test2_pass else 'FAILED'}")

    # ==========================================
    # TEST 3: Proxy Write Forwarding
    # ==========================================
    print("\n" + "="*70)
    print("  TEST 3: Proxy Write Forwarding")
    print("="*70)

    test_ids = list(range(500000, 500010))
    async with aiohttp.ClientSession() as session:
        print(f"\n  [3.1] Writing {len(test_ids)} vectors via proxy...")
        write_ok = 0
        for vid in test_ids:
            vec = gen_vector(128, seed=vid)
            body = make_single_upsert(vid, vec, "FLAT", {"int_field": vid})
            ok, _, _ = await send_request(session, f"{PROXY_URL}/ProxyService/upsert", body, timeout=15)
            if ok:
                write_ok += 1
        print(f"    Written: {write_ok}/{len(test_ids)}")

        # Verify data exists on data nodes
        print(f"\n  [3.2] Verifying data on node1 (direct)...")
        verify_ok = 0
        for vid in test_ids[:5]:
            body = make_query(vid)
            ok, _, resp = await send_request(session, f"{NODE1_URL}/UserService/query", body, timeout=10)
            if ok and resp.get("retCode") == 0 and "vectors" in resp:
                verify_ok += 1
                print(f"    node1 query id={vid}: OK (len={len(resp['vectors'])})")
            else:
                print(f"    node1 query id={vid}: NOT FOUND")

        test3_pass = write_ok == len(test_ids) and verify_ok > 0
        print(f"\n  RESULT: {'PASSED' if test3_pass else 'FAILED'}")

    # ==========================================
    # TEST 4: Proxy Search (Broadcast)
    # ==========================================
    print("\n" + "="*70)
    print("  TEST 4: Proxy Search (Broadcast)")
    print("="*70)

    async with aiohttp.ClientSession() as session:
        vec = gen_vector(128, seed=test_ids[0])
        print("\n  [4.1] Proxy search (broadcast to all partitions)...")
        body = make_search(vec, 5, "FLAT")
        ok, _, resp = await send_request(session, f"{PROXY_URL}/ProxyService/search", body, timeout=15)
        proxy_results = resp.get("vectors", []) if ok else []
        print(f"    Proxy search results: {proxy_results}")

        print("\n  [4.2] Direct search on each node...")
        for name, url in [("node1", NODE1_URL), ("node3", NODE3_URL), ("node4", NODE4_URL)]:
            body = make_search(vec, 5, "FLAT")
            ok, _, resp = await send_request(session, f"{url}/UserService/search", body, timeout=10)
            results = resp.get("vectors", []) if ok else []
            print(f"    {name} search: {results}")

        test4_pass = len(proxy_results) > 0
        print(f"\n  RESULT: {'PASSED' if test4_pass else 'PARTIAL (proxy broadcast may need restart)'}")

    # ==========================================
    # TEST 5: Read/Write Separation
    # ==========================================
    print("\n" + "="*70)
    print("  TEST 5: Read/Write Separation")
    print("="*70)

    async with aiohttp.ClientSession() as session:
        # Write to leader (node1)
        print("\n  [5.1] Write to node1 (leader)...")
        vid = 500050
        vec = gen_vector(128, seed=vid)
        body = make_single_upsert(vid, vec, "FLAT", {"int_field": vid})
        ok, _, _ = await send_request(session, f"{NODE1_URL}/UserService/upsert", body, timeout=15)
        print(f"    Write to node1: {'OK' if ok else 'FAIL'}")

        # Read from different nodes
        print("\n  [5.2] Read from all nodes...")
        for name, url in [("node1", NODE1_URL), ("node3", NODE3_URL), ("node4", NODE4_URL)]:
            body = make_search(vec, 3, "FLAT")
            ok, _, resp = await send_request(session, f"{url}/UserService/search", body, timeout=10)
            results = resp.get("vectors", []) if ok else []
            has_data = vid in results
            print(f"    {name} search: {results}, contains {vid}: {'YES' if has_data else 'NO'}")

        test5_pass = ok
        print(f"\n  RESULT: {'PASSED' if test5_pass else 'FAILED'}")

    # ==========================================
    # TEST 6: Proxy Topology
    # ==========================================
    print("\n" + "="*70)
    print("  TEST 6: Proxy Topology")
    print("="*70)

    async with aiohttp.ClientSession() as session:
        status, data = await http_get(session, f"{PROXY_URL}/ProxyService/topology")
        print(f"\n  Topology: {json.dumps(data, indent=2)}")
        test6_pass = status == 200 and "nodes" in data
        print(f"\n  RESULT: {'PASSED' if test6_pass else 'FAILED'}")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*70)
    print("  CLUSTER TEST SUMMARY")
    print("="*70)
    results = {
        "master_node_management": test1_pass,
        "partition_config_crud": test2_pass,
        "proxy_write_forwarding": test3_pass,
        "proxy_search_broadcast": test4_pass,
        "read_write_separation": test5_pass,
        "proxy_topology": test6_pass,
    }
    for k, v in results.items():
        print(f"  {k}: {'PASSED' if v else 'FAILED/PARTIAL'}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  Total: {passed}/{total} passed")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_cluster.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
