#!/usr/bin/env python3
"""Raft distributed test: 2-node cluster replication and consistency.

Uses node2 as leader (clean WAL data) and node3 as follower.
Node1 is skipped due to old WAL data from benchmarks.

Prerequisites:
  - Node 2 running on port 7782 (raft 8082)
  - Node 3 running on port 7783 (raft 8083)

Usage:
  python3 raft_test.py
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

LEADER_URL = "http://127.0.0.1:7784"  # node4 as leader (clean data, new binary)
FOLLOWER_URL = "http://127.0.0.1:7783"  # node3 as follower
FOLLOWER_RAFT_ENDPOINT = "127.0.0.1:8083"
FOLLOWER_NODE_ID = 3


async def http_get(session, url, timeout=10):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            return resp.status, await resp.json()
    except Exception as e:
        return 0, {"error": str(e)}


async def wait_for_node(url, name, timeout=15):
    start = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(f"{url}/AdminService/GetNode", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        print(f"  {name} is up.")
                        return True
            except Exception:
                pass
            await asyncio.sleep(1)
    print(f"  {name} is NOT responding.")
    return False


async def get_node_info(session, url):
    status, data = await http_get(session, f"{url}/AdminService/GetNode")
    return data


async def list_nodes(session, url):
    status, data = await http_get(session, f"{url}/AdminService/ListNode")
    return data


async def set_leader(session, url):
    status, data = await http_get(session, f"{url}/AdminService/SetLeader")
    return status, data


async def add_follower(session, url, endpoint, node_id):
    body = {"endpoint": endpoint, "nodeId": node_id}
    try:
        async with session.post(f"{url}/AdminService/AddFollower", json=body, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            return resp.status, await resp.json()
    except Exception as e:
        return 0, {"error": str(e)}


async def upsert_to_leader(session, vid, vector, index_type="FLAT"):
    body = make_single_upsert(vid, vector, index_type, {"int_field": vid})
    ok, _, resp = await send_request(session, f"{LEADER_URL}/UserService/upsert", body, timeout=30)
    return ok


async def query_from_node(session, url, vid):
    body = make_query(vid)
    ok, _, resp = await send_request(session, f"{url}/UserService/query", body, timeout=10)
    return ok, resp


async def search_from_node(session, url, vector, k=5, index_type="FLAT"):
    body = make_search(vector, k, index_type)
    ok, _, resp = await send_request(session, f"{url}/UserService/search", body, timeout=10)
    return ok, resp


async def main():
    print("="*70)
    print("  vectorDB Raft 分布式测试 (2节点: node4=leader, node3=follower)")
    print("="*70)

    # Check nodes
    print("\n  Checking node availability...")
    leader_ok = await wait_for_node(LEADER_URL, "node2(leader)", timeout=10)
    follower_ok = await wait_for_node(FOLLOWER_URL, "node3(follower)", timeout=10)

    if not (leader_ok and follower_ok):
        print("\n  ERROR: Nodes are not running!")
        print("  Start nodes with:")
        print("    cd /data/workspace/vectorDB/build && export VECTORDB_CODE_BASE=/data/workspace/vectorDB")
        print("    ./bin/vdb_server 2 &  ./bin/vdb_server 3 &")
        sys.exit(1)

    # Form cluster
    print("\n  Forming Raft cluster...")
    async with aiohttp.ClientSession() as session:
        print("  Setting node2 as leader (election timeout: 10-20s)...")
        status, data = await set_leader(session, LEADER_URL)
        print(f"  SetLeader: {status} {data}")

        # Wait for election to complete (EnableElectionTimeout sets 10-20s)
        print("  Waiting 25s for election to complete...")
        for i in range(5):
            await asyncio.sleep(5)
            info = await get_node_info(session, LEADER_URL)
            info_str = json.dumps(info).lower()
            is_leader = "leader" in info_str and "true" in info_str if "error" not in info_str else False
            print(f"    [{(i+1)*5}s] node2 state: {info.get('node', {}).get('state', 'unknown')}")
            if is_leader:
                print("    node2 is now leader!")
                break

        # Verify node2 is leader
        info = await get_node_info(session, LEADER_URL)
        print(f"\n  node2 status: {json.dumps(info, indent=2)}")

        # Retry AddFollower
        print("\n  Adding node3 as follower...")
        for attempt in range(3):
            status, data = await add_follower(session, LEADER_URL, FOLLOWER_RAFT_ENDPOINT, FOLLOWER_NODE_ID)
            print(f"  AddFollower node3 (attempt {attempt+1}): {status} {data}")
            if status == 200 and data.get("retCode") == 0:
                break
            await asyncio.sleep(5)

        # List nodes
        print("\n  Cluster nodes:")
        nodes = await list_nodes(session, LEADER_URL)
        print(f"  {json.dumps(nodes, indent=2)}")

    # ==========================================
    # TEST 1: Data Replication
    # ==========================================
    print("\n" + "="*70)
    print("  TEST 1: Data Replication (leader → follower)")
    print("="*70)

    test_ids = list(range(700000, 700020))
    test_vectors = {}
    async with aiohttp.ClientSession() as session:
        print(f"\n  [1.1] Writing {len(test_ids)} vectors to leader (node2)...")
        write_ok = 0
        for vid in test_ids:
            vec = gen_vector(128, seed=vid)
            test_vectors[vid] = vec
            ok = await upsert_to_leader(session, vid, vec, "FLAT")
            if ok:
                write_ok += 1
            else:
                print(f"    FAIL: upsert id={vid}")
        print(f"    Written: {write_ok}/{len(test_ids)}")

        print(f"\n  [1.2] Waiting 3s for replication...")
        await asyncio.sleep(3)

        print(f"\n  [1.3] Querying data from follower (node3)...")
        replicate_ok = 0
        for vid in test_ids[:10]:
            ok, resp = await query_from_node(session, FOLLOWER_URL, vid)
            if ok and resp.get("retCode") == 0 and "vectors" in resp:
                print(f"    node3 query id={vid}: OK (len={len(resp['vectors'])})")
                replicate_ok += 1
            else:
                print(f"    node3 query id={vid}: NOT FOUND ({resp})")

        print(f"\n  Replication: {replicate_ok}/10 queries found on follower")

        # ==========================================
        # TEST 2: Search Consistency
        # ==========================================
        print("\n" + "="*70)
        print("  TEST 2: Search Consistency (leader vs follower)")
        print("="*70)

        print("\n  [2.1] Searching from leader and follower with same query...")
        consistency_ok = 0
        for vid in test_ids[:5]:
            vec = test_vectors[vid]
            ok_l, resp_l = await search_from_node(session, LEADER_URL, vec, k=5, index_type="FLAT")
            ok_f, resp_f = await search_from_node(session, FOLLOWER_URL, vec, k=5, index_type="FLAT")
            l_ids = resp_l.get("vectors", []) if ok_l else []
            f_ids = resp_f.get("vectors", []) if ok_f else []
            match = set(l_ids) == set(f_ids) if l_ids and f_ids else False
            print(f"    query id={vid}: leader={l_ids}, follower={f_ids}, match={match}")
            if match:
                consistency_ok += 1

        print(f"\n  Search consistency: {consistency_ok}/5 queries matched")

        # ==========================================
        # TEST 3: Batch Write Replication
        # ==========================================
        print("\n" + "="*70)
        print("  TEST 3: Batch Write Replication")
        print("="*70)

        print("\n  [3.1] Batch upsert 50 vectors to leader...")
        batch_ids = list(range(700100, 700150))
        batch_vecs = [gen_vector(128, seed=i) for i in batch_ids]
        items = [{"id": bid, "vectors": batch_vecs[i], "int_field": bid} for i, bid in enumerate(batch_ids)]
        batch_body = {"operationType": "batch_upsert", "indexType": "FLAT", "items": items}
        ok, _, resp = await send_request(session, f"{LEADER_URL}/UserService/upsert", batch_body, timeout=30)
        print(f"    Batch upsert: {'OK' if ok else 'FAIL'} ({resp})")

        print("\n  [3.2] Waiting 3s for replication...")
        await asyncio.sleep(3)

        print("\n  [3.3] Querying batch data from follower...")
        batch_replicate_ok = 0
        for vid in batch_ids[:10]:
            ok, resp = await query_from_node(session, FOLLOWER_URL, vid)
            if ok and resp.get("retCode") == 0 and "vectors" in resp:
                batch_replicate_ok += 1
        print(f"    Batch replication: {batch_replicate_ok}/10 queries found on follower")

        # ==========================================
        # TEST 4: Node Status
        # ==========================================
        print("\n" + "="*70)
        print("  TEST 4: Cluster Node Status")
        print("="*70)

        print("\n  [4.1] Leader node status:")
        leader_info = await get_node_info(session, LEADER_URL)
        print(f"  {json.dumps(leader_info, indent=2)}")

        print("\n  [4.2] Follower node status:")
        follower_info = await get_node_info(session, FOLLOWER_URL)
        print(f"  {json.dumps(follower_info, indent=2)}")

        print("\n  [4.3] Cluster node list:")
        nodes = await list_nodes(session, LEADER_URL)
        print(f"  {json.dumps(nodes, indent=2)}")

    # Summary
    print("\n" + "="*70)
    print("  RAFT TEST SUMMARY")
    print("="*70)
    results = {
        "write_to_leader": f"{write_ok}/{len(test_ids)}",
        "replication_to_follower": f"{replicate_ok}/10",
        "search_consistency": f"{consistency_ok}/5",
        "batch_replication": f"{batch_replicate_ok}/10",
        "cluster_nodes": len(nodes.get("nodes", [])) if isinstance(nodes, dict) else 0,
    }
    for k, v in results.items():
        print(f"  {k}: {v}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_raft.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\n  NOTE: Leader failover test requires manually killing the leader process.")
    print("  To test failover manually:")
    print("    1. kill the node2 process (leader)")
    print("    2. Wait 5s for election")
    print("    3. Check if node3 becomes leader: curl http://127.0.0.1:7783/AdminService/GetNode")
    print("    4. Write data to new leader (node3)")
    print("    5. Restart node2: cd /data/workspace/vectorDB/build && ./bin/vdb_server 2")
    print("    6. Verify data sync: curl -X POST http://127.0.0.1:7782/UserService/query -d '{\"id\":700000}'")


if __name__ == "__main__":
    asyncio.run(main())
