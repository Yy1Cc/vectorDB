#!/usr/bin/env python3
"""One-click runner: run all benchmarks and generate charts + summary report."""
import asyncio
import subprocess
import sys
import os
import json
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name):
    """Run a benchmark script and return exit code."""
    path = os.path.join(BENCH_DIR, script_name)
    print(f"\n{'#'*70}")
    print(f"# Running: {script_name}")
    print(f"{'#'*70}")
    start = time.time()
    result = subprocess.run([sys.executable, path], cwd=BENCH_DIR)
    elapsed = time.time() - start
    print(f"\n  {script_name} finished in {elapsed:.1f}s (exit code: {result.returncode})")
    return result.returncode


def generate_report():
    """Generate a consolidated markdown report from all result JSON files."""
    report = []
    report.append("# vectorDB 性能压测报告\n")
    report.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**向量维度**: 128\n")
    report.append(f"**测试环境**: 8核 CPU, TencentOS Server 4.4\n\n")

    # Write results
    write_data = None
    write_path = os.path.join(BENCH_DIR, "results_write.json")
    if os.path.exists(write_path):
        with open(write_path) as f:
            write_data = json.load(f)
        report.append("## 1. 写入吞吐量测试\n\n")
        report.append("| 测试场景 | 请求数 | 成功 | QPS | p50(ms) | p95(ms) | p99(ms) |\n")
        report.append("|----------|--------|------|-----|---------|---------|---------|\n")
        for r in write_data:
            report.append(f"| {r['name']} | {r['total_requests']} | {r['success']} | {r['qps']} | {r['latency_p50_ms']} | {r['latency_p95_ms']} | {r['latency_p99_ms']} |\n")
        report.append("\n")

    # Search results
    search_data = None
    search_path = os.path.join(BENCH_DIR, "results_search.json")
    if os.path.exists(search_path):
        with open(search_path) as f:
            search_data = json.load(f)
        report.append("## 2. 搜索延迟与 QPS 测试\n\n")
        report.append("| 测试场景 | 请求数 | QPS | p50(ms) | p95(ms) | p99(ms) | avg(ms) |\n")
        report.append("|----------|--------|-----|---------|---------|---------|---------|\n")
        for r in search_data:
            report.append(f"| {r['name']} | {r['total_requests']} | {r['qps']} | {r['latency_p50_ms']} | {r['latency_p95_ms']} | {r['latency_p99_ms']} | {r['latency_avg_ms']} |\n")
        report.append("\n")

    # Recall results
    recall_path = os.path.join(BENCH_DIR, "results_recall.json")
    if os.path.exists(recall_path):
        with open(recall_path) as f:
            recall_data = json.load(f)
        report.append("## 3. HNSW 召回率测试\n\n")
        report.append("| k值 | FLAT召回率 | HNSW召回率 | FLAT延迟(ms) | HNSW延迟(ms) | 加速比 |\n")
        report.append("|-----|-----------|-----------|-------------|-------------|--------|\n")
        for key, r in recall_data.items():
            report.append(f"| {r['k']} | {r['flat_recall']*100:.2f}% | {r['hnsw_recall']*100:.2f}% | {r['flat_avg_latency_ms']} | {r['hnsw_avg_latency_ms']} | {r['speedup']}x |\n")
        report.append("\n")

    # Mixed results
    mixed_path = os.path.join(BENCH_DIR, "results_mixed.json")
    if os.path.exists(mixed_path):
        with open(mixed_path) as f:
            mixed_data = json.load(f)
        report.append("## 4. 混合读写测试\n\n")
        report.append("| 测试场景 | 总操作数 | 读QPS | 写QPS | 读p95(ms) | 写p95(ms) |\n")
        report.append("|----------|----------|-------|-------|-----------|-----------|\n")
        for r in mixed_data:
            ex = r.get('extra', {})
            report.append(f"| {r['name']} | {r['total_requests']} | {ex.get('read_qps',0)} | {ex.get('write_qps',0)} | {ex.get('read_latency_p95_ms',0)} | {ex.get('write_latency_p95_ms',0)} |\n")
        report.append("\n")

    report.append("## 可视化图表\n\n")
    for chart in ["chart_write.html", "chart_search.html", "chart_recall.html", "chart_mixed.html"]:
        p = os.path.join(BENCH_DIR, chart)
        if os.path.exists(p):
            report.append(f"- [{chart}](./bench/{chart})\n")

    report_path = os.path.join(os.path.dirname(BENCH_DIR), "BENCHMARK_REPORT.md")
    with open(report_path, "w") as f:
        f.writelines(report)
    print(f"\n  Report saved to {report_path}")
    return report_path


async def main():
    print("="*70)
    print("  vectorDB 完整性能压测套件")
    print("="*70)

    # Check service is up
    sys.path.insert(0, BENCH_DIR)
    from common import wait_for_service
    ok = await wait_for_service()
    if not ok:
        print("ERROR: vectorDB service is not running on port 7781!")
        print("Start it with: cd /data/workspace/vectorDB/build && export VECTORDB_CODE_BASE=/data/workspace/vectorDB && ./bin/vdb_server")
        sys.exit(1)
    print("  Service is up.\n")

    # Run benchmarks
    scripts = ["bench_write.py", "bench_search.py", "bench_mixed.py", "bench_recall.py"]
    for script in scripts:
        rc = run_script(script)
        if rc != 0:
            print(f"  WARNING: {script} exited with code {rc}")

    # Generate charts
    print(f"\n{'#'*70}")
    print("# Generating visualization charts")
    print(f"{'#'*70}")
    run_script("plot_results.py")

    # Generate report
    print(f"\n{'#'*70}")
    print("# Generating consolidated report")
    print(f"{'#'*70}")
    report_path = generate_report()

    print(f"\n{'='*70}")
    print("  ALL BENCHMARKS COMPLETE!")
    print(f"{'='*70}")
    print(f"  Report: {report_path}")
    print(f"  Charts: {BENCH_DIR}/chart_*.html")
    print(f"  Raw data: {BENCH_DIR}/results_*.json")


if __name__ == "__main__":
    asyncio.run(main())
