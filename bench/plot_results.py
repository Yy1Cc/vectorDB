#!/usr/bin/env python3
"""Generate visualization charts from benchmark results."""
import json
import os
import sys

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(name):
    path = os.path.join(BENCH_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_write_results():
    data = load_json("results_write.json")
    if not data:
        print("  No write results found, skipping.")
        return

    names = [r['name'].split(' [')[0] for r in data]
    qps_vals = [r['qps'] for r in data]
    p50_vals = [r['latency_p50_ms'] for r in data]
    p99_vals = [r['latency_p99_ms'] for r in data]

    if HAS_PLOTLY:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Throughput (QPS)", "Latency (ms)"))
        fig.add_trace(go.Bar(x=names, y=qps_vals, name="QPS", marker_color='steelblue'), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=p50_vals, name="p50", marker_color='orange'), row=1, col=2)
        fig.add_trace(go.Bar(x=names, y=p99_vals, name="p99", marker_color='red'), row=1, col=2)
        fig.update_layout(title="Write Throughput Benchmarks", barmode='group', height=500)
        fig.update_xaxes(tickangle=45)
        out = os.path.join(BENCH_DIR, "chart_write.html")
        fig.write_html(out)
        print(f"  Write chart saved to {out}")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.bar(range(len(names)), qps_vals, color='steelblue')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax1.set_title("Throughput (QPS)")
        ax2.bar(range(len(names)), p50_vals, color='orange', label='p50')
        ax2.bar(range(len(names)), p99_vals, color='red', alpha=0.7, label='p99')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax2.set_title("Latency (ms)")
        ax2.legend()
        plt.tight_layout()
        out = os.path.join(BENCH_DIR, "chart_write.png")
        plt.savefig(out, dpi=150)
        print(f"  Write chart saved to {out}")


def plot_search_results():
    data = load_json("results_search.json")
    if not data:
        print("  No search results found, skipping.")
        return

    names = [r['name'] for r in data]
    qps_vals = [r['qps'] for r in data]
    p50_vals = [r['latency_p50_ms'] for r in data]
    p99_vals = [r['latency_p99_ms'] for r in data]

    if HAS_PLOTLY:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Search QPS", "Search Latency (ms)"))
        fig.add_trace(go.Bar(x=names, y=qps_vals, name="QPS", marker_color='green'), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=p50_vals, name="p50", marker_color='orange'), row=1, col=2)
        fig.add_trace(go.Bar(x=names, y=p99_vals, name="p99", marker_color='red'), row=1, col=2)
        fig.update_layout(title="Search Performance Benchmarks", barmode='group', height=500)
        fig.update_xaxes(tickangle=45)
        out = os.path.join(BENCH_DIR, "chart_search.html")
        fig.write_html(out)
        print(f"  Search chart saved to {out}")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.bar(range(len(names)), qps_vals, color='green')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax1.set_title("Search QPS")
        ax2.bar(range(len(names)), p50_vals, color='orange', label='p50')
        ax2.bar(range(len(names)), p99_vals, color='red', alpha=0.7, label='p99')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax2.set_title("Search Latency (ms)")
        ax2.legend()
        plt.tight_layout()
        out = os.path.join(BENCH_DIR, "chart_search.png")
        plt.savefig(out, dpi=150)
        print(f"  Search chart saved to {out}")


def plot_recall_results():
    data = load_json("results_recall.json")
    if not data:
        print("  No recall results found, skipping.")
        return

    k_vals = [int(k.replace('k=', '')) for k in data.keys()]
    flat_recalls = [data[f'k={k}']['flat_recall'] * 100 for k in k_vals]
    hnsw_recalls = [data[f'k={k}']['hnsw_recall'] * 100 for k in k_vals]
    flat_lats = [data[f'k={k}']['flat_avg_latency_ms'] for k in k_vals]
    hnsw_lats = [data[f'k={k}']['hnsw_avg_latency_ms'] for k in k_vals]

    if HAS_PLOTLY:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Recall Rate (%)", "Avg Latency (ms)"))
        fig.add_trace(go.Bar(x=[str(k) for k in k_vals], y=flat_recalls, name="FLAT", marker_color='steelblue'), row=1, col=1)
        fig.add_trace(go.Bar(x=[str(k) for k in k_vals], y=hnsw_recalls, name="HNSW", marker_color='orange'), row=1, col=1)
        fig.add_trace(go.Bar(x=[str(k) for k in k_vals], y=flat_lats, name="FLAT", marker_color='steelblue', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=[str(k) for k in k_vals], y=hnsw_lats, name="HNSW", marker_color='orange', showlegend=False), row=1, col=2)
        fig.update_layout(title="HNSW vs FLAT: Recall Rate & Latency", barmode='group', height=500)
        out = os.path.join(BENCH_DIR, "chart_recall.html")
        fig.write_html(out)
        print(f"  Recall chart saved to {out}")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x = range(len(k_vals))
        ax1.bar([i - 0.2 for i in x], flat_recalls, 0.4, label='FLAT', color='steelblue')
        ax1.bar([i + 0.2 for i in x], hnsw_recalls, 0.4, label='HNSW', color='orange')
        ax1.set_xticks(list(x))
        ax1.set_xticklabels([str(k) for k in k_vals])
        ax1.set_xlabel("k")
        ax1.set_ylabel("Recall (%)")
        ax1.set_title("Recall Rate")
        ax1.legend()
        ax2.bar([i - 0.2 for i in x], flat_lats, 0.4, label='FLAT', color='steelblue')
        ax2.bar([i + 0.2 for i in x], hnsw_lats, 0.4, label='HNSW', color='orange')
        ax2.set_xticks(list(x))
        ax2.set_xticklabels([str(k) for k in k_vals])
        ax2.set_xlabel("k")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Avg Latency")
        ax2.legend()
        plt.tight_layout()
        out = os.path.join(BENCH_DIR, "chart_recall.png")
        plt.savefig(out, dpi=150)
        print(f"  Recall chart saved to {out}")


def plot_mixed_results():
    data = load_json("results_mixed.json")
    if not data:
        print("  No mixed results found, skipping.")
        return

    names = [r['name'] for r in data]
    read_qps = [r['extra'].get('read_qps', 0) for r in data]
    write_qps = [r['extra'].get('write_qps', 0) for r in data]
    read_p95 = [r['extra'].get('read_latency_p95_ms', 0) for r in data]
    write_p95 = [r['extra'].get('write_latency_p95_ms', 0) for r in data]

    if HAS_PLOTLY:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("QPS (Read vs Write)", "p95 Latency (ms)"))
        fig.add_trace(go.Bar(x=names, y=read_qps, name="Read QPS", marker_color='green'), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=write_qps, name="Write QPS", marker_color='steelblue'), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=read_p95, name="Read p95", marker_color='green', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=names, y=write_p95, name="Write p95", marker_color='steelblue', showlegend=False), row=1, col=2)
        fig.update_layout(title="Mixed Read/Write Benchmarks", barmode='group', height=500)
        fig.update_xaxes(tickangle=45)
        out = os.path.join(BENCH_DIR, "chart_mixed.html")
        fig.write_html(out)
        print(f"  Mixed chart saved to {out}")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        x = range(len(names))
        ax1.bar([i - 0.2 for i in x], read_qps, 0.4, label='Read', color='green')
        ax1.bar([i + 0.2 for i in x], write_qps, 0.4, label='Write', color='steelblue')
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax1.set_title("QPS")
        ax1.legend()
        ax2.bar([i - 0.2 for i in x], read_p95, 0.4, label='Read p95', color='green')
        ax2.bar([i + 0.2 for i in x], write_p95, 0.4, label='Write p95', color='steelblue')
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax2.set_title("p95 Latency (ms)")
        ax2.legend()
        plt.tight_layout()
        out = os.path.join(BENCH_DIR, "chart_mixed.png")
        plt.savefig(out, dpi=150)
        print(f"  Mixed chart saved to {out}")


def main():
    print("\n" + "="*70)
    print("  GENERATING VISUALIZATION CHARTS")
    print("="*70)
    lib = "plotly" if HAS_PLOTLY else "matplotlib"
    print(f"  Using: {lib}")
    plot_write_results()
    plot_search_results()
    plot_recall_results()
    plot_mixed_results()
    print("\n  All charts generated.")


if __name__ == "__main__":
    main()
