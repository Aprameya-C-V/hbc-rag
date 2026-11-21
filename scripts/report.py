"""Build the comparison report from persisted run results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hbc_harness.eval import mcnemar, paired_bootstrap
from hbc_harness.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "reports"


def read_run(path: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((path / "summary.json").read_text())
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines() if line.strip()]
    return summary, rows


def by_dataset(summaries: list[tuple[Path, dict, list[dict]]], dataset: str) -> dict[str, tuple[dict, dict[str, dict]]]:
    out = {}
    for path, summary, rows in summaries:
        if summary["dataset"] != dataset:
            continue
        out[summary["policy"]] = (summary, {r["qid"]: r for r in rows})
    return out


def paired(policies: dict[str, tuple[dict, dict[str, dict]]]) -> dict | None:
    if "fine" not in policies or "coarse" not in policies:
        return None
    fine_rows = policies["fine"][1]
    coarse_rows = policies["coarse"][1]
    common = sorted(set(fine_rows) & set(coarse_rows))
    if not common:
        return None
    f1_fine = [float(fine_rows[q]["f1"]) for q in common]
    f1_coarse = [float(coarse_rows[q]["f1"]) for q in common]
    em_fine = [bool(fine_rows[q]["em"]) for q in common]
    em_coarse = [bool(coarse_rows[q]["em"]) for q in common]
    return {
        "n": len(common),
        "f1_delta": paired_bootstrap(f1_coarse, f1_fine),
        "em_mcnemar": mcnemar(em_coarse, em_fine),
        "cost_delta": paired_bootstrap(
            [float(coarse_rows[q]["cost_usd"]) for q in common],
            [float(fine_rows[q]["cost_usd"]) for q in common],
        ),
    }


def plot(dataset: str, policies: dict[str, tuple[dict, dict[str, dict]]]) -> list[str]:
    figures = []
    OUT.mkdir(exist_ok=True)
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)

    metrics = [("em", "Exact match"), ("f1", "Token F1"), ("support_recall", "Support recall"), ("hops_mean", "Mean hops")]
    names = [p for p in ("fine", "coarse") if p in policies]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (key, label) in zip(axes, metrics):
        values = [policies[p][0][key] for p in names]
        ax.bar(names, values, color=["#8b8f96", "#c9f27c"][: len(names)])
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, max(1.0, max(values) * 1.25))
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / f"{dataset}-quality.png", dpi=160)
    plt.close(fig)
    figures.append(f"figures/{dataset}-quality.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    latency = [policies[p][0] for p in names]
    axes[0].bar(names, [s["latency_p50_ms"] / 1000 for s in latency], color="#8b8f96")
    axes[0].set_title("p50 latency (s)", fontsize=10)
    axes[1].bar(names, [s["latency_p95_ms"] / 1000 for s in latency], color="#8b8f96")
    axes[1].set_title("p95 latency (s)", fontsize=10)
    axes[2].bar(names, [s["cost_usd"] for s in latency], color="#c9f27c")
    axes[2].set_title("Total cost (USD)", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / f"{dataset}-serving.png", dpi=160)
    plt.close(fig)
    figures.append(f"figures/{dataset}-serving.png")
    return figures


def table(dataset: str, policies: dict[str, tuple[dict, dict[str, dict]]]) -> str:
    cols = ["policy", "n", "em", "f1", "support_recall", "hops_mean", "cache_hit_rate", "cost_usd", "latency_p50_ms", "latency_p95_ms"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name in ("fine", "coarse"):
        if name not in policies:
            continue
        s = policies[name][0]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(s["questions"]),
                    f"{s['em']:.3f}",
                    f"{s['f1']:.3f}",
                    f"{s['support_recall']:.3f}",
                    f"{s['hops_mean']:.2f}",
                    f"{s['cache_hit_rate'] * 100:.1f}%",
                    f"${s['cost_usd']:.5f}",
                    f"{s['latency_p50_ms'] / 1000:.2f}s",
                    f"{s['latency_p95_ms'] / 1000:.2f}s",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def load_section() -> tuple[str, str, list[str]]:
    files = sorted(RUNS.glob("load-*.jsonl"))
    if not files:
        return "", "", []
    rows_out = []
    for path in files:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        latencies = [r["latency_ms"] for r in rows if r["ok"]]
        stats = summarize(latencies, slo_ms=8000)
        label = path.stem.replace("load-", "").replace("qps", " qps")
        rows_out.append((label, stats, len(rows)))

    md = ["| offered load | requests | p50 | p95 | goodput at 8s |\n|---|---|---|---|---|"]
    for label, stats, total in rows_out:
        md.append(
            f"| {label} | {total} | {stats.get('p50_ms', 0)/1000:.2f}s | "
            f"{stats.get('p95_ms', 0)/1000:.2f}s | {stats.get('goodput', 0)*100:.1f}% |"
        )

    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    labels = [r[0] for r in rows_out]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].bar(labels, [r[1].get("p50_ms", 0) / 1000 for r in rows_out], color="#8b8f96")
    axes[0].set_title("p50 latency (s)", fontsize=10)
    axes[1].bar(labels, [r[1].get("p95_ms", 0) / 1000 for r in rows_out], color="#8b8f96")
    axes[1].set_title("p95 latency (s)", fontsize=10)
    axes[2].bar(labels, [r[1].get("goodput", 0) * 100 for r in rows_out], color="#c9f27c")
    axes[2].set_title("goodput at 8s SLO (%)", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "load-curve.png", dpi=160)
    plt.close(fig)
    return "\n".join(md), "\n".join(md), ["figures/load-curve.png"]


def main() -> None:
    summaries = []
    for path in sorted(RUNS.glob("*/summary.json")):
        summary, rows = read_run(path.parent)
        summaries.append((path.parent, summary, rows))

    datasets = sorted({s["dataset"] for _, s, _ in summaries})
    blocks = []
    for dataset in datasets:
        policies = by_dataset(summaries, dataset)
        stats = paired(policies)
        figures = plot(dataset, policies)
        delta = ""
        if stats:
            d = stats["f1_delta"]
            m = stats["em_mcnemar"]
            c = stats["cost_delta"]
            delta = (
                f"Paired over {stats['n']} questions: coarse minus fine F1 "
                f"{d['mean_diff']:+.3f} (95% CI {d['ci_low']:+.3f} to {d['ci_high']:+.3f}); "
                f"McNemar EM chi2 {m['statistic']:.2f}, p {m['p_value']:.3f}; "
                f"cost delta ${c['mean_diff']:+.6f} per question."
            )
        blocks.append(
            f"## {dataset}\n\n{table(dataset, policies)}\n\n{delta}\n\n"
            + "\n\n".join(f"![{dataset}]({f})" for f in figures)
            + "\n"
        )

    load_md, _, load_figs = load_section()
    load_html = ""
    if load_md:
        load_html = "<h2>Offered load</h2>" + load_md.replace("|", "").replace("\n", "<br>") if False else (
            "<h2>Offered load</h2><p class=\"note\">Open-loop Poisson arrivals against the HTTP API. Latency measured from arrival, not send. Goodput is the share of requests inside an 8s SLO.</p>"
            + "\n".join(f'<img src="{f}" />' for f in load_figs)
            + "<pre>" + load_md + "</pre>"
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>HBC-RAG results</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;max-width:980px;margin:48px auto;padding:0 20px;background:#0b0c0e;color:#e8e6e1}}
h1,h2{{letter-spacing:-0.02em}} table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}
th,td{{border-bottom:1px solid #23262b;padding:8px 10px;text-align:left}} th{{color:#8b8f96;font-weight:500}}
code{{background:#121417;padding:2px 5px;border-radius:4px}} img{{max-width:100%;border:1px solid #23262b;border-radius:10px;margin:10px 0}}
.note{{color:#8b8f96;font-size:13px}}
</style></head><body>
<h1>HBC-RAG results</h1>
<p class="note">Hop-bounded agentic RAG, DeepSeek V4.1 Flash via OpenCode Go. Deterministic scoring: normalized exact match, token F1, supporting-paragraph recall. No LLM judge in the headline.</p>
{''.join(blocks)}
{load_html}
<h2>Threats to validity</h2>
<ul class="note">
<li>One seed per dataset. Differences below 2 points are noise until replicated.</li>
<li>MuSiQue is a 50-question holdout, not tuned on.</li>
<li>p95 includes gateway 429 retries, so it measures the served path, not model latency alone.</li>
<li>Cache hit rate is affected by gateway routing, which varies run to run.</li>
</ul>
</body></html>
"""
    OUT.mkdir(exist_ok=True)
    (OUT / "report.html").write_text(html)
    (OUT / "index.html").write_text(html)
    (OUT / "report.md").write_text("\n".join(blocks) + "\n\n## offered load\n\n" + load_md + "\n")
    print(f"wrote {OUT / 'report.html'}")
    print(f"wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
