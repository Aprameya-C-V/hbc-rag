"""Generate the HBC-RAG report, figures and thumbnail."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hbc_harness.eval import mcnemar, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "reports"
DATA = OUT / "data"
FIGS = OUT / "figures"

PAPER = "#f3f1eb"
INK = "#14140f"
MUTED = "#6b675c"
RULE = "#d8d3c6"
ACCENT = "#1e4b3d"
FILL = "#c9c3b4"

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": RULE,
        "font.family": "Helvetica Neue",
        "font.size": 9,
    }
)


def read_summaries():
    groups = defaultdict(dict)
    for path in sorted(RUNS.glob("*/summary.json")):
        s = json.loads(path.read_text())
        rows = [json.loads(line) for line in (path.parent / "results.jsonl").read_text().splitlines() if line.strip()]
        by_qid = {r["qid"]: r for r in rows}
        groups[(s["dataset"], s.get("model", "unknown"))][s["policy"]] = (s, by_qid)
    return groups


def paired_block(policies):
    if "fine" not in policies or "coarse" not in policies:
        return ""
    fine, coarse = policies["fine"][1], policies["coarse"][1]
    common = sorted(set(fine) & set(coarse))
    if not common:
        return ""
    f1 = paired_bootstrap([float(coarse[q]["f1"]) for q in common], [float(fine[q]["f1"]) for q in common])
    em = mcnemar([bool(coarse[q]["em"]) for q in common], [bool(fine[q]["em"]) for q in common])
    cost_c = sum(float(coarse[q]["cost_usd"]) for q in common) / max(1, sum(bool(coarse[q]["em"]) for q in common))
    cost_f = sum(float(fine[q]["cost_usd"]) for q in common) / max(1, sum(bool(fine[q]["em"]) for q in common))
    return (
        f"Paired over {len(common)} questions: coarse minus fine F1 <b>{f1['mean_diff']:+.3f}</b> "
        f"(95% CI {f1['ci_low']:+.3f} to {f1['ci_high']:+.3f}). McNemar on exact match "
        f"chi2 {em['statistic']:.2f}, p {em['p_value']:.3f}. Cost per correct answer "
        f"${cost_f:.6f} fine, ${cost_c:.6f} coarse."
    )


def table(policies):
    head = ["policy", "n", "EM", "F1", "support recall", "hops", "p50", "p95", "cost"]
    rows = []
    for name in ("fine", "coarse"):
        if name not in policies:
            continue
        s = policies[name][0]
        rows.append(
            [name, str(s["questions"]), f"{s['em']:.3f}", f"{s['f1']:.3f}",
             f"{s['support_recall']:.3f}", f"{s['hops_mean']:.2f}",
             f"{s['latency_p50_ms']/1000:.2f}s", f"{s['latency_p95_ms']/1000:.2f}s",
             f"${s['cost_usd']:.5f}"]
        )
    html = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
    for row in rows:
        html += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return html + "</tbody></table>"


def figures(tag, policies):
    FIGS.mkdir(parents=True, exist_ok=True)
    names = [p for p in ("fine", "coarse") if p in policies]
    colors = [FILL, ACCENT][: len(names)]

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, (key, label) in zip(axes, [("em", "Exact match"), ("f1", "Token F1"), ("support_recall", "Support recall"), ("hops_mean", "Mean hops")]):
        vals = [policies[p][0][key] for p in names]
        ax.bar(names, vals, color=colors, width=0.55)
        ax.set_title(label, fontsize=9, color=MUTED)
        ax.set_ylim(0, max(1.0, max(vals) * 1.25))
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    q = FIGS / f"{tag}-quality.png"
    fig.savefig(q, dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    axes[0].bar(names, [policies[p][0]["latency_p50_ms"] / 1000 for p in names], color=colors, width=0.55)
    axes[0].set_title("p50 latency (s)", fontsize=9, color=MUTED)
    axes[1].bar(names, [policies[p][0]["latency_p95_ms"] / 1000 for p in names], color=colors, width=0.55)
    axes[1].set_title("p95 latency (s)", fontsize=9, color=MUTED)
    axes[2].bar(names, [policies[p][0]["cost_usd"] for p in names], color=colors, width=0.55)
    axes[2].set_title("Total cost (USD)", fontsize=9, color=MUTED)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    s = FIGS / f"{tag}-serving.png"
    fig.savefig(s, dpi=170)
    plt.close(fig)
    return q.name, s.name


def cache_sections():
    gem = DATA / "gemini_threshold.json"
    probe = DATA / "cache_probe.json"
    if not gem.exists() or not probe.exists():
        return ""
    g = json.loads(gem.read_text())
    p = json.loads(probe.read_text())

    grows = "<table><thead><tr><th>new tokens in hop</th><th>prompt tokens</th><th>cached tokens</th><th>cache share</th></tr></thead><tbody>"
    for r in g["rows"]:
        grows += (f"<tr><td>{r['delta_tokens_requested']}</td><td>{r['prompt_tokens']}</td>"
                  f"<td>{r['cached_tokens']}</td><td>{r['cache_rate']*100:.1f}%</td></tr>")
    grows += "</tbody></table>"

    hits = sum(1 for r in p["rows"] if r["cached_tokens"] > 0)
    lo = min(r["allowance_usd"] or 0 for r in p["rows"])
    hi = max(r["allowance_usd"] or 0 for r in p["rows"])
    ratio = hi / lo if lo else 0
    prow = "<table><thead><tr><th>attempt</th><th>prompt</th><th>cached</th><th>allowance USD</th></tr></thead><tbody>"
    for r in p["rows"]:
        prow += (f"<tr><td>{r['attempt']+1}</td><td>{r['prompt_tokens']}</td><td>{r['cached_tokens']}</td>"
                 f"<td>{r['allowance_usd']:.8f}</td></tr>")
    prow += "</tbody></table>"

    return f"""
<h2>Novelty: what the cache actually does</h2>
<p class="note">Two measurements that the hop-granularity table alone cannot explain.</p>

<h3>Gemini Interactions, one stateful chain per trial</h3>
<p>For a fixed prefix of roughly four thousand tokens, the cached span stays flat near 4,070 tokens across every hop size tested. The cache share therefore falls as the hop grows, because new tokens are always billed fresh. No 2,050-token incremental threshold was observed on Gemini 3.5 Flash Lite; Gemini 4 Hydrogen returned 503 and timed out during the test window.</p>
{grows}

<h3>OpenCode Go, DeepSeek, identical 19k-token prefix</h3>
<p>{hits} of {len(p['rows'])} calls hit the prefix cache. Billed allowance ranged from ${lo:.8f} to ${hi:.8f} for identical input, a {ratio:.0f}x spread caused by gateway routing rather than by the prompt.</p>
{prow}
"""


def thumbnail():
    fig, ax = plt.subplots(figsize=(12, 6.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.3)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 12, 6.3, color=PAPER, zorder=0))
    ax.add_patch(plt.Rectangle((0, 6.1), 12, 0.2, color=ACCENT, zorder=1))
    ax.text(0.7, 4.9, "HBC-RAG", fontsize=34, color=INK, weight="bold")
    ax.text(0.7, 4.35, "Hop-bounded agentic RAG with measured cache behaviour", fontsize=13, color=MUTED)
    stats = [("+0.119", "F1, coarse vs fine"), ("0.917", "support recall"), ("$0.00022", "cost per correct answer")]
    for i, (value, label) in enumerate(stats):
        x = 0.7 + i * 3.8
        ax.text(x, 2.5, value, fontsize=24, color=ACCENT, weight="bold")
        ax.text(x, 2.1, label, fontsize=10, color=MUTED)
    ax.text(0.7, 0.8, "DeepSeek V4.1 Flash · Neon Postgres + pgvector · 5,155 paragraphs · 300 questions", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(ROOT / "thumbnail.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    groups = read_summaries()
    order = [("hotpotqa", "deepseek-v4.1-flash"), ("musique", "deepseek-v4.1-flash"), ("hotpotqa", "mimo-v2.6-flash")]
    blocks = []
    for dataset, model in order:
        if (dataset, model) not in groups:
            continue
        policies = groups[(dataset, model)]
        tag = f"{dataset}-{model.split('-')[0]}"
        q, s = figures(tag, policies)
        title = f"{dataset}, {model}"
        blocks.append(f"""
<h2>{title}</h2>
{table(policies)}
<p class="note">{paired_block(policies)}</p>
<figure><img src="figures/{q}" alt="quality"><figcaption>Answer quality and evidence coverage</figcaption></figure>
<figure><img src="figures/{s}" alt="serving"><figcaption>Latency and cost</figcaption></figure>
""")

    cache = cache_sections()

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HBC-RAG results</title>
<style>
:root{{--paper:{PAPER};--ink:{INK};--muted:{MUTED};--rule:{RULE};--accent:{ACCENT}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue","Segoe UI",system-ui,sans-serif;font-size:16px;line-height:1.65;-webkit-font-smoothing:antialiased}}
.bar{{height:6px;background:var(--accent)}}
main{{max-width:980px;margin:0 auto;padding:56px 24px 96px}}
.eyebrow{{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:0 0 12px}}
h1{{font-size:44px;letter-spacing:-.03em;margin:0 0 10px;font-weight:600}}
h2{{font-size:22px;letter-spacing:-.01em;margin:52px 0 12px;font-weight:600}}
h3{{font-size:15px;margin:28px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:500}}
p{{margin:10px 0}}
.lede{{font-size:18px;color:var(--muted);max-width:70ch}}
table{{width:100%;border-collapse:collapse;margin:14px 0;font-size:14px}}
th{{text-align:left;font-weight:500;color:var(--muted);border-bottom:1px solid var(--rule);padding:8px 10px;font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase}}
td{{border-bottom:1px solid var(--rule);padding:9px 10px;font-variant-numeric:tabular-nums}}
.note{{color:var(--muted);font-size:14px}}
figure{{margin:18px 0 26px}}
img{{max-width:100%;border:1px solid var(--rule);border-radius:6px;display:block}}
figcaption{{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--muted);margin-top:6px;letter-spacing:.06em;text-transform:uppercase}}
ul{{margin:10px 0 10px 18px;padding:0}}
li{{margin:6px 0}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:13px;background:#e9e4d8;padding:1px 5px;border-radius:3px}}
.rule{{height:1px;background:var(--rule);margin:40px 0}}
</style></head><body>
<div class="bar"></div>
<main>
<p class="eyebrow">Study report · November 2026</p>
<h1>HBC-RAG</h1>
<p class="lede">Does bounding retrieval into fewer, larger hops improve agentic RAG, and what does the model cache actually contribute? Measured on DeepSeek V4.1 Flash and MiMo V2.6 Flash, scored deterministically, paired by question.</p>

<h2>Method</h2>
<ul>
<li>Corpus: 5,155 paragraphs from HotpotQA and MuSiQue, paragraph-level retrieval, local bge-small embeddings, pgvector HNSW plus Postgres full-text, fused with reciprocal rank fusion.</li>
<li>Policies: <code>fine</code> retrieves one chunk per hop with a four-hop cap; <code>coarse</code> retrieves four per hop with a two-hop cap. Same prompt, model, temperature 0 and stopping rule.</li>
<li>Scoring: normalized exact match, token F1, supporting-paragraph recall, hop count, billed cost, latency. No LLM judge in the headline.</li>
<li>Statistics: paired bootstrap over 10,000 resamples, McNemar on paired exact match.</li>
</ul>
{''.join(blocks)}
{cache}
<h2>What is not claimed</h2>
<p>Concurrency and scaling. Repeat load runs at one request per second produced goodput of 100 percent and 36 percent on identical settings, so the variation between runs is larger than the effect. The open-loop harness is in the repository, but no scaling conclusion is drawn from it.</p>
<div class="rule"></div>
<h2>Threats to validity</h2>
<ul class="note">
<li>One seed per dataset and model. Deltas below two points are noise until replicated.</li>
<li>MuSiQue uses 50 questions; its intervals are wide.</li>
<li>MiMo replication uses 100 questions, not the full 150.</li>
<li>p95 includes gateway 429 retries, so it measures the served path, not model latency alone.</li>
<li>Gemini 4 Hydrogen was unavailable during the cache threshold test, so that experiment ran on 3.5 Flash Lite.</li>
</ul>

<h2>Reproduce</h2>
<pre class="note">uv run python scripts/build_corpus.py --hotpot 300 --musique 150
uv run python scripts/run_eval.py --dataset hotpotqa --limit 150 --policies fine,coarse --concurrency 4
uv run python scripts/run_eval.py --dataset musique --limit 50 --policies fine,coarse --concurrency 4
uv run python scripts/run_eval.py --dataset hotpotqa --limit 100 --policies fine,coarse --concurrency 4 --model mimo-v2.6-flash --provider go
uv run python scripts/cache_probe.py
uv run python scripts/gemini_cache_threshold.py
uv run python scripts/report.py</pre>
</main></body></html>
"""
    (OUT / "report.html").write_text(html)
    (OUT / "index.html").write_text(html)
    thumbnail()
    print("wrote report.html, index.html, thumbnail.png")


if __name__ == "__main__":
    main()
