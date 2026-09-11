"""Build comparable dataset-centered English plots/tables from verified evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"arxivqa": "ArxivQA", "docvqa": "DocVQA", "infovqa": "InfoVQA",
         "tabfquad": "TabFQuAD", "tatdqa": "TAT-DQA", "shiftproject": "Shift"}
TARGETS = {"vidore-v0.2": "ColQwen2.5 v0.2", "metric-ai-3b": "Metric-AI 3B",
           "tsystems-3b": "T-Systems 3B", "colnomic-3b": "ColNomic 3B"}
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#8B949E"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    for sub in ("figures", "tables", "data"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    used = {}
    def read(rel, lines=False):
        p = args.evidence / rel
        used[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        return [json.loads(s) for s in p.read_text().splitlines()] if lines else json.loads(p.read_text())
    def dump(name, value):
        (out / "data" / name).write_text(json.dumps(value, indent=2) + "\n")
    def csv_out(name, rows):
        with (out / "data" / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    def table(name, headers, rows, align=None):
        align = align or "l" + "r" * (len(headers) - 1)
        s = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
        s += [" & ".join(str(v) for v in row) + r" \\" for row in rows]
        s += [r"\bottomrule", r"\end{tabular}"]
        (out / "tables" / name).write_text("\n".join(s) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight"})
    def save(fig, name):
        fig.savefig(out / "figures" / (name + ".pdf"))
        fig.savefig(out / "figures" / (name + ".png"), dpi=300)
        plt.close(fig)

    sources_path = Path(__file__).parents[1] / "independent-endpoint/sources.json"
    used["repo:experiments/independent-endpoint/sources.json"] = hashlib.sha256(sources_path.read_bytes()).hexdigest()
    dump("input-sources.json",json.loads(sources_path.read_text()))
    dump("official-release-contract.json",read("2026-08-22-colqwen2-release-transition-v1/cpu-result.json"))

    # Primary official release: derive every top-k metric from saved ordered ranks.
    base = "2026-09-09-colqwen2-pooled-upgrade-v1/raw-output/output/"
    raw = read(base + "result.json")
    rk = read(base + "rankings.json")
    mask = read("2026-09-07-pooled-gallery-upgrade-matrix-v1/raw-output/query_mask.json")
    assert raw["hardware"] == "NVIDIA A100-SXM4-80GB"
    assert raw["processor"]["max_pixels"] == 602112
    target = "colqwen2-v1.0"
    pos, keep = {c: 0 for c in mask}, []
    for i, c in enumerate(rk["collection"]):
        valid = c != "shiftproject"
        if c in mask:
            valid &= bool(mask[c]["holdout_valid"][pos[c]])
            pos[c] += 1
        if valid: keep.append(i)
    assert len(keep) == 1033
    metrics = []
    routes = ["target_full", "stale", "bf16", "int8_token", "pca512_int8", "pca256_int8", "pca128_int8"]
    for route in routes:
        for i in keep:
            ranking = rk["top20"][target][route][i]
            ref = rk["top20"][target]["target_full"][i]
            gold = rk["gold"][i]
            rank = ranking.index(gold) + 1 if gold in ranking else 21
            metrics.append({"query_index": i, "collection": rk["collection"][i], "gold": gold,
                "route": route, "ndcg5": 1 / np.log2(rank + 1) if rank <= 5 else 0.,
                "recall10": int(rank <= 10), "ta10": len(set(ranking[:10]) & set(ref[:10])) / 10,
                "ordered1": int(ranking[0] == ref[0]), "ordered10": int(ranking[:10] == ref[:10])})
    csv_out("ranking-per-query.csv", metrics)
    datasets = list(NAMES)[:5]
    aggregates = []
    for c in datasets + ["ALL"]:
        for route in routes:
            subset = [m for m in metrics if m["route"] == route and (c == "ALL" or m["collection"] == c)]
            aggregates.append({"dataset": c, "route": route, "queries": len(subset),
                "gold_page_clusters": len(set((m["collection"], m["gold"]) for m in subset)),
                **{k: float(np.mean([m[k] for m in subset])) for k in ("ndcg5", "recall10", "ta10", "ordered1", "ordered10")}})
    csv_out("ranking-summary.csv", aggregates)
    def agg(c, route):
        return next(a for a in aggregates if a["dataset"] == c and a["route"] == route)
    assert agg("ALL","bf16")["ordered10"] == agg("ALL","bf16")["ordered1"] == 1
    assert agg("ALL","bf16")["ndcg5"] == agg("ALL","target_full")["ndcg5"]
    table("ranking.tex", ["Dataset", "$|\\mathcal Q|$", "Target nDCG@5", "Stale nDCG@5", "Stale TA@10", "Replay TA@10"],
          [[NAMES.get(c, "All (micro)"), agg(c,"bf16")["queries"], f'{agg(c,"target_full")["ndcg5"]:.3f}',
            f'{agg(c,"stale")["ndcg5"]:.3f}', f'{agg(c,"stale")["ta10"]:.3f}', f'{agg(c,"bf16")["ta10"]:.3f}'] for c in datasets + ["ALL"]])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), sharey=True)
    for ax, key, xlabel in zip(axes, ("ndcg5", "ta10"), ("nDCG@5 (quality)", "TA@10 (target fidelity)")):
        for y, c in enumerate(datasets):
            a, b = agg(c,"stale")[key], agg(c,"bf16")[key]
            ax.plot([a,b], [y,y], color="#C5CDD5", lw=2, zorder=1)
            ax.scatter(a, y, color=ORANGE, marker="s", s=30, label="Stale source index" if y==0 else None)
            ax.scatter(b, y, color=BLUE, marker="o", s=32, label="Target reference / BF16 replay" if y==0 else None)
        ax.set_xlim(0,1.04); ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.15)
        ax.set_yticks(range(5), [NAMES[c] for c in datasets]); ax.invert_yaxis()
    # Shared y axis was inverted twice above; set explicit order once.
    axes[0].set_ylim(4.5,-.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(top=.82, bottom=.2, left=.14, right=.98, wspace=.22)
    save(fig,"01-quality-vs-fidelity")

    # Independent, within-page matched native raw / replay times, never old stage clocks.
    endpoint_rows, endpoint_summary = [], []
    inputs = read("independent-endpoint/inputs.json")
    env = read("independent-endpoint/manifest.json")
    proto = read("independent-endpoint/protocol.json")
    assert env["gpu"] == "NVIDIA A100-SXM4-80GB"
    assert proto["max_pixels"] == 12845056 and proto["image_batch_size"] == 1
    dump("endpoint-contract.json", {"protocol":proto, "environment":env["environment"],
         "gpu":env["gpu"], "cuda":env["cuda"], "shared_gpu":env["shared_gpu"],
         "cudnn_sdpa":env["cudnn_sdpa"], "allow_tf32":env["allow_tf32"]})
    dump("endpoint-page-selection.json",inputs["pages"])
    bits_path = Path(__file__).parents[1] / "independent-endpoint/bitwise-verification.json"
    used["repo:experiments/independent-endpoint/bitwise-verification.json"] = hashlib.sha256(bits_path.read_bytes()).hexdigest()
    for key, label in TARGETS.items():
        ps = read(f"independent-endpoint/{key}-pages.jsonl", True)
        result = read(f"independent-endpoint/{key}-result.json")
        ranks = read(f"independent-endpoint/{key}-rankings.json")
        assert len(ps) == 200 and len(ranks["raw_top10"]) == 1033
        assert all(p["image_sha256"] == q["image_sha256"] for p,q in zip(ps,inputs["pages"],strict=True))
        for i,p in enumerate(ps):
            endpoint_rows.append(dict(target=key,index=i,**p))
        endpoint_summary.append({"target": label, "pages":len(ps), "equal_pages":sum(p["tensor_equal"] for p in ps),
            "max_abs_error":max(p["max_abs_error"] for p in ps),
            "ordered10_equal":sum(a==b for a,b in zip(ranks["raw_top10"],ranks["replay_top10"],strict=True)),
            "queries":len(ranks["raw_top10"]), "raw_seconds":sum(p["raw_page_seconds"] for p in ps),
            "replay_seconds":sum(p["replay_with_read_seconds"] for p in ps)})
        assert endpoint_summary[-1]["equal_pages"] == result["tensor_equal_pages"]
        assert abs(endpoint_summary[-1]["raw_seconds"]-result["raw_page_seconds"]) < 1e-8
        assert abs(endpoint_summary[-1]["replay_seconds"]-result["replay_with_read_seconds"]) < 1e-8
    dump("independent-pages.json",endpoint_rows)
    csv_out("independent-summary.csv",endpoint_summary)
    table("independent-targets.tex",["Target (source: ColQwen2.5 v0.1)","Equal pages","Max error","Ordered top-10"],
        [[r["target"],f'{r["equal_pages"]}/{r["pages"]}',f'{r["max_abs_error"]:.0f}',f'{r["ordered10_equal"]}/{r["queries"]}'] for r in endpoint_summary])
    timing=[]
    for c in NAMES:
        ps=[p for p in endpoint_rows if p["target"]=="vidore-v0.2" and p["collection"]==c]
        r=np.array([p["raw_page_seconds"] for p in ps]); t=np.array([p["replay_with_read_seconds"] for p in ps])
        timing.append({"dataset":c,"pages":len(ps),"raw_mean":float(r.mean()),"raw_sd":float(r.std(ddof=1)),
            "replay_mean":float(t.mean()),"replay_sd":float(t.std(ddof=1)),"ratio_of_sums":float(t.sum()/r.sum())})
    csv_out("timing-by-dataset.csv",timing)
    table("timing.tex",["Dataset","Pages","Raw s/page","Replay s/page","Replay / raw"],
        [[NAMES[r["dataset"]],r["pages"],f'{r["raw_mean"]:.3f}',f'{r["replay_mean"]:.3f}',f'{100*r["ratio_of_sums"]:.1f}\\%'] for r in timing])
    fig,ax=plt.subplots(figsize=(7,3.3))
    y=np.arange(6); x=[r["raw_mean"] for r in timing]; z=[r["replay_mean"] for r in timing]
    ax.barh(y-.16,x,.29,color=GRAY,label="Full target encoding")
    ax.barh(y+.16,z,.29,color=BLUE,label="Replay incl. state read")
    for i,r in enumerate(timing):
        ax.text(max(x)*1.03,i,f'{100*r["ratio_of_sums"]:.1f}%',va="center",fontsize=9,color=BLUE)
    ax.text(max(x)*1.03,-.8,"Replay / raw",fontsize=8,ha="center")
    ax.set_yticks(y,[f'{NAMES[r["dataset"]]} (n={r["pages"]})' for r in timing]);ax.invert_yaxis()
    ax.set_xlim(0,max(x)*1.2);ax.set_xlabel("Mean document-side time (seconds / page)");ax.grid(axis="x",alpha=.15)
    ax.legend(loc="upper center",bbox_to_anchor=(.43,1.23),ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.2,right=.97,bottom=.17,top=.8)
    save(fig,"02-matched-encoding-time")

    # Mechanism: show the operation, not an abstract model/vendor label.
    core=read("core-mechanism/result.json"); core_rows=read("core-mechanism/pages.jsonl",True)
    assert core["all_checks_finite"] and core["all_native_called_vision"] and core["all_replay_skipped_vision"]
    specs=[("target_text_embedding","Target text embeddings change","Rebuild target text","Keep source text","factorized_replay","stale_fused_replay"),
           ("target_vision","Vision weights change","Raw fallback","Force old visual state","raw_fallback","factorized_replay"),
           ("target_merger","Merger weights change","Raw fallback","Force old visual state","raw_fallback","factorized_replay"),
           ("missing_positions","Position reconstruction ablation","Rebuild positions","Zero positions","factorized_replay","zero_position_replay")]
    mechanism=[]
    for case,desc,good,bad,g,b in specs:
        for route,label in ((g,good),(b,bad)):
            ps=[r["checks"][route] for r in core_rows if r["case"]==case]
            mechanism.append({"condition":desc,"execution":label,"equal_pages":sum(p["bit_equal"] for p in ps),
                "pages":len(ps),"min_max_error":min(p["max_abs_error"] for p in ps),"max_max_error":max(p["max_abs_error"] for p in ps)})
    csv_out("mechanism.csv",mechanism)
    table("mechanism.tex",["Condition","Execution","Exact pages"],
        [[r["condition"],r["execution"],f'{r["equal_pages"]}/{r["pages"]}'] for r in mechanism],"llr")
    fig,ax=plt.subplots(figsize=(7,3.15))
    for y,(case,desc,good,bad,g,b) in enumerate(specs):
        rows=[r for r in core_rows if r["case"]==case]
        for check,color,marker,label in [(g,BLUE,"o","Complete target execution"),(b,ORANGE,"x","Shortcut / ablation")]:
            errors=[r["checks"][check]["max_abs_error"] for r in rows]
            ax.scatter(errors,y+np.linspace(-.12,.12,len(errors)),color=color,marker=marker,s=25,label=label if y==0 else None)
        ax.text(.018,y,"8/8 exact",color=BLUE,va="center",fontsize=8)
    ax.set_yticks(range(4),["Target text embeddings change","Vision weights change","Merger weights change","Position reconstruction ablation"])
    ax.set_ylim(3.5,-.5);ax.set_xlim(-.025,.56);ax.set_xlabel("Maximum absolute representation error per page")
    ax.grid(axis="x",alpha=.15)
    ax.legend(loc="upper center",bbox_to_anchor=(.42,1.23),ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.36,right=.98,top=.79,bottom=.2);save(fig,"03-mechanism-conditions")

    # Processor boundary is a scoped validity result, never a saving percentage.
    processor=read("page-validity/cpu-v1/summary.json")
    fig,ax=plt.subplots(figsize=(6.7,2.8))
    for i,c in enumerate(NAMES):
        n=processor["by_collection"][c]["pages"]; k=processor["by_collection"][c]["equal"]
        ax.barh(i,k/n,color=BLUE,label="Identical processor outputs" if i==0 else None)
        ax.barh(i,1-k/n,left=k/n,color="#D9DFE5",label="Changed outputs" if i==0 else None)
        ax.text(1.02,i,f"{k}/{n}",va="center",fontsize=8)
    ax.set_yticks(range(6),list(NAMES.values()));ax.invert_yaxis();ax.set_xlim(0,1.12)
    ax.set_xticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"]);ax.set_xlabel("Fraction of pages (not time savings)")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.2),ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.16,right=.97,bottom=.19,top=.85);save(fig,"04-processor-boundary")

    # Approximation within the SAME ColQwen2 run only.
    labels={"bf16":"BF16", "int8_token":"INT8", "pca512_int8":"PCA-512", "pca256_int8":"PCA-256", "pca128_int8":"PCA-128"}
    compression=[]
    for r,label in labels.items():
        m=agg("ALL",r)
        compression.append({"codec":label,"MB_per_page":raw["payload_bytes_per_page"][r]/1e6,
             "ta10":m["ta10"],"ndcg5":m["ndcg5"],"delta_ndcg5":m["ndcg5"]-agg("ALL","target_full")["ndcg5"]})
    csv_out("compression.csv",compression)
    table("compression.tex",["State codec","MB/page","TA@10","nDCG@5","$\\Delta$nDCG@5"],
        [[r["codec"],f'{r["MB_per_page"]:.3f}',f'{r["ta10"]:.3f}',f'{r["ndcg5"]:.3f}',f'{r["delta_ndcg5"]:+.3f}'] for r in compression])
    fig,ax=plt.subplots(figsize=(6.5,2.8))
    for r in compression:
        exact=r["codec"]=="BF16"
        ax.scatter(r["MB_per_page"],r["ta10"],s=48,marker="s" if exact else "o",color=BLUE if exact else ORANGE)
        ax.annotate(r["codec"],(r["MB_per_page"],r["ta10"]),xytext=(0,-15 if exact else 8),textcoords="offset points",ha="center",fontsize=8)
    ax.set_xscale("log");ax.set_xlim(.065,3);ax.set_ylim(.35,1.06)
    ax.set_xlabel("Stored state payload (MB / page, log scale)");ax.set_ylabel("TA@10");ax.grid(alpha=.15)
    fig.subplots_adjust(left=.12,right=.98,bottom=.23,top=.93);save(fig,"05-compression-tradeoff")

    dump("manifest.json",{"source_sha256":used,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection":json.loads(Path(__file__).with_name("selection.json").read_text()),
        "versions":{"numpy":np.__version__,"matplotlib":matplotlib.__version__},
        "micro_metrics":{r:agg("ALL",r) for r in routes},
        "macro_ndcg5":{r:float(np.mean([agg(c,r)["ndcg5"] for c in datasets])) for r in routes}})
    print(json.dumps({"ranking_queries":len(keep),"independent":endpoint_summary,"timing":timing,"compression":compression},indent=2))


if __name__ == "__main__":
    main()
