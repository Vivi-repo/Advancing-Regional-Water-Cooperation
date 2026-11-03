"""Command line entry point.

  python -m wnego run --backend ollama --model qwen2.5:32b --rag chroma --events qosh_tepa_shock
  python -m wnego run --backend mock --scenario dry                 # offline plumbing test
  python -m wnego run --backend ollama --model qwen2.5:32b --hitl final   # human approves the final text
  python -m wnego search "obligation not to cause significant harm" --rag chroma
  python -m wnego index --docs ../data/raw --db ../chroma_db        # improved index (amu_darya_v2)
  python -m wnego experiment config/experiment.yaml
  python -m wnego benchmarks --scenario normal
"""
import argparse
import json

from .agents import Condition
from .benchmarks import nash_bargaining_solution, utilitarian_optimum
from .environment import Basin
from .experiment import ROOT, load_configs, run_experiment, run_one
from .protocol import save_transcript
from .retrieval import build_index, make_retriever


def _configs_with_rag(a):
    cfg = load_configs()
    if getattr(a, "rag", None):
        cfg["rag"]["backend"] = a.rag
    if getattr(a, "collection", None):
        cfg["rag"]["chroma"]["collection"] = a.collection
    return cfg


def main():
    ap = argparse.ArgumentParser(prog="wnego")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one negotiation and print the public record")
    r.add_argument("--backend", default="mock", choices=["mock", "ollama", "anthropic", "openai_compat"])
    r.add_argument("--model", default="")
    r.add_argument("--base-url", default=None, help="openai_compat server URL")
    r.add_argument("--host", default=None, help="Ollama host, e.g. http://localhost:11434")
    r.add_argument("--persona", default="rich")
    r.add_argument("--payoff", default="hidden", choices=["hidden", "own", "full"])
    r.add_argument("--grounding", default="rag", choices=["rag", "none"])
    r.add_argument("--legal", default="on", choices=["on", "off"])
    r.add_argument("--events", default="none")
    r.add_argument("--hitl", default="off", choices=["off", "final"])
    r.add_argument("--scenario", default="normal")
    r.add_argument("--rounds", type=int, default=6)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--rag", default=None, choices=["bm25", "chroma"], help="override config/rag.yaml backend")
    r.add_argument("--collection", default=None, help="Chroma collection (local_docs or amu_darya_v2)")
    r.add_argument("--out", default=str(ROOT / "runs" / "single_run.json"))

    s = sub.add_parser("search", help="test the retriever the agents will use")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--rag", default=None, choices=["bm25", "chroma"])
    s.add_argument("--collection", default=None)
    s.add_argument("--doc-type", default=None, help="filter, e.g. legal")

    ix = sub.add_parser("index", help="build the improved Chroma collection from a documents folder")
    ix.add_argument("--docs", required=True, help="the RAG app's documents folder (data/raw)")
    ix.add_argument("--db", required=True, help="the RAG app's chroma_db folder")
    ix.add_argument("--collection", default="amu_darya_v2")
    ix.add_argument("--embed-model", default="nomic-embed-text")
    ix.add_argument("--max-chars", type=int, default=1000)

    e = sub.add_parser("experiment", help="run a factorial experiment from a YAML config")
    e.add_argument("config")
    e.add_argument("--out", default=None)

    bm = sub.add_parser("benchmarks", help="print no deal, Protocol 566, Nash bargaining and utilitarian reference points")
    bm.add_argument("--scenario", default="normal")

    a = ap.parse_args()
    if a.cmd == "run":
        spec = {"name": a.model or a.backend, "type": a.backend, "model": a.model}
        if a.base_url:
            spec["base_url"] = a.base_url
        if a.host:
            spec["host"] = a.host
        cond = Condition(persona=a.persona, payoff=a.payoff, grounding=a.grounding, legal=a.legal, events=a.events,
                         hitl=a.hitl, scenario=a.scenario, max_rounds=a.rounds)
        res = run_one(spec, cond, a.seed, configs=_configs_with_rag(a))
        print("\n".join(res["public_record"]))
        print("\nAGREEMENT" if res["agreement"] else "\nNO DEAL", "after", res["rounds"], "rounds")
        print(json.dumps(res["metrics"], indent=1, default=float))
        save_transcript(res, a.out)
        print("transcript saved to", a.out)
    elif a.cmd == "search":
        cfg = _configs_with_rag(a)
        ret = make_retriever(cfg["rag"], ROOT)
        where = {"doc_type": a.doc_type} if a.doc_type else None
        for p in ret.search(a.query, a.k, where):
            print(f"\n[{p.id}] score {p.score}  ({p.source}, doc_type={p.meta.get('doc_type', 'n/a')})\n{p.text[:500]}")
    elif a.cmd == "index":
        from .retrieval import ollama_embedder
        stats = build_index(a.docs, a.db, a.collection, ollama_embedder(a.embed_model), a.max_chars)
        print(stats)
    elif a.cmd == "experiment":
        run_experiment(a.config, a.out)
    elif a.cmd == "benchmarks":
        b = Basin.load(ROOT / "config/basin.yaml", a.scenario)
        print("NO DEAL\n" + b.describe(b.no_deal_proposal()), "\n", b.no_deal_utilities())
        p = b.protocol_566_proposal()
        print("\nPROTOCOL 566 ANCHOR\n" + b.describe(p), "\n", b.utilities(p))
        for name, fn in [("NASH BARGAINING SOLUTION", nash_bargaining_solution), ("UTILITARIAN OPTIMUM", utilitarian_optimum)]:
            sol = fn(b)
            print(f"\n{name}\n" + b.describe(sol["proposal"]), "\n", sol["utilities"])


if __name__ == "__main__":
    main()
