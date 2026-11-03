"""Policy-brief evaluation: does connecting verified numbers + per-stakeholder retrieval help?

Conditions per target (BCM/yr):
  full      verified calculations (calculator.py) + per-stakeholder retrieval
  no_calc   per-stakeholder retrieval, model does the arithmetic itself from scaling.txt chunks
  baseline  no retrieval, no calculator (plain Llama 3.2)

Real run:   python evaluate_policy.py --docs ./knowledge_base
Dry run:    python evaluate_policy.py --docs ./knowledge_base --dry-run

Automatic columns: stakeholders_covered (0-5), calc_numbers_correct (0-1), numbers_not_in_support.
Raters then fill correct_r1/_r2, grounded_r1/_r2 (1 or 0), stakeholder_quality_r1/_r2 (1-5), error_type.
"""
import argparse
import csv
import statistics

import policy as pol
import rag_core as rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=rc.DOCS_FOLDER)
    ap.add_argument("--db", default="./eval_chroma_db")
    ap.add_argument("--targets", default="2,3,4,5,6,8,10", help="BCM/yr targets, comma separated (7 gives a usable paired comparison)")
    ap.add_argument("--conditions", default="full,no_calc,baseline")
    ap.add_argument("--k-each", type=int, default=2, help="chunks retrieved per stakeholder")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="policy_results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        rc.DRY_RUN = True
        print("*** DRY RUN: fake model outputs. Do NOT report these numbers. ***")
    embed_fn, generate_fn = rc.default_embed_fn(), rc.default_generate_fn()
    out_prefix = args.out + ("_DRYRUN" if args.dry_run else "")

    col, status, info = rc.build_index(args.docs, args.db, embed_fn=embed_fn, rebuild=args.dry_run)
    if col is None:
        raise SystemExit(f"No documents found in {args.docs}")
    print(f"Index ({status}): {info['documents']} documents, {info['chunks']} chunks")

    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    rows = []
    for t in targets:
        for cond in conditions:
            for rep in range(args.repeats):
                out = pol.policy_brief(t, col, condition=cond, k_each=args.k_each, embed_fn=embed_fn,
                                       generate_fn=generate_fn, temperature=args.temperature, seed=args.seed)
                support = out["calc_text"] + " " + " ".join(c["text"] for cs in out["chunks"].values() for c in cs)
                rows.append({
                    "target_bcm": t, "condition": cond, "repeat": rep + 1,
                    "stakeholders_covered": pol.stakeholders_covered(out["text"]),
                    "calc_numbers_correct": pol.calc_numbers_correct(out["text"], t),
                    "numbers_not_in_support": ("" if cond == "baseline" else len(pol.numbers_not_in_support(out["text"], support + f" {t:g}"))),
                    "seconds": out["seconds"],
                    "brief": out["text"].replace("\r", " ").strip(),
                    "verified_calculations": out["calc_text"].replace("\n", " | "),
                    "correct_r1": "", "correct_r2": "", "grounded_r1": "", "grounded_r2": "",
                    "stakeholder_quality_r1": "", "stakeholder_quality_r2": "", "error_type": "", "notes": "",
                })
        print(f"  done: target {t:g} BCM")

    with open(out_prefix + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== SUMMARY (automatic columns only) ===")
    print("condition | n | stakeholders_covered | calc_numbers_correct | numbers_not_in_support")
    for cond in conditions:
        sub = [r for r in rows if r["condition"] == cond]
        nis = [r["numbers_not_in_support"] for r in sub if r["numbers_not_in_support"] != ""]
        print(f"{cond} | {len(sub)} | {statistics.mean(r['stakeholders_covered'] for r in sub):.2f} | "
              f"{statistics.mean(r['calc_numbers_correct'] for r in sub):.2f} | "
              f"{(statistics.mean(nis) if nis else float('nan')):.2f}")
    print(f"\nWrote {out_prefix}.csv. Raters: read each brief against verified_calculations and fill the rater columns.")


if __name__ == "__main__":
    main()
