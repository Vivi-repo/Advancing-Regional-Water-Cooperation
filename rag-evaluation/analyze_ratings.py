"""Analyze human ratings: accuracy per condition, inter-rater agreement, pairwise win rates, error taxonomy.

Usage:
    python analyze_ratings.py results.csv                  # question-answering ratings
    python analyze_ratings.py results.csv policy_results.csv
    python analyze_ratings.py results.csv --all-pairs      # every condition pair instead of the planned ones

Expected rater columns (blank rows are skipped): correct_r1/_r2, grounded_r1/_r2 (1 or 0),
stakeholder_quality_r1/_r2 (1-5, policy file only), and error_type.

What it reports (maps to the evaluation-plan rubric):
  - mean rating per condition (mean of the two raters; QA split into answerable vs unanswerable)
  - Cohen's kappa between raters (quadratic-weighted for the 1-5 scale); target kappa >= 0.6
  - pairwise win/loss/tie between conditions per query, with an exact two-sided sign test
    and a Bonferroni-adjusted p-value
  - error taxonomy counts per condition
"""
import csv
import itertools
import math
import statistics
import sys


# ---------------------------------------------------------------- statistics
def cohen_kappa(a, b, weighted=False):
    """Cohen's kappa for two raters. weighted=True uses quadratic weights (ordinal scales)."""
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if str(x).strip() != "" and str(y).strip() != ""]
    n = len(pairs)
    if n == 0:
        return None
    cats = sorted({v for p in pairs for v in p})
    if len(cats) < 2:
        return None  # no variation: kappa undefined
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in pairs:
        obs[idx[x]][idx[y]] += 1.0 / n
    r = [sum(obs[i]) for i in range(k)]
    c = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    span = (cats[-1] - cats[0]) or 1.0

    def w(i, j):
        if weighted:
            return ((cats[i] - cats[j]) / span) ** 2
        return 0.0 if i == j else 1.0

    po = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    pe = sum(w(i, j) * r[i] * c[j] for i in range(k) for j in range(k))
    return None if pe == 0 else 1 - po / pe


def sign_test_p(wins, losses):
    """Exact two-sided sign test (ties excluded)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ------------------------------------------------------------------- helpers
def rating(row, metric):
    vals = []
    for r in ("r1", "r2"):
        v = (row.get(f"{metric}_{r}") or "").strip()
        if v != "":
            vals.append(float(v))
    return statistics.mean(vals) if vals else None


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    is_policy = "target_bcm" in rows[0]
    unit = "target_bcm" if is_policy else "id"
    return rows, is_policy, unit


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def planned_pairs(conds, all_pairs=False):
    """Pre-specified comparisons (fewer comparisons = less Bonferroni penalty).
    QA:     each RAG condition vs the no-retrieval baseline; k=3 vs k=1 and k=5 within each prompt style;
            basic vs structured prompt at each k.
    Policy: full vs no_calc, full vs baseline, no_calc vs baseline."""
    if all_pairs:
        return list(itertools.combinations(conds, 2))
    have = set(conds)
    pairs = []
    if "full" in have:
        pairs += [("full", "no_calc"), ("full", "baseline"), ("no_calc", "baseline")]
    base = "baseline(no retrieval)"
    if base in have:
        pairs += [(c, base) for c in conds if c.startswith("k=")]
    for style in ("basic", "structured"):
        for k in (1, 5):
            pairs.append((f"k=3|{style}", f"k={k}|{style}"))
    for k in (1, 3, 5):
        pairs.append((f"k={k}|basic", f"k={k}|structured"))
    return [(a, b) for a, b in pairs if a in have and b in have]


def analyze(path, out_rows, all_pairs=False):
    rows, is_policy, unit = load(path)
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    print(f"\n==================== {path} ({len(rows)} rows, {len(conds)} conditions)")

    rated = [r for r in rows if rating(r, "correct") is not None]
    if not rated:
        print("No ratings found yet. Fill correct_r1/_r2 (and grounded_r1/_r2) first.")
        return
    print(f"Rated rows: {len(rated)} of {len(rows)}")

    # ---- accuracy per condition
    print("\nMean rating per condition (1 = correct / grounded):")
    header = ["condition", "n", "correct", "grounded"] + (["stakeholder_quality(1-5)"] if is_policy else [])
    print(" | ".join(header))
    segments = [("all", rated)]
    if not is_policy:
        segments = [("answerable", [r for r in rated if r["in_scope"] == "1"]),
                    ("unanswerable (correct = refused)", [r for r in rated if r["in_scope"] == "0"])]
    for seg_name, seg_rows in segments:
        if not is_policy:
            print(f"-- {seg_name}")
        for cond in conds:
            sub = [r for r in seg_rows if r["condition"] == cond]
            if not sub:
                continue
            cor = [rating(r, "correct") for r in sub if rating(r, "correct") is not None]
            gro = [rating(r, "grounded") for r in sub if rating(r, "grounded") is not None]
            line = [cond, str(len(sub)), fmt(statistics.mean(cor)) if cor else "n/a", fmt(statistics.mean(gro)) if gro else "n/a"]
            rec = {"file": path, "segment": seg_name, "condition": cond, "n": len(sub),
                   "correct": statistics.mean(cor) if cor else "", "grounded": statistics.mean(gro) if gro else ""}
            if is_policy:
                sq = [rating(r, "stakeholder_quality") for r in sub if rating(r, "stakeholder_quality") is not None]
                line.append(fmt(statistics.mean(sq)) if sq else "n/a")
                rec["stakeholder_quality"] = statistics.mean(sq) if sq else ""
            print(" | ".join(line))
            out_rows.append(rec)

    # ---- inter-rater agreement
    print("\nInter-rater agreement (Cohen's kappa, target >= 0.6):")
    for metric, weighted in (("correct", False), ("grounded", False), ("stakeholder_quality", True)):
        a = [r.get(f"{metric}_r1", "") for r in rows]
        b = [r.get(f"{metric}_r2", "") for r in rows]
        both = sum(1 for x, y in zip(a, b) if str(x).strip() != "" and str(y).strip() != "")
        if both:
            k = cohen_kappa(a, b, weighted)
            agree = statistics.mean(1.0 if float(x) == float(y) else 0.0
                                    for x, y in zip(a, b) if str(x).strip() != "" and str(y).strip() != "")
            note = "" if k is not None else " (undefined: no variation in ratings)"
            print(f"  {metric}: kappa = {fmt(k)}{note}, raw agreement = {agree:.2f}, n = {both}"
                  f"{' (quadratic-weighted)' if weighted else ''}")

    # ---- pairwise win rates on `correct`
    per = {}
    for r in rated:
        s = rating(r, "correct")
        per.setdefault(r["condition"], {}).setdefault(r[unit], []).append(s)
    per = {c: {u: statistics.mean(v) for u, v in d.items()} for c, d in per.items()}
    pairs = planned_pairs([c for c in conds if c in per], all_pairs)
    if pairs:
        print(f"\nPairwise comparison on `correct` per {unit} (wins/losses for A over B; ties excluded from the test; "
              f"Bonferroni over {len(pairs)} pairs):")
        print("A | B | n | A wins | B wins | ties | p | p_bonferroni")
        for a, b in pairs:
            common = sorted(set(per[a]) & set(per[b]))
            if not common:
                continue
            wins = sum(1 for u in common if per[a][u] > per[b][u])
            losses = sum(1 for u in common if per[a][u] < per[b][u])
            ties = len(common) - wins - losses
            p = sign_test_p(wins, losses)
            pb = min(1.0, p * len(pairs))
            print(f"{a} | {b} | {len(common)} | {wins} | {losses} | {ties} | {p:.3f} | {pb:.3f}")
            out_rows.append({"file": path, "segment": "pairwise", "condition": f"{a} vs {b}", "n": len(common),
                             "correct": f"wins={wins};losses={losses};ties={ties}", "grounded": f"p={p:.3f};p_bonf={pb:.3f}"})

    # ---- error taxonomy
    print("\nError taxonomy (rows with an error_type):")
    errs = {}
    for r in rows:
        e = (r.get("error_type") or "").strip()
        if e:
            errs.setdefault(r["condition"], {}).setdefault(e, 0)
            errs[r["condition"]][e] += 1
    if not errs:
        print("  none recorded")
    for cond, d in errs.items():
        print(f"  {cond}: " + ", ".join(f"{e}={n}" for e, n in sorted(d.items(), key=lambda x: -x[1])))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out_rows = []
    all_pairs = "--all-pairs" in sys.argv
    for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
        analyze(path, out_rows, all_pairs)
    if out_rows:
        keys = list(dict.fromkeys(k for r in out_rows for k in r))
        with open("analysis_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(out_rows)
        print("\nWrote analysis_summary.csv")


if __name__ == "__main__":
    main()
