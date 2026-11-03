ti# Amu Darya decision-support prototype: app + evaluation

## Files
| file | what it does |
|---|---|
| `rag_core.py` | Shared pipeline: PDF cleaning, 500/50 chunking, ChromaDB index (auto-rebuilds when files change), retrieval, prompts |
| `calculator.py` | Computes hectares / % lined / cost from the scaling.txt rules, so numbers come from code, not the 3B model |
| `policy.py` | Policy-brief mode: verified calculations + one retrieval per stakeholder country -> structured brief |
| `app.py` | Streamlit app: "Ask the documents" and "Policy brief" modes, logs every query to `query_log.jsonl` |
| `evaluate.py` | Q&A evaluation: k = 1/3/5 x prompt style x no-retrieval baseline |
| `evaluate_policy.py` | Policy evaluation: full vs no-calculator vs no-retrieval baseline |
| `analyze_ratings.py` | Accuracy per condition, Cohen's kappa, pairwise win rates + sign test (Bonferroni), error taxonomy |
| `test_queries.csv` | 22 queries from your knowledge base with hand-checked ground truth |

## Setup (once)
    pip install streamlit chromadb ollama pypdf
    ollama pull llama3.2
    ollama pull nomic-embed-text
Put the 15 knowledge-base files in one folder, e.g. `./knowledge_base` (nothing else .txt/.pdf in it).

## Run order
    python evaluate.py --docs ./knowledge_base --queries test_queries.csv
    python evaluate.py --docs ./knowledge_base --queries test_queries.csv --repeats 3 --temperature 0.7 --out consistency --no-baseline --ks 3 --styles basic
    python evaluate_policy.py --docs ./knowledge_base
    streamlit run app.py        # set RAG_DOCS=./knowledge_base first (or run from inside that folder)

Add `--dry-run` to any evaluate script to test the plumbing with fake models (numbers are meaningless).

## Rating (the part that gives you real numbers)
Two team members open `results.csv` and `policy_results.csv` and fill, independently:
- `correct_r1` / `correct_r2`: 1 if the answer is right (for unanswerable questions: 1 if it correctly refused), else 0.
  Compare against `ground_truth` (QA) or `verified_calculations` (policy).
- `grounded_r1` / `grounded_r2`: 1 if every claim is supported by the retrieved chunks, else 0.
- `stakeholder_quality_r1` / `_r2` (policy only): 1-5, are the five governments' positions accurate and specific?
- `error_type` (one per wrong row): retrieval_miss | hallucination | wrong_number | arithmetic_error | false_refusal | missed_refusal | incomplete | other
Then: `python analyze_ratings.py results.csv policy_results.csv`

## Useful options
    --chunk-size 1000 --overlap 100     different chunking (index rebuilds automatically)
    --no-clean                          keep web-page boilerplate in the PDFs (ablation)
    --ks 1,3,5,10                       more k values
    --targets 2,3,5,8                   policy targets in BCM/yr
Query-file columns: `expected_source` may list files separated by `;`; `expected_keywords` separated by `;`
(`|` inside one keyword = alternatives); numbers match with or without thousands separators.
