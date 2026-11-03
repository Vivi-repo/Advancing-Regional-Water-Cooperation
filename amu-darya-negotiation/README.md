# Amu Darya Multiagent Negotiation

A RAG grounded, hub and spoke multiagent workflow in which LLM delegates for Afghanistan, Tajikistan, Uzbekistan and Turkmenistan negotiate Amu Darya water allocation through a neutral mediator. It doubles as an experimental harness: persona framing, payoff visibility and document grounding are switchable factors, so every run measures how those design choices change the outcome.

## Why it is built this way

**Manoranjan and Gaikwad, "When Identity Overrides Incentives" (FAccT 2026)** show that giving agents role personas shifts equilibrium attainment by up to 90 percentage points, even when agents see full payoffs. Their study is single shot, binary action and ungrounded, and they name multiturn deliberation and richer action spaces as untested. This workflow keeps their 2 x 2 design (persona x payoff visibility) and carries it into exactly that untested setting: multiround bargaining over continuous allocations, with a third factor for document grounding. If persona wording decides whether Uzbekistan's agent accepts a worse deal than walking away, that is a finding about the simulation, not about Uzbekistan, and the harness is built to catch it.

**Guo et al., "LLM based Multi Agents: A Survey of Progress and Challenges" (IJCAI 2024)** supplies the architecture vocabulary. Each design choice maps to one of its dimensions:

| Survey dimension | Choice here |
|---|---|
| Agents environment interface | Sandbox: a basin water ledger that checks feasibility and computes payoffs |
| Agent profiling | Predefined personas (the experimental factor) plus data derived briefings retrieved from the corpus |
| Communication paradigm | Competitive bargaining among delegates, cooperative mediator |
| Communication structure | Hub and spoke: a centralized mediator, a shared public record, and confidential caucus channels through the mediator |
| Feedback | Environment feedback (feasibility errors, payoffs) and agent feedback (votes, amendments) |
| Memory | Each delegate's caucus thread, retrieved passages and past positions |
| Capability acquisition | Goal files updated by events (controlled), tool use through lookups and what ifs |
| Challenge: hallucination cascades | A verifier checks every cited claim against the cited passage before it enters the public record |
| Challenge: evaluation | Game theoretic reference points and fixed metrics computed by code, not by an LLM judge |

## Agents and tools (hub and spoke)

| Agent | Role | Sees | Never sees |
|---|---|---|---|
| Country delegate x4 | Afghanistan, Tajikistan (upstream), Uzbekistan, Turkmenistan (downstream). Bargains for its goal file | Shared record, its own goals, its own caucus thread, payoffs per condition | Other delegates' goals and caucus |
| Mediator (hub) | Tables the single text each round, reads every caucus, replies privately | Everything shared plus all caucus messages | Nothing hidden, but it has no payoff |
| Legal advisor (spoke) | Reviews each draft against water law with citations; no stake, no vote | Drafts and `doc_type: legal` passages | Payoffs and goals |
| Verifier | Checks each cited claim against the cited passage | Claims and passages | n/a |
| Human gate | Optional: approves a unanimous text before it counts (`--hitl final`) | Everything | n/a |

| Tool | Requested through | What it returns |
|---|---|---|
| `lookup` | `"lookups": [...]` in a delegate's reply | Up to 4 more passages from the document library, added to that delegate's context |
| `simulate` | `"what_if": {...}` in a delegate's reply | Feasibility, water to each country and the Aral delta, payoff per condition |
| hydrology check | automatic on every mediator draft | Errors the mediator must fix (up to 2 repairs) |

Tools are requested through JSON fields and run by the orchestrator, so they work with small local models that lack reliable native function calling, and every call is logged.

**Goals are data, not prompt text.** `config/personas.yaml` holds identity (the experimental factor), `config/goals.yaml` holds each government's objectives, priorities, red lines and fallback, and `config/events.yaml` holds event sets that change goal files at given rounds. Agents can only request goal changes; requests are logged, not applied. Every vote records the goal version in force.

## How one episode runs

```
Round 0   4 delegates ──► opening asks (public) + caucus notes (to mediator)
Round r   scenario controller ──► due events: public news + goal file updates
          Mediator ─────► single negotiating text, private caucus replies
          hydrology ────► feasibility check (mediator repairs up to twice)
          Legal advisor ► public review with citations
          4 delegates ──► vote, amend, caucus, request lookups and what ifs (simultaneous)
          verifier ─────► checks citations before the public record
          all accept? ──► human gate (optional) ──► DEAL      else next round
After max rounds ───────► NO DEAL (unilateral takes, no delta flow, conflict cost)
```

## Connecting to the team's RAG pipeline

The agents read the same ChromaDB folder and use the same Ollama `nomic-embed-text` embeddings as the Streamlit app.

1. In `config/rag.yaml` set `backend: chroma` and `chroma.db_path` to the app's `chroma_db` folder.
2. Check retrieval: `python -m wnego search "Rogun winter releases" --rag chroma`
3. Run: `python -m wnego run --backend ollama --model qwen2.5:32b --rag chroma`

The app's `local_docs` collection works as is. Its chunk ids are counters, so passage ids look like `rogun_report#17`; they stay valid only while that database is not rebuilt. For stable ids, page numbers and a `doc_type` filter for the legal advisor, build the improved collection alongside it (the app is untouched):

```bash
python -m wnego index --docs ../data/raw --db ../chroma_db      # creates amu_darya_v2, incremental on reruns
python -m wnego run --backend ollama --model qwen2.5:32b --rag chroma --collection amu_darya_v2
```

Tag documents for the legal advisor with a `docs_meta.yaml` in the documents folder, for example `un_watercourses_convention.pdf: {doc_type: legal}`; markdown files can carry `doc_type` in front matter.

Use a model of at least 14B parameters for the agents (Qwen2.5 32B is in the FAccT paper and on Ollama). `llama3.2` (3B) is fine for testing retrieval but too small for strategic negotiation. The Ollama backend sets a 16k context window; Ollama's default silently truncates these prompts.

A proposal is a full treaty text: growing season and winter volumes for each country, how much Tajikistan shifts from summer to winter through storage, side payments between countries, and free text provisions. The seasonal shift is where the real conflict sits: winter release powers Tajikistan, summer release irrigates Uzbekistan and Turkmenistan.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                                   # offline tests
python -m wnego benchmarks --scenario dry             # no deal, Protocol 566, Nash bargaining, utilitarian
python -m wnego run --backend mock --scenario dry     # one episode with the rule based mock
python -m wnego experiment config/experiment.yaml     # factorial sweep, writes runs/<name>/
```

The mock backend only tests the plumbing. It ignores persona and payoff framing by construction, so its outputs say nothing about LLM behaviour.

## Running real models

```bash
# Local, on the same Ollama server as the RAG app
python -m wnego run --backend ollama --model qwen2.5:32b --rag chroma --events qosh_tepa_shock

export ANTHROPIC_API_KEY=...
python -m wnego run --backend anthropic --model claude-sonnet-5-5 --persona rich --payoff own --scenario dry

# The open models from the FAccT paper, served locally with vLLM or Ollama
python -m wnego run --backend openai_compat --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-32B-Instruct
```

For a sweep, list models in `config/experiment.yaml`. Each episode is roughly 4 openings plus up to 6 rounds x (1 mediator draft + 1 legal review + 4 responses + verifier calls), so budget about 50 to 70 calls per episode.

## Experimental factors

| Factor | Levels | Source of the idea |
|---|---|---|
| `persona` | `none` (role label plus "act strategically"), `rich` (identity narrative), `rich_v2`, `rich_v3` (wording variants) | Persona arm and prompt sensitivity variants in the FAccT paper |
| `payoff` | `hidden` (narrative only), `own` (own value of each text and of no deal), `full` (everyone's values) | Payoff visibility arm |
| `grounding` | `rag` (cited briefing passages), `none` | New: does grounding change outcomes or just the talk? |
| `legal` | `on`, `off` | Does the legal advisor shift outcomes toward its principles? |
| `events` | `none`, `qosh_tepa_shock`, `energy_crisis` (see `config/events.yaml`) | Dynamic goals: do agents adapt, and how? |
| `scenario` | `normal`, `dry`, `wet` flow years | Scarcity as stress test |
| model, seed | any | Model family effects |

## Metrics (per episode, in `results.csv`)

| Metric | What it tells you |
|---|---|
| `agreement`, `rounds` | Analogue of equilibrium attainment |
| `below_batna_accept_rate` | Share of accept votes for a text worth less to that country than no deal. A payoff following agent never does this. The cleanest signal of identity overriding incentives |
| `payoff_consistent_rate` | Accept exactly when the text beats no deal. Rejection above no deal can be strategic hold out, so read with the rate above |
| `gain_ratio_vs_nbs`, `nbs_distance_km3` | How the deal compares with the Nash bargaining solution |
| `anchor_566_distance_km3` | Distance from the Protocol 566 status quo. Tests whether retrieved precedent anchors the outcome |
| `min_gain_share` | Equity: smallest country gain relative to the mean |
| `delta_flow_met` | Whether the Aral delta minimum survives the deal |
| `claims_*_rate` | Citation fidelity: verified, partly verified, unverified, nonexistent ids |
| `legal_concerns` | Legal advisor findings of concern or inconsistency |
| `lookups`, `what_ifs`, `caucus_messages`, `goal_requests` | How agents used tools, the hub, and requests to change their own goals |

`stats.txt` reports, per model and scenario, an omnibus chi square on agreement across conditions with Cramer's V and Holm corrected pairwise Fisher tests, matching the FAccT paper's analysis. Full transcripts, including each agent's private reasoning, go to `runs/<name>/transcripts/` for the qualitative analysis.

## Before trusting any result

1. **Calibrate `config/basin.yaml`.** Every number tagged ILLUSTRATIVE is a placeholder that creates a real tension, not a measured value. Replace each with a sourced figure and cite it in the comment.
2. **Grow the corpus.** `corpus/` holds five seed summaries (four background, one legal) built from IDSA, Carnegie and climate diplomacy sources. Add primary texts (the 1992 Almaty Agreement, ICWC protocols, World Bank Rogun assessments, CAREC and UNECE basin reports). Drop `.md`, `.txt` or `.pdf` files in and the retriever indexes them.
3. **Review goals.yaml.** Red lines marked DRAFT are placeholders; each line should trace to a source.
4. **Treat personas as governance decisions.** Write them from the corpus, keep them neutral, and version them. The paper's core point is that this text decides outcomes.
5. **Record the git hash.** `run_info.json` stores it so every result ties to the exact prompts used.

## Research questions this can answer

1. Does the persona effect on equilibrium selection survive multiround bargaining with a mediator, or does deliberation wash it out?
2. Does seeing payoffs make agents walk away more often, as the no persona plus visible payoff arm did for Qwen?
3. Does RAG grounding change the allocation, or only the citations? Does it anchor deals to Protocol 566?
4. Does Afghanistan, absent from every existing agreement, get systematically less than the Nash bargaining share, and does that depend on persona wording?
5. Do dry years push agents toward no deal, and does that interact with persona?

## File map

```
config/basin.yaml        environment: flows, storage, payoff parameters, no deal rule
config/personas.yaml     persona conditions and wording variants (identity)
config/goals.yaml        each government's objectives, priorities, red lines (interests, as data)
config/events.yaml       event sets that update goal files during a negotiation
config/rag.yaml          retriever: bm25, or the team's Chroma + Ollama pipeline
config/experiment.yaml   factorial design and model list
corpus/                  documents for retrieval (seed summaries with source URLs)
wnego/environment.py     water ledger, feasibility, payoffs
wnego/benchmarks.py      Nash bargaining solution and utilitarian optimum
wnego/retrieval.py       Chroma adapter for the team's pipeline, improved indexer, BM25 fallback
wnego/goals.py           goal files, versioning and the scenario controller
wnego/prompts.py         every prompt, in one auditable place
wnego/agents.py          delegates, mediator, legal advisor, verifier, toolbox
wnego/protocol.py        episode orchestration
wnego/evaluation.py      metrics
wnego/experiment.py      factorial runner and statistics
wnego/llm.py             Ollama, Anthropic, OpenAI compatible and mock backends
tests/test_smoke.py      offline tests
tests/test_chroma.py     runs the agents on a Chroma DB built exactly like the RAG app's
```
