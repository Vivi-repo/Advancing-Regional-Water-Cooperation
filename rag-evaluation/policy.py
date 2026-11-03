"""Policy-brief mode: connects an 'optimal' water-saving pathway to stakeholder interests.

Research question: can data-driven policy bridge the gap between an optimal water
management solution and the interests of Central Asian stakeholders?

How this module serves that question:
  1. "Optimal" is made explicit and computable: the pathway table (calculator.py) gives, for a
     target of X BCM/yr, the hectares/percent needed and the cost of each single measure.
  2. Stakeholder interests are retrieved PER COUNTRY (one retrieval each), so a single top-k
     lookup cannot crowd out four of the five governments.
  3. The LLM only writes the brief; numbers come from code (condition "full"). Ablations:
       "full"     = verified calculations + per-stakeholder retrieval
       "no_calc"  = per-stakeholder retrieval, but the model must do the arithmetic from scaling.txt chunks
       "baseline" = no retrieval, no calculator (plain Llama 3.2)
"""
import re
import time

import calculator as calc
import rag_core as rc

STAKEHOLDERS = ["Uzbekistan", "Turkmenistan", "Afghanistan", "Tajikistan", "Kyrgyzstan"]
STANCE_WORDS = ["Supports", "Conditional", "Opposes", "Unclear", "Not covered"]
STAKEHOLDER_SOURCES = ["government.txt", "info.txt"]   # files describing country positions
SCALING_SOURCES = ["scaling.txt"]


def _context_block(chunks_by_name):
    parts = []
    for name, chunks in chunks_by_name.items():
        text = "\n".join(c["text"] for c in chunks) if chunks else "(nothing retrieved)"
        parts.append(f"[{name}]\n{text}")
    return "\n\n".join(parts)


_FORMAT = """Write the brief in exactly this format:

Recommended pathway: <2-4 sentences naming which measures to prioritize and why>

Stakeholder positions:
- Uzbekistan: <Supports / Conditional / Opposes / Unclear / Not covered> - <one-sentence reason from the context> - <what would make it acceptable>
- Turkmenistan: (same format)
- Afghanistan: (same format)
- Tajikistan: (same format)
- Kyrgyzstan: (same format)

Main risks and open questions: <2-3 sentences>
"""


def build_policy_prompt(target_bcm, condition, calc_text, context_text):
    head = ("You are drafting an exploratory policy brief on reducing water losses in the Amu Darya Basin. "
            "This is decision support, not an official directive.\n\n")
    if condition == "baseline":
        return (head + f"TARGET: save {target_bcm:g} BCM of water per year in the Amu Darya Basin.\n\n"
                "Use your general knowledge.\n\n" + _FORMAT)
    rules = ("Rules:\n- Use ONLY the material below. Do not invent facts, figures or quotes.\n"
             "- If the material does not say what a stakeholder's position is, write \"Not covered\" for that stakeholder.\n")
    if condition == "full":
        return (head + rules + "- Do not do your own arithmetic: copy numbers from VERIFIED CALCULATIONS.\n\n"
                f"TARGET: save {target_bcm:g} BCM of water per year.\n\n"
                f"VERIFIED CALCULATIONS (computed by code from the scaling rules):\n{calc_text}\n\n"
                f"CONTEXT FROM THE DOCUMENTS:\n{context_text}\n\n" + _FORMAT)
    # no_calc: the model must compute from the scaling rules itself
    return (head + rules + "- Use the scaling rules in the context to calculate hectares and costs yourself, showing the numbers.\n\n"
            f"TARGET: save {target_bcm:g} BCM of water per year.\n\n"
            f"CONTEXT FROM THE DOCUMENTS:\n{context_text}\n\n" + _FORMAT)


def policy_brief(target_bcm, collection, condition="full", k_each=2, embed_fn=None, generate_fn=None,
                 temperature=None, seed=None, network_km=None):
    """Return dict(text, rows, chunks, calc_text, prompt, seconds)."""
    assert condition in ("full", "no_calc", "baseline")
    embed_fn = embed_fn or rc.default_embed_fn()
    generate_fn = generate_fn or rc.default_generate_fn()
    t0 = time.time()
    rows = calc.pathway_table(target_bcm, network_km)
    calc_text = calc.format_table(rows)

    chunks = {}
    if condition != "baseline" and collection is not None:
        for name in STAKEHOLDERS:
            q = f"{name} official stance vulnerabilities red lines policy priorities water"
            chunks[name] = rc.retrieve(collection, embed_fn(q), k_each, sources=STAKEHOLDER_SOURCES)
        if condition == "no_calc":
            q = "scaling rules hectares to convert to save BCM cost per hectare drip lining rice wheat"
            chunks["Scaling rules"] = rc.retrieve(collection, embed_fn(q), 4, sources=SCALING_SOURCES)
    prompt = build_policy_prompt(target_bcm, condition, calc_text, _context_block(chunks))
    text = generate_fn(prompt, temperature=temperature, seed=seed)
    return {"text": text, "rows": [r.as_dict() for r in rows], "chunks": chunks, "calc_text": calc_text,
            "prompt": prompt, "seconds": round(time.time() - t0, 2)}


# ------------------------------------------------------------- automatic checks
def stakeholders_covered(text):
    """Count stakeholders that get an explicit stance word within 40 chars of their name."""
    n = 0
    for name in STAKEHOLDERS:
        for m in re.finditer(name, text):
            window = text[m.end(): m.end() + 40]
            if any(w.lower() in window.lower() for w in STANCE_WORDS):
                n += 1
                break
    return n


def number_forms(x):
    """Ways a number might be written, after thousands separators are removed."""
    forms = {f"{x:.0f}", f"{round(x, -3):.0f}"}
    if x >= 1e9:
        forms |= {f"{x / 1e9:.2f} billion", f"{x / 1e9:.1f} billion"}
    if x >= 1e6:
        forms |= {f"{x / 1e6:.0f} million", f"{x / 1e6:.2f} million", f"{x / 1e6:.1f} million"}
    return forms


def _norm(text):
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.lower())


def calc_numbers_correct(text, target_bcm):
    """Share of the three key computed quantities (drip hectares, drip cost, rice hectares)
    that appear in the text in any common format. A proxy; confirm by reading."""
    t = _norm(text)
    wanted = [calc.drip_hectares(target_bcm), calc.drip_cost_usd(target_bcm), calc.rice_to_wheat_hectares(target_bcm)]
    hit = sum(1 for x in wanted if any(f in t for f in number_forms(x)))
    return round(hit / len(wanted), 3)


def numbers_not_in_support(text, support_text):
    """Numbers (>10) in the generated text that are absent from the retrieved/verified material."""
    def nums(s):
        out = set()
        for m in re.findall(r"\d[\d,]*\.?\d*", _norm(s)):
            m = m.rstrip(".,")
            try:
                v = float(m.replace(",", ""))
            except ValueError:
                continue
            if v > 10:
                out.add(m.replace(",", ""))
        return out
    return sorted(nums(text) - nums(support_text))
