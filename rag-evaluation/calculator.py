"""Deterministic water-saving cost calculator.

Implements the scaling rules in scaling.txt so that numbers in policy outputs are
COMPUTED by code, not guessed by a 3B language model. If you change a number in
scaling.txt, change it here too (single source of truth for the evaluation).

Constants and where they come from (scaling.txt unless noted):
  rule 1  canal lining:       every 1% of the primary/secondary network lined saves ~0.30 BCM/yr (range 0.25-0.35)
  rule 2  cotton drip:        12,000 -> 5,000 m3/ha, so 7,000 m3 saved per converted hectare
  rule 3  rice -> wheat:      18,000 -> 5,000 m3/ha, so 13,000 m3 saved per converted hectare
  rule 4  costs:              drip kit ~$1,800/ha; geomembrane ~$8.50/m2 = ~$850,000/km for a 10 m wide canal
  soln.txt                    rice -> wheat has near-zero structural cost (policy/subsidy costs not estimated)
  infrastructure.txt          total seepage loss ~20-25 BCM/yr (upper limit on what lining can save)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

DRIP_SAVING_M3_PER_HA = 7_000
DRIP_COST_USD_PER_HA = 1_800
RICE_WHEAT_SAVING_M3_PER_HA = 13_000
LINING_BCM_PER_PERCENT = 0.30
LINING_BCM_PER_PERCENT_RANGE = (0.25, 0.35)
GEOMEMBRANE_USD_PER_KM = 850_000
TOTAL_SEEPAGE_LOSS_BCM = (20, 25)
BCM_TO_M3 = 1_000_000_000


@dataclass
class Measure:
    measure: str
    quantity: float          # hectares, or % of network
    unit: str
    cost_usd: float | None   # None = cannot be computed from the knowledge base
    note: str

    def as_dict(self):
        return asdict(self)


def drip_hectares(target_bcm):
    return target_bcm * BCM_TO_M3 / DRIP_SAVING_M3_PER_HA


def drip_cost_usd(target_bcm):
    return drip_hectares(target_bcm) * DRIP_COST_USD_PER_HA


def rice_to_wheat_hectares(target_bcm):
    return target_bcm * BCM_TO_M3 / RICE_WHEAT_SAVING_M3_PER_HA


def lining_percent(target_bcm, bcm_per_percent=LINING_BCM_PER_PERCENT):
    return target_bcm / bcm_per_percent


def lining_cost_usd(target_bcm, network_km=None):
    """Needs the total primary/secondary network length, which is NOT in the knowledge base."""
    if network_km is None:
        return None
    return lining_percent(target_bcm) / 100 * network_km * GEOMEMBRANE_USD_PER_KM


def pathway_table(target_bcm, network_km=None):
    """Single-measure pathways that each deliver `target_bcm` BCM/yr on their own."""
    lo, hi = LINING_BCM_PER_PERCENT_RANGE
    pct = lining_percent(target_bcm)
    lining_note = (f"scaling.txt rule 1 (0.30 BCM per 1%; range {lo}-{hi} gives "
                   f"{target_bcm / hi:.0f}-{target_bcm / lo:.0f}%). ")
    if network_km is None:
        lining_note += "Cost needs total network length in km (not in the documents)."
    if pct > 100 or target_bcm > TOTAL_SEEPAGE_LOSS_BCM[0]:
        lining_note += (" WARNING: target exceeds the 20-25 BCM total seepage loss cited in the documents, "
                        "so lining alone cannot deliver it.")
    return [
        Measure("Cotton drip irrigation", drip_hectares(target_bcm), "ha converted", drip_cost_usd(target_bcm),
                "scaling.txt rules 2 and 4 (7,000 m3/ha saved, $1,800/ha)"),
        Measure("Rice-to-wheat switching", rice_to_wheat_hectares(target_bcm), "ha switched", 0.0,
                "scaling.txt rule 3 (13,000 m3/ha saved); structural cost near zero per soln.txt, "
                "subsidy and quota-reform costs not estimated"),
        Measure("Canal lining (geomembrane)", pct, "% of primary/secondary network", lining_cost_usd(target_bcm, network_km),
                lining_note),
    ]


def format_table(rows):
    lines = []
    for r in rows:
        cost = "cannot be computed from the documents" if r.cost_usd is None else f"${r.cost_usd:,.0f}"
        qty = f"{r.quantity:,.0f}" if r.unit.startswith("ha") else f"{r.quantity:.1f}"
        lines.append(f"- {r.measure}: {qty} {r.unit}; cost: {cost}. Basis: {r.note}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    print(f"Target: {t} BCM per year")
    print(format_table(pathway_table(t)))
