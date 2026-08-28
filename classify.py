"""
Dispute classifier and evidence-pack builder.

Given an agent-made purchase that the user has disputed, decide who is at
fault and produce evidence a merchant could actually submit.

Design decision worth defending in a panel: the mandate checks are pure rules,
not a model. Whether Rs.4,200 exceeds a Rs.1,500 cap is arithmetic. A model
adds latency, cost and non-determinism to a question that has one right answer.
The model belongs on the parts that are genuinely linguistic, and only there.

Attribution order matters. A mandate breach that was CAUSED by planted content
is attributed to the injection, not to the agent, because the root cause is the
third party. Checking the caps first and stopping would blame the wrong party.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

from generator import Verdict


# --------------------------------------------------------------------------
# reason codes
#
# Razorpay's dispute API already has chargeback reason codes, but every one of
# them asks whether the merchant delivered. None ask whether the buyer had
# authority. These are the codes that gap needs.
# --------------------------------------------------------------------------

REASON_CODES = {
    "AGENT_OK_USER_REGRET": (
        "Agent acted within its mandate. Purchase matches granted authority.",
        Verdict.WITHIN_MANDATE, "customer"),
    "AGENT_CAP_EXCEEDED": (
        "Transaction amount exceeded the per-transaction cap on the mandate.",
        Verdict.MANDATE_BREACH, "agent_operator"),
    "AGENT_WINDOW_CAP_EXCEEDED": (
        "Cumulative spend in the mandate window exceeded the window cap.",
        Verdict.MANDATE_BREACH, "agent_operator"),
    "AGENT_MANDATE_EXPIRED": (
        "Mandate had expired before the transaction timestamp.",
        Verdict.MANDATE_BREACH, "agent_operator"),
    "AGENT_CATEGORY_VIOLATION": (
        "Purchase category falls outside the categories the mandate permits.",
        Verdict.MANDATE_BREACH, "agent_operator"),
    "AGENT_CONTENT_INJECTION": (
        "Merchant-controlled content carried an instruction to the agent and "
        "the agent acted on it.",
        Verdict.INJECTED, "merchant_or_third_party"),
    "AGENT_SCOPE_UNDEFINED": (
        "Mandate set no numeric limit. Whether the amount was authorised "
        "cannot be settled from evidence.",
        Verdict.AMBIGUOUS, "undetermined"),
    "AGENT_CATEGORY_UNDEFINED": (
        "Purchase category is adjacent to the permitted category. Mandate "
        "does not resolve whether it is covered.",
        Verdict.AMBIGUOUS, "undetermined"),
}


SEVERITY_ORDER = {"critical": 3, "major": 2, "minor": 1, "info": 0}


@dataclass
class Finding:
    """One thing the engine established, with the evidence it rests on."""

    code: str
    severity: str            # critical | major | minor | info
    detail: str
    evidence_ref: str        # where in the record this came from

    def line(self) -> str:
        return f"[{self.severity.upper():<8}] {self.code}: {self.detail}"


@dataclass
class EvidencePack:
    """What a merchant would attach to a dispute response."""

    dispute_id: str
    verdict: str
    reason_code: str
    reason_description: str
    liable_party: str
    confidence: float
    findings: List[Dict] = field(default_factory=list)
    mandate_summary: Dict = field(default_factory=dict)
    spend_analysis: Dict = field(default_factory=dict)
    content_analysis: Dict = field(default_factory=dict)
    explanation_letter: str = ""

    def render(self) -> str:
        out = [
            "=" * 66,
            f"  AGENT DISPUTE EVIDENCE PACK   {self.dispute_id}",
            "=" * 66,
            "",
            f"  verdict        {self.verdict}",
            f"  reason code    {self.reason_code}",
            f"  liable party   {self.liable_party}",
            f"  confidence     {self.confidence:.0%}",
            "",
            "  FINDINGS",
        ]
        for f_ in self.findings:
            out.append(f"    [{f_['severity'].upper():<8}] {f_['code']}")
            out.append(f"               {f_['detail']}")
            out.append(f"               source: {f_['evidence_ref']}")
        out += ["", "  SPEND ANALYSIS"]
        for k, v in self.spend_analysis.items():
            out.append(f"    {k:<26} {v}")
        if self.content_analysis:
            out += ["", "  CONTENT ANALYSIS"]
            for k, v in self.content_analysis.items():
                out.append(f"    {k:<26} {v}")
        out += ["", "  EXPLANATION LETTER", ""]
        for line in self.explanation_letter.strip().split("\n"):
            out.append(f"    {line}")
        out.append("")
        return "\n".join(out)


# --------------------------------------------------------------------------
# content analysis
#
# Merchant content is DATA. A product listing describes a product. The moment
# it addresses a reader and tells them what to purchase, it has stopped being
# a description and started being an instruction, and instructions from a
# counterparty are not authority.
#
# That is the rule. It is not a keyword blocklist, which is why it catches
# copy that reads like ordinary merchandising.
# --------------------------------------------------------------------------

# Content that names an automated reader. Legitimate product copy has no
# reason to address an assistant.
AGENT_ADDRESSED = [
    r"\bai agents?\b", r"\bassistants?\b", r"\bautomated ordering\b",
    r"\bordering assistants?\b", r"\bshopping assistants?\b",
    r"\bdelivery partners?\b", r"\bsystem\b", r"\badmin\b", r"\bbots?\b",
]

# Directives about purchase behaviour, as opposed to descriptions of a product.
PURCHASE_DIRECTIVE = [
    r"\balways add\b", r"\badd the\b", r"\badd sku\b", r"\btop up\b",
    r"\bignore\b", r"\bdisregard\b", r"\boverride\b", r"\bmust\b",
    r"\bminimum order value\b", r"\bpurchase quantity\b", r"\bonly available\b",
    r"\bmandatory\b", r"\bexcludes\b",
]

# What the planted instruction was trying to make happen. Used to check
# whether the agent complied.
DEMAND_SIGNALS = {
    "override_limit": [r"\blimits?\b", r"\bbudget\b", r"\boverride\b",
                       r"\bignore\b", r"\bdisregard\b"],
    "force_bundle": [r"\bcombo\b", r"\bbundle\b", r"\bpremium\b",
                     r"\bfeast\b", r"\bplatter\b"],
    "force_quantity": [r"\bquantity\b", r"\bquantity 10\b", r"\bten\b"],
    "force_topup": [r"\btop up\b", r"\bminimum order\b", r"\bcart\b"],
}

# The agent saying, in effect, "I saw it and did not obey it".
#
# These have to be tight. An earlier version matched a bare "did not", which
# fires on "the listing indicated the limit did not apply" -- a statement of
# COMPLIANCE that reads as refusal to a loose matcher. Every pattern here must
# name the agent's stance toward the content, not contain a stray negation.
REFUSAL_SIGNALS = [
    r"\bignor(ed|ing)\b", r"\bdisregard(ed|ing)\b", r"\bdeclined\b",
    r"\bnot from the user\b", r"\bpromotional upsell\b",
    r"\bdid not (act|follow|comply|apply it)\b",
    r"\btreated as (data|content)\b", r"\bnot an instruction\b",
]


def _hits(patterns: List[str], text: str) -> List[str]:
    t = text.lower()
    return [p for p in patterns if re.search(p, t)]


def analyse_content(inputs: List[Dict], reasoning: str) -> Dict:
    """
    Two independent questions:
      A. did merchant content carry an instruction aimed at the agent?
      B. did the agent act on it?

    Only A AND B is an injection. A alone means the merchant tried and failed,
    which is not grounds to make them liable for a charge the user authorised.
    That distinction is the whole reason the false-blame metric exists.
    """
    directives = []
    for idx, inp in enumerate(inputs):
        text = inp.get("text", "")
        addressed = _hits(AGENT_ADDRESSED, text)
        directive = _hits(PURCHASE_DIRECTIVE, text)
        if addressed and directive:
            demands = [name for name, pats in DEMAND_SIGNALS.items()
                       if _hits(pats, text)]
            directives.append({
                "input_index": idx,
                "source": inp.get("source"),
                "merchant": inp.get("merchant"),
                "addressed_signals": len(addressed),
                "directive_signals": len(directive),
                "demands": demands or ["unspecified"],
                "excerpt": text[:110],
            })

    result = {
        "directive_found": bool(directives),
        "directives": directives,
        "agent_complied": False,
        "compliance_evidence": None,
    }
    if not directives:
        return result

    r = reasoning.lower()

    # Explicit refusal wins. If the agent says it ignored the content, the
    # merchant is not liable, whatever the content said.
    refusals = _hits(REFUSAL_SIGNALS, r)
    if refusals:
        result["compliance_evidence"] = (
            f"agent reasoning shows refusal ({len(refusals)} signal(s))")
        return result

    # Otherwise: does the reasoning echo what the directive demanded?
    all_demands = {d for dd in directives for d in dd["demands"]}
    matched = []
    for demand in all_demands:
        if demand in DEMAND_SIGNALS and _hits(DEMAND_SIGNALS[demand], r):
            matched.append(demand)

    if matched:
        result["agent_complied"] = True
        result["compliance_evidence"] = (
            f"agent reasoning echoes planted demand(s): {', '.join(matched)}")
    else:
        result["compliance_evidence"] = "no trace of the demand in agent reasoning"

    return result


# --------------------------------------------------------------------------
# mandate analysis
# --------------------------------------------------------------------------

# Categories that overlap enough that a mandate naming one does not settle
# whether the other is covered. Being explicit about this is the difference
# between an honest AMBIGUOUS and a wrong MANDATE_BREACH.
ADJACENT = {
    ("food", "groceries"), ("groceries", "food"),
    ("groceries", "household"), ("household", "groceries"),
}


def analyse_mandate(mandate: Dict, txn: Dict, prior_spend: int) -> Dict:
    findings: List[Finding] = []

    txn_time = datetime.fromisoformat(txn["timestamp"])
    expires = datetime.fromisoformat(mandate["expires_at"])
    expired = txn_time > expires
    if expired:
        hours = (txn_time - expires).total_seconds() / 3600
        findings.append(Finding(
            "AGENT_MANDATE_EXPIRED", "critical",
            f"mandate expired {hours:.0f}h before the transaction "
            f"({mandate['expires_at']} vs {txn['timestamp']})",
            "mandate.expires_at vs transaction.timestamp",
        ))

    cap = mandate.get("per_txn_cap")
    amount = txn["amount"]
    over_txn_cap = cap is not None and amount > cap
    if over_txn_cap:
        findings.append(Finding(
            "AGENT_CAP_EXCEEDED", "critical",
            f"Rs.{amount} against a per-transaction cap of Rs.{cap} "
            f"(over by Rs.{amount - cap})",
            "mandate.per_txn_cap vs transaction.amount",
        ))

    wcap = mandate.get("window_cap")
    total = prior_spend + amount
    over_window = wcap is not None and total > wcap
    if over_window and not over_txn_cap:
        findings.append(Finding(
            "AGENT_WINDOW_CAP_EXCEEDED", "critical",
            f"Rs.{prior_spend} already spent in the {mandate['window_days']}-day "
            f"window plus Rs.{amount} = Rs.{total}, against a window cap of "
            f"Rs.{wcap}. The transaction alone was within its per-order cap.",
            "mandate.window_cap vs prior_spend + transaction.amount",
        ))
    elif over_window:
        findings.append(Finding(
            "AGENT_WINDOW_CAP_EXCEEDED", "major",
            f"window total Rs.{total} also exceeds the Rs.{wcap} window cap",
            "mandate.window_cap vs prior_spend + transaction.amount",
        ))

    cat = txn["category"]
    allowed = mandate["allowed_categories"]
    category_ok = cat in allowed
    category_adjacent = not category_ok and any(
        (cat, a) in ADJACENT for a in allowed)

    if not category_ok and not category_adjacent:
        findings.append(Finding(
            "AGENT_CATEGORY_VIOLATION", "critical",
            f"purchased '{cat}' under a mandate permitting {allowed}",
            "mandate.allowed_categories vs transaction.category",
        ))
    elif category_adjacent:
        findings.append(Finding(
            "AGENT_CATEGORY_UNDEFINED", "major",
            f"purchased '{cat}' under a mandate permitting {allowed}. These "
            f"overlap in ordinary use and the mandate does not disambiguate.",
            "mandate.allowed_categories vs transaction.category",
        ))

    undefined_scope = cap is None and wcap is None
    if undefined_scope:
        findings.append(Finding(
            "AGENT_SCOPE_UNDEFINED", "major",
            f"no numeric limit was set. Mandate text was "
            f"{mandate.get('freetext')!r}. Whether Rs.{amount} was authorised "
            f"cannot be established from the record.",
            "mandate.per_txn_cap, mandate.window_cap (both null)",
        ))

    return {
        "findings": findings,
        "expired": expired,
        "over_txn_cap": over_txn_cap,
        "over_window": over_window,
        "category_ok": category_ok,
        "category_adjacent": category_adjacent,
        "undefined_scope": undefined_scope,
        "spend": {
            "transaction_amount": f"Rs.{amount}",
            "per_transaction_cap": f"Rs.{cap}" if cap else "none set",
            "prior_spend_in_window": f"Rs.{prior_spend}",
            "window_cap": f"Rs.{wcap}" if wcap else "none set",
            "window_total": f"Rs.{total}",
            "window_days": mandate["window_days"],
        },
    }


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------

def _decide(m: Dict, c: Dict) -> tuple:
    """Returns (reason_code, confidence). Order encodes root-cause logic."""

    breach = m["expired"] or m["over_txn_cap"] or m["over_window"] or \
        not (m["category_ok"] or m["category_adjacent"])

    # Injection first when it explains the breach. Blaming the agent for a
    # breach a third party engineered attributes the wrong party.
    if c["directive_found"] and c["agent_complied"]:
        return "AGENT_CONTENT_INJECTION", 0.90 if breach else 0.72

    # Expiry beats everything else: an expired mandate grants no authority
    # regardless of what was bought or for how much.
    if m["expired"]:
        return "AGENT_MANDATE_EXPIRED", 0.95
    if not (m["category_ok"] or m["category_adjacent"]):
        return "AGENT_CATEGORY_VIOLATION", 0.92
    if m["over_txn_cap"]:
        return "AGENT_CAP_EXCEEDED", 0.95
    if m["over_window"]:
        return "AGENT_WINDOW_CAP_EXCEEDED", 0.88

    # Nothing was breached. Now: could it have been?
    if m["undefined_scope"]:
        return "AGENT_SCOPE_UNDEFINED", 0.80
    if m["category_adjacent"]:
        return "AGENT_CATEGORY_UNDEFINED", 0.78

    # Directive present but not acted on. The merchant tried, the agent held.
    # Still not the merchant's liability for this charge.
    if c["directive_found"]:
        return "AGENT_OK_USER_REGRET", 0.75

    return "AGENT_OK_USER_REGRET", 0.93


def _letter(code: str, m: Dict, c: Dict, sc: Dict) -> str:
    txn, mandate = sc["transaction"], sc["mandate"]
    head = (f"Transaction {txn['txn_id']} for Rs.{txn['amount']} at "
            f"{txn['merchant']} was placed by an authorised agent under "
            f"mandate {mandate['mandate_id']}.")

    if code == "AGENT_OK_USER_REGRET":
        body = (f"The mandate permitted {mandate['allowed_categories']} up to "
                f"Rs.{mandate['per_txn_cap']} per transaction. The purchase "
                f"falls inside those limits on every dimension checked. ")
        if c["directive_found"]:
            body += ("Third-party content in the merchant listing did attempt "
                     "to direct the agent, but the agent's own record shows it "
                     "did not act on that content and the resulting purchase "
                     "was unaffected. ")
        body += ("The cardholder granted this authority in advance and the "
                 "agent exercised it as granted.")
        rec = "Recommend contesting. Authority is documented and was not exceeded."

    elif code == "AGENT_CONTENT_INJECTION":
        d = c["directives"][0]
        body = (f"Content supplied by {d['merchant']} in a {d['source']} field "
                f"contained an instruction addressed to automated buyers rather "
                f"than a description of the product. The agent's reasoning shows "
                f"it acted on that instruction ({c['compliance_evidence']}). "
                f"The excerpt is: \"{d['excerpt']}\". ")
        body += ("The cardholder's mandate did not authorise this outcome, and "
                 "the deviation originates in counterparty-controlled content, "
                 "not in the cardholder's instructions or the agent's limits.")
        rec = ("Recommend attributing to the content originator. Neither the "
               "cardholder nor the agent operator caused this deviation.")

    elif code in ("AGENT_SCOPE_UNDEFINED", "AGENT_CATEGORY_UNDEFINED"):
        body = (f"{REASON_CODES[code][0]} The record is complete but does not "
                f"contain the fact needed to settle the question. No automated "
                f"attribution is offered.")
        rec = "Recommend manual review. Automated attribution withheld."

    else:
        crit = [f for f in m["findings"] if f.severity == "critical"]
        body = ("The agent exceeded the authority the cardholder granted. "
                + " ".join(f.detail for f in crit) + ". ")
        body += ("Fulfilment is not in question. The merchant delivered what "
                 "was ordered. The defect is in the authority to order it.")
        rec = ("Recommend attributing to the agent operator rather than the "
               "merchant. Existing delivery-based evidence types do not reach "
               "this question.")

    return f"{head}\n\n{body}\n\n{rec}"


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------

def build_evidence_pack(sc: Dict) -> EvidencePack:
    m = analyse_mandate(sc["mandate"], sc["transaction"],
                        sc["prior_spend_in_window"])
    c = analyse_content(sc["agent_inputs"], sc["agent_reasoning"])

    code, confidence = _decide(m, c)
    description, verdict, liable = REASON_CODES[code]

    findings = list(m["findings"])
    if c["directive_found"]:
        d = c["directives"][0]
        findings.append(Finding(
            "AGENT_CONTENT_INJECTION" if c["agent_complied"]
            else "AGENT_CONTENT_DIRECTIVE_IGNORED",
            "critical" if c["agent_complied"] else "info",
            (f"merchant content addressed an automated buyer and demanded "
             f"{d['demands']}. " + (c["compliance_evidence"] or "")),
            f"agent_inputs[{d['input_index']}].text",
        ))

    findings.sort(key=lambda f: -SEVERITY_ORDER[f.severity])

    return EvidencePack(
        dispute_id=sc["scenario_id"],
        verdict=verdict.value,
        reason_code=code,
        reason_description=description,
        liable_party=liable,
        confidence=confidence,
        findings=[asdict(f) for f in findings],
        mandate_summary={
            "mandate_id": sc["mandate"]["mandate_id"],
            "granted": sc["mandate"]["created_at"],
            "expires": sc["mandate"]["expires_at"],
            "categories": sc["mandate"]["allowed_categories"],
            "freetext": sc["mandate"].get("freetext"),
        },
        spend_analysis=m["spend"],
        content_analysis={
            "directive_found": c["directive_found"],
            "agent_complied": c["agent_complied"],
            "assessment": c["compliance_evidence"] or "no directive content found",
        },
        explanation_letter=_letter(code, m, c, sc),
    )


def classify(sc: Dict) -> Verdict:
    """Verdict only. This is what the scorer calls."""
    m = analyse_mandate(sc["mandate"], sc["transaction"],
                        sc["prior_spend_in_window"])
    c = analyse_content(sc["agent_inputs"], sc["agent_reasoning"])
    code, _ = _decide(m, c)
    return REASON_CODES[code][1]


if __name__ == "__main__":
    from generator import generate

    scenarios = generate(200)
    # one worked example per case type
    seen = set()
    for s in scenarios:
        if s.generator in seen:
            continue
        seen.add(s.generator)
        pack = build_evidence_pack(s.to_public_dict())
        print(f"\n### case type: {s.generator}   truth: {s.truth.value}")
        print(pack.render())
        if len(seen) >= 3:
            break
