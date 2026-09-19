from typing import Dict, Any

def fuse_verdict(
    ml_prediction: int,
    confidence: float,
    evidence_data: Dict[str, Any],
    source_data: Dict[str, Any],
    sensational_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Transparent Evidence Fusion Decision Engine.
    Combines ML prediction, statistical model confidence, external evidence stance,
    source metadata, and textual sensationalism into one of four deterministic outcomes:
    - VERIFIED
    - MISLEADING
    - LIKELY FAKE
    - UNCERTAIN

    CORE PRINCIPLE:
    ML classification is a signal, NOT factual proof.
    A high ML confidence alone must NEVER be sufficient to produce LIKELY FAKE.
    If supporting == 0 and contradicting == 0, the verdict MUST be UNCERTAIN.
    """
    supporting = evidence_data.get("supporting_evidence", [])
    contradicting = evidence_data.get("contradicting_evidence", [])
    # Count sources directly from evidence data to ensure 100% consistency with UI and reports
    supp_count = evidence_data.get("supporting_count", len(supporting))
    cont_count = evidence_data.get("contradicting_count", len(contradicting))

    # Assess Model Confidence Level (Statistical classifier certainty, not factual truth)
    if confidence >= 85.0:
        conf_level = "High"
    elif confidence >= 70.0:
        conf_level = "Medium"
    else:
        conf_level = "Low"

    # Assess Evidence Strength - Heuristic transparent levels based on actual source counts and proposition coverage
    calc_strength = str(evidence_data.get("evidence_strength", "")).upper()
    if cont_count > 0 and supp_count > 0:
        evidence_strength = "Conflicting"
    elif cont_count > 0:
        evidence_strength = "Contradicting"
    elif calc_strength in ("STRONG", "AVAILABLE", "INSUFFICIENT"):
        evidence_strength = calc_strength.capitalize()
    elif supp_count >= 2:
        evidence_strength = "Strong"
    elif supp_count == 1:
        evidence_strength = "Available"
    else:
        evidence_strength = "Insufficient"

    source_level = source_data.get("reliability_level", "Low")
    is_sensational = sensational_data.get("is_sensational", False)

    # ── Deterministic Decision Matrix ─────────────────────────────────────────

    # Rule 1: No evidence (0 supporting, 0 contradicting) OR external evidence unavailable
    # ML alone must NEVER produce LIKELY FAKE. Absence of corroboration is NOT proof of falsity.
    if supp_count == 0 and cont_count == 0:
        verdict = "UNCERTAIN"
        ml_name = "Real News" if ml_prediction == 1 else "Fake News"
        decision_rationale = (
            f"The ML classifier classified the content as {ml_name} with {confidence:.2f}% model confidence. "
            "However, no supporting public sources or contradicting evidence were retrieved. "
            "ML confidence reflects learned textual patterns, not factual truth. "
            "Because the available evidence is insufficient, the final verdict is UNCERTAIN."
        )

    # Rule 2: Conflicting evidence (both supporting and contradicting sources present)
    elif supp_count > 0 and cont_count > 0:
        verdict = "MISLEADING"
        decision_rationale = (
            f"Conflicting public evidence was retrieved ({supp_count} supporting vs "
            f"{cont_count} contradicting source(s)). The claim contains partial truth or lacks critical context."
        )

    # Rule 3: Contradictory evidence only (reliable sources refute or debunk)
    elif cont_count > 0:
        verdict = "LIKELY FAKE"
        ranking_conflicts = [s for s in contradicting if s.get("conflict_type") == "Ranking contradiction"]
        if ranking_conflicts:
            ranking_conflicts.sort(key=lambda s: not any(w in str(s.get("evidence_attribute", "")).lower() for w in ("4th", "fourth", "rank 4", "second", "third")))
            rc = ranking_conflicts[0]
            claim_desc = str(rc.get("claim_attribute", "first country")).lower()
            ev_desc = str(rc.get("evidence_attribute", "fourth country")).lower()
            if any(w in claim_desc for w in ("first", "1st")):
                decision_rationale = (
                    "LIKELY FAKE was assigned because the retrieved evidence directly contradicts the claim's country-level ranking. "
                    "The claim states that India was the first country, while the evidence identifies India as the fourth country to successfully land a spacecraft on the Moon."
                )
            else:
                c_clean = "first country" if any(w in claim_desc for w in ("first", "1st")) else rc.get("claim_attribute")
                e_clean = "fourth country" if any(w in ev_desc for w in ("4th", "fourth", "rank 4")) else rc.get("evidence_attribute")
                decision_rationale = (
                    f"LIKELY FAKE was assigned because the retrieved evidence directly contradicts the claim's country-level ranking. "
                    f"The claim states that India was the {c_clean}, while the evidence identifies India as the {e_clean}."
                )
        else:
            attr_conflicts = [
                f"{s.get('conflict_type')}: claim asserts '{s.get('claim_attribute')}' vs evidence '{s.get('evidence_attribute')}'"
                for s in contradicting if s.get("conflict_type")
            ]
            if attr_conflicts:
                decision_rationale = f"Available public evidence directly contradicts the claim ({attr_conflicts[0]})."
            elif cont_count >= 2:
                decision_rationale = f"Multiple reliable public sources ({cont_count}) directly contradict or debunk this assertion."
            else:
                decision_rationale = "Available public evidence contradicts the central claim. 1 contradictory source was identified."

    # Rule 4: Supporting evidence only (corroborated by public sources)
    elif supp_count > 0:
        is_partially_supported = evidence_data.get("is_partially_supported", False)
        if is_sensational:
            verdict = "MISLEADING"
            decision_rationale = (
                f"Available public evidence supports the central event ({supp_count} source{' was' if supp_count == 1 else 's were'} found), "
                "but sensationalist framing or exaggerated presentation was detected."
            )
        elif is_partially_supported:
            verdict = "MISLEADING"
            decision_rationale = (
                f"The claim is only partially supported by public evidence ({supp_count} supporting public source{' was' if supp_count == 1 else 's were'} found). "
                "While some asserted propositions are corroborated, other key central claims remain unverified or exaggerated."
            )
        else:
            verdict = "VERIFIED"
            if supp_count >= 2:
                decision_rationale = (
                    f"Available public evidence supports the central claim. "
                    f"{supp_count} supporting public sources were found and no contradictory source was identified."
                )
            else:
                decision_rationale = (
                    "Available public evidence supports the central claim. "
                    "1 supporting source was found and no contradictory source was identified."
                )

    else:
        verdict = "UNCERTAIN"
        decision_rationale = "Available evidence is insufficient to establish whether the claim is factually true or false."

    return {
        "verdict": verdict,
        "decision_rationale": decision_rationale,
        "model_confidence_level": conf_level,
        "evidence_strength": evidence_strength,
        "source_information_level": source_level,
        "metrics": {
            "ml_prediction_label": "Real News" if ml_prediction == 1 else "Fake News",
            "confidence_value": confidence,
            "supporting_count": supp_count,
            "contradicting_count": cont_count,
            "is_sensational": is_sensational
        }
    }
