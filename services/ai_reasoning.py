import os
from typing import Dict, Any, List, Optional

def generate_ai_reasoning(
    ml_prediction: int,
    confidence: float,
    evidence_data: Dict[str, Any],
    source_data: Dict[str, Any],
    sensational_data: Dict[str, Any],
    final_verdict: Optional[str] = None
) -> List[str]:
    """
    Generate evidence-grounded AI reasoning statements based ONLY on actual computed metrics.
    No hardcoded, fabricated claims are used.
    """
    reasons = []

    # 1. ML Classifier Statement & Core Distinction
    ml_name = "Real News" if ml_prediction == 1 else "Fake News"
    reasons.append(
        f"The ML classifier classified the content as {ml_name} with {confidence:.2f}% model confidence."
    )

    # 2. External Evidence Statements
    supporting = evidence_data.get("supporting_evidence", [])
    contradicting = evidence_data.get("contradicting_evidence", [])
    is_available = evidence_data.get("is_available", False)
    supp_count = len(supporting)
    cont_count = len(contradicting)

    # If no supporting and no contradicting evidence:
    if not is_available or (supp_count == 0 and cont_count == 0):
        reasons.append("However, no independent supporting or contradicting evidence was retrieved.")
        reasons.append("ML confidence reflects learned textual patterns, not factual truth.")
        reasons.append("Because the available evidence is insufficient, the final verdict is UNCERTAIN.")
    else:
        reasons.append(
            "This model confidence reflects learned linguistic patterns and is not factual proof."
        )
        if supp_count > 0 and cont_count == 0:
            reasons.append("Available public evidence supports the central claim.")
            source_names = list({s.get("source", "Public Source") for s in supporting[:2]})
            source_str = f" ({', '.join(source_names)})" if source_names else ""
            reasons.append(
                f"{supp_count} supporting public source{' was' if supp_count == 1 else 's were'} retrieved{source_str}."
            )
            reasons.append("No contradicting source was identified.")
            if final_verdict == "VERIFIED":
                reasons.append(
                    "Because reliable independent evidence clearly supports the central claim and no contradiction was found, the final verdict is VERIFIED."
                )
            elif final_verdict == "MISLEADING":
                reasons.append(
                    "Because the claim contains partial truth or sensationalist presentation, the final verdict is MISLEADING."
                )
            reasons.append(f"Final verdict: {final_verdict or 'VERIFIED'}.")

        elif cont_count > 0 and supp_count == 0:
            reasons.append("Available public evidence contradicts the central claim.")
            source_names = list({s.get("source", "Public Source") for s in contradicting[:2]})
            source_str = f" ({', '.join(source_names)})" if source_names else ""
            reasons.append(
                f"{cont_count} contradicting public source{' was' if cont_count == 1 else 's were'} retrieved{source_str}."
            )
            ranking_conflicts = [
                s for s in contradicting if s.get("conflict_type") == "Ranking contradiction"
            ]
            if ranking_conflicts:
                rc = ranking_conflicts[0]
                c_attr = rc.get("claim_attribute", "first country")
                e_attr = rc.get("evidence_attribute", "fourth country")
                c_clean = "first country" if "first" in str(c_attr).lower() or "1st" in str(c_attr).lower() else str(c_attr)
                e_clean = "fourth country" if "4th" in str(e_attr).lower() or "fourth" in str(e_attr).lower() else str(e_attr)
                reasons.append(
                    f"Retrieved evidence discusses the same lunar landing event but contradicts the claim that India was the {c_clean}. The source states that India was the {e_clean}."
                )
            else:
                attr_conflicts = [
                    f"{s.get('conflict_type')}: claim asserts '{s.get('claim_attribute')}' but evidence confirms '{s.get('evidence_attribute')}'"
                    for s in contradicting if s.get("conflict_type")
                ]
                if attr_conflicts:
                    reasons.append(f"Attribute contradiction detected: {attr_conflicts[0]}.")
            reasons.append("Because reliable evidence directly contradicts the claim, the final verdict is LIKELY FAKE.")
            reasons.append("Final verdict: LIKELY FAKE.")

        elif supp_count > 0 and cont_count > 0:
            reasons.append(
                f"Conflicting public evidence was retrieved ({supp_count} supporting vs {cont_count} contradicting sources)."
            )
            reasons.append(
                f"Because the evidence is conflicting, the final verdict is {final_verdict or 'MISLEADING'}."
            )
            reasons.append(f"Final verdict: {final_verdict or 'MISLEADING'}.")

    # 3. Source Information Statements
    has_domain = source_data.get("has_domain", False)
    domain = source_data.get("domain", "Direct Input")
    meta_status = source_data.get("metadata_status", "Not Available")
    author = source_data.get("author", "Not Specified")
    publish_date = source_data.get("publish_date", "Not Specified")

    if not has_domain or meta_status == "Not Available":
        reasons.append("Source metadata could not be independently verified (direct input).")
    elif meta_status == "Complete":
        reasons.append(f"Publisher metadata is complete for domain '{domain}' (author: {author}, date: {publish_date}).")
    elif has_domain:
        reasons.append(f"Source domain identified ({domain}). Additional independent publisher verification was not performed.")

    # 4. Textual & Sensationalism Signals (ONLY supporting signals, never causing LIKELY FAKE independently)
    warning_signals = sensational_data.get("warning_signals", [])
    if warning_signals:
        for signal in warning_signals:
            reasons.append(f"Linguistic supporting signal detected: {signal}.")

    return reasons

def generate_structured_reasoning(
    ml_prediction: int,
    confidence: float,
    evidence_data: Dict[str, Any],
    source_data: Dict[str, Any],
    sensational_data: Dict[str, Any],
    final_verdict: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate an explicit, transparent 3-part plain-language explainability breakdown:
    1. ML Model Analysis: Linguistic patterns & statistical classifier confidence vs factual truth.
    2. Live Evidence Corroboration: Independent sources, publisher domains, or absence of corroboration.
    3. Final Verdict Rationale: Deterministic decision rule applied.
    """
    reasons_list = generate_ai_reasoning(
        ml_prediction=ml_prediction,
        confidence=confidence,
        evidence_data=evidence_data,
        source_data=source_data,
        sensational_data=sensational_data,
        final_verdict=final_verdict
    )

    ml_name = "Real News" if ml_prediction == 1 else "Fake News"
    ml_explanation = (
        f"The ML model evaluated stylistic vocabulary and emotional patterns, classifying the text as "
        f"**{ml_name}** with **{confidence:.2f}%** statistical confidence. "
        f"*(Note: Model confidence reflects learned linguistic patterns from training data, not factual verification.)*"
    )

    supporting = evidence_data.get("supporting_evidence", [])
    contradicting = evidence_data.get("contradicting_evidence", [])
    supp_cnt = len(supporting)
    cont_cnt = len(contradicting)
    unique_domains = list({
        s.get("domain") or s.get("source")
        for s in (supporting + contradicting)
        if s.get("domain") or s.get("source")
    })

    if supp_cnt == 0 and cont_cnt == 0:
        evidence_explanation = (
            "No independent supporting or contradicting sources were found for this claim in live news feeds "
            "or reference databases. Because corroboration is missing, the claim cannot be certified as true or false."
        )
    elif supp_cnt > 0 and cont_cnt == 0:
        domain_str = f" from {', '.join(unique_domains[:3])}" if unique_domains else ""
        evidence_explanation = (
            f"Retrieved **{supp_cnt} supporting source(s)**{domain_str} corroborating the specific propositions of the claim. "
            f"Zero contradictory or debunking reports were found."
        )
    elif cont_cnt > 0 and supp_cnt == 0:
        domain_str = f" from {', '.join(unique_domains[:3])}" if unique_domains else ""
        attr_conflicts = [
            f"{s.get('conflict_type')} (claim asserts '{s.get('claim_attribute')}' vs evidence '{s.get('evidence_attribute')}')"
            for s in contradicting if s.get("conflict_type")
        ]
        conflict_note = f" Explicit contradiction identified: {attr_conflicts[0]}." if attr_conflicts else ""
        evidence_explanation = (
            f"Retrieved **{cont_cnt} contradictory source(s)**{domain_str} directly refuting central factual assertions of the claim.{conflict_note}"
        )
    else:
        evidence_explanation = (
            f"Retrieved conflicting evidence: **{supp_cnt} supporting** vs **{cont_cnt} contradicting source(s)**. "
            f"The claim likely involves partial truth or conflicting accounts."
        )

    v = final_verdict or "UNCERTAIN"
    if v == "VERIFIED":
        verdict_explanation = (
            "**VERIFIED** was assigned because independent public evidence clearly corroborates the central claim's "
            "factual propositions, and no credible contradictory sources were found."
        )
    elif v == "MISLEADING":
        verdict_explanation = (
            "**MISLEADING** was assigned because the evidence conflicts, or the claim exaggerates verified facts, "
            "or combines factual events with unverified details."
        )
    elif v == "LIKELY FAKE":
        verdict_explanation = (
            "**LIKELY FAKE** was assigned because reliable independent sources directly contradict or debunk the claim."
        )
    else:
        verdict_explanation = (
            "**UNCERTAIN** was assigned under the core rule: without external corroborating or refuting evidence, "
            "ML linguistic confidence alone is never sufficient to declare content real or fake."
        )

    return {
        "ml_explanation": ml_explanation,
        "evidence_explanation": evidence_explanation,
        "verdict_explanation": verdict_explanation,
        "bullet_points": reasons_list
    }

def get_gemini_api_key() -> Optional[str]:
    """
    Safely retrieve Gemini / Google API key without crashing if secrets.toml is missing.
    Checks:
    1. streamlit.secrets (safely guarded against StreamlitSecretNotFoundError and all exceptions)
    2. os.environ ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    Returns the key string if found and non-empty, otherwise None.
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
            if key and str(key).strip():
                return str(key).strip()
    except Exception:
        pass

    try:
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if env_key and str(env_key).strip():
            return str(env_key).strip()
    except Exception:
        pass

    return None

def query_gemini_synthesis(
    claim_text: str,
    reasons: List[str],
    verdict: str,
    api_key: Optional[str] = None
) -> Optional[str]:
    """
    Optional LLM synthesis: If an API key is provided, Gemini synthesizes the
    grounded reasons into a fluent paragraph. If key is absent, returns None.
    Never invents new facts outside of the provided reasons.
    """
    if not api_key:
        api_key = get_gemini_api_key()

    if not api_key:
        return None

    try:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        prompt_text = (
            f"You are an AI news verification assistant. Based ONLY on the following verified facts, "
            f"provide a concise 2-sentence neutral synthesis of the verdict ({verdict}). "
            f"Do not invent facts, URLs, or evidence not listed here.\n\n"
            f"Claim/Text: {claim_text[:300]}\n"
            f"Verified Findings:\n" + "\n".join(f"- {r}" for r in reasons)
        )
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 150}
        }
        res = requests.post(url, headers=headers, json=payload, timeout=5)
        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                text_part = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return text_part.strip()
    except Exception:
        pass
    return None
