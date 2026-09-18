import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.text_cleaner import clean_text, analyze_sensationalism
from utils.source_analyzer import analyze_source
from services.verdict_engine import fuse_verdict
from services.ai_reasoning import generate_ai_reasoning
from services.evidence_service import (
    decompose_claim,
    evaluate_evidence_relevance,
    verify_claim_evidence,
    classify_evidence_stance
)

def run_section_11_validations():
    print("==================================================================")
    print("RUNNING SECTION 11 MANDATORY FINAL VALIDATIONS")
    print("==================================================================")

    model = pickle.load(open('fake_news_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    # -------------------------------------------------------------
    # CASE 1: Ranking Contradiction
    # Claim: "India was the first country to land a spacecraft on the Moon in 2023."
    # Evidence: "India became the 4th country to successfully land a spacecraft on the Moon."
    # -------------------------------------------------------------
    print("\n--- [CASE 1] Ranking Contradiction ---")
    claim_1 = "India was the first country to land a spacecraft on the Moon in 2023."
    ev_1 = {
        'source': 'Reuters',
        'domain': 'reuters.com',
        'title': 'India becomes 4th country to land spacecraft on the Moon',
        'snippet': 'India became the 4th country to successfully land a spacecraft on the Moon.'
    }
    props_1 = decompose_claim(claim_1)
    eval_1 = evaluate_evidence_relevance(claim_1, props_1, ev_1)
    
    print(f"Stance: {eval_1['stance']}")
    print(f"Conflict Type: {eval_1.get('conflict_type')}")
    print(f"Claim Attribute: {eval_1.get('claim_attribute')}")
    print(f"Evidence Attribute: {eval_1.get('evidence_attribute')}")
    print(f"Reason: {eval_1.get('reason')}")

    assert eval_1['stance'] == 'CONTRADICTING', f"Expected CONTRADICTING, got {eval_1['stance']}"
    assert eval_1.get('conflict_type') == 'Ranking contradiction'

    # Build evidence dictionary simulating retrieved contradiction
    ev_data_1 = {
        'is_available': True,
        'status': 'Contradicted',
        'supporting_evidence': [],
        'contradicting_evidence': [eval_1],
        'contextual_evidence': [],
        'all_evaluations': [eval_1],
        'supporting_count': 0,
        'contradicting_count': 1,
        'contextual_count': 0,
        'is_partially_supported': False
    }
    v_1 = fuse_verdict(
        ml_prediction=0,
        confidence=65.0,
        evidence_data=ev_data_1,
        source_data=analyze_source(domain="Direct Input"),
        sensational_data=analyze_sensationalism(claim_1)
    )
    reasons_1 = generate_ai_reasoning(
        ml_prediction=0,
        confidence=65.0,
        evidence_data=ev_data_1,
        source_data=analyze_source(domain="Direct Input"),
        sensational_data=analyze_sensationalism(claim_1),
        final_verdict=v_1['verdict']
    )
    print(f"Final Verdict: {v_1['verdict']}")
    print(f"Rationale: {v_1['decision_rationale']}")
    print("Reasoning lines:")
    for r in reasons_1:
        print(f"  - {r}")

    assert v_1['verdict'] == 'LIKELY FAKE', f"Expected LIKELY FAKE, got {v_1['verdict']}"
    assert v_1['verdict'] != 'VERIFIED', "CASE 1 MUST NEVER BE VERIFIED!"
    assert any("contradicts the claim that India was the first country" in r for r in reasons_1)
    print(">>> CASE 1 VALIDATED PERFECTLY: Supporting=0, Ranking conflict detected, Verdict=LIKELY FAKE, Explanation confirmed.")

    # -------------------------------------------------------------
    # CASE 2: Genuine Chandrayaan-3 Landing
    # Claim: "India's Chandrayaan-3 mission successfully landed on the Moon on August 23, 2023."
    # -------------------------------------------------------------
    print("\n--- [CASE 2] Genuine Chandrayaan-3 Landing ---")
    claim_2 = "India's Chandrayaan-3 mission successfully landed on the Moon on August 23, 2023."
    ev_2 = {
        'source': 'ISRO Official',
        'domain': 'isro.gov.in',
        'title': 'Chandrayaan-3 Landing Success',
        'snippet': 'India successfully landed Chandrayaan-3 on the Moon on August 23, 2023.'
    }
    props_2 = decompose_claim(claim_2)
    eval_2 = evaluate_evidence_relevance(claim_2, props_2, ev_2)
    print(f"Stance: {eval_2['stance']}")
    print(f"Reason: {eval_2.get('reason')}")
    assert eval_2['stance'] == 'SUPPORTING', f"Expected SUPPORTING, got {eval_2['stance']}"

    ev_data_2 = {
        'is_available': True,
        'status': 'Strong',
        'supporting_evidence': [eval_2, eval_2], # 2 reliable supporting sources
        'contradicting_evidence': [],
        'contextual_evidence': [],
        'all_evaluations': [eval_2, eval_2],
        'supporting_count': 2,
        'contradicting_count': 0,
        'contextual_count': 0,
        'is_partially_supported': False
    }
    v_2 = fuse_verdict(
        ml_prediction=1,
        confidence=92.0,
        evidence_data=ev_data_2,
        source_data=analyze_source(domain="isro.gov.in"),
        sensational_data=analyze_sensationalism(claim_2)
    )
    print(f"Final Verdict: {v_2['verdict']}")
    print(f"Rationale: {v_2['decision_rationale']}")
    assert v_2['verdict'] == 'VERIFIED', f"Expected VERIFIED, got {v_2['verdict']}"
    print(">>> CASE 2 VALIDATED PERFECTLY: Reliable supporting evidence -> VERIFIED.")

    # -------------------------------------------------------------
    # CASE 3: Moon Colony Claim with Chandrayaan Landing Evidence
    # Claim: "India has built a permanent human colony on the Moon."
    # Evidence: "Chandrayaan-3 successfully landed on the Moon."
    # Expected: CONTEXTUAL, not SUPPORTING; Never VERIFIED solely from this evidence.
    # -------------------------------------------------------------
    print("\n--- [CASE 3] Moon Colony Claim with Chandrayaan-3 Evidence ---")
    claim_3 = "India has built a permanent human colony on the Moon."
    ev_3 = {
        'source': 'Space Daily',
        'domain': 'spacedaily.com',
        'title': 'Chandrayaan-3 landing success',
        'snippet': 'Chandrayaan-3 successfully landed on the Moon.'
    }
    props_3 = decompose_claim(claim_3)
    eval_3 = evaluate_evidence_relevance(claim_3, props_3, ev_3)
    print(f"Stance: {eval_3['stance']}")
    print(f"Reason: {eval_3.get('reason')}")
    assert eval_3['stance'] == 'CONTEXTUAL', f"Expected CONTEXTUAL, got {eval_3['stance']}"
    assert eval_3['stance'] != 'SUPPORTING', "Must NOT be SUPPORTING!"

    # Verdict with only contextual evidence
    ev_data_3 = {
        'is_available': False,
        'status': 'Insufficient',
        'supporting_evidence': [],
        'contradicting_evidence': [],
        'contextual_evidence': [eval_3],
        'all_evaluations': [eval_3],
        'supporting_count': 0,
        'contradicting_count': 0,
        'contextual_count': 1,
        'is_partially_supported': False
    }
    v_3 = fuse_verdict(
        ml_prediction=0,
        confidence=85.0,
        evidence_data=ev_data_3,
        source_data=analyze_source(domain="Direct Input"),
        sensational_data=analyze_sensationalism(claim_3)
    )
    print(f"Final Verdict: {v_3['verdict']}")
    print(f"Rationale: {v_3['decision_rationale']}")
    assert v_3['verdict'] == 'UNCERTAIN', f"Expected UNCERTAIN, got {v_3['verdict']}"
    assert v_3['verdict'] != 'VERIFIED', "Moon colony MUST NEVER be VERIFIED!"
    print(">>> CASE 3 VALIDATED PERFECTLY: CONTEXTUAL, not SUPPORTING; Verdict is UNCERTAIN (never VERIFIED).")

    # -------------------------------------------------------------
    # CASE 4: Mars City Claim with Gateway Corrosion Evidence
    # Claim: "NASA confirmed India's permanent city on Mars by 2027."
    # Evidence: "NASA Lunar Gateway modules were corroded."
    # Expected: CONTEXTUAL or IRRELEVANT; Never SUPPORTING; UNCERTAIN unless contradiction found.
    # -------------------------------------------------------------
    print("\n--- [CASE 4] Mars City Claim with Lunar Gateway Corrosion Evidence ---")
    claim_4 = "NASA confirmed India's permanent city on Mars by 2027."
    ev_4 = {
        'source': 'SpaceNews',
        'domain': 'spacenews.com',
        'title': 'Lunar Gateway modules check',
        'snippet': 'NASA Lunar Gateway modules were corroded during testing.'
    }
    props_4 = decompose_claim(claim_4)
    eval_4 = evaluate_evidence_relevance(claim_4, props_4, ev_4)
    print(f"Stance: {eval_4['stance']}")
    print(f"Reason: {eval_4.get('reason')}")
    assert eval_4['stance'] in ('CONTEXTUAL', 'IRRELEVANT'), f"Expected CONTEXTUAL or IRRELEVANT, got {eval_4['stance']}"
    assert eval_4['stance'] != 'SUPPORTING', "Must NEVER be SUPPORTING!"

    ev_data_4 = {
        'is_available': False,
        'status': 'Insufficient',
        'supporting_evidence': [],
        'contradicting_evidence': [],
        'contextual_evidence': [eval_4],
        'all_evaluations': [eval_4],
        'supporting_count': 0,
        'contradicting_count': 0,
        'contextual_count': 1,
        'is_partially_supported': False
    }
    v_4 = fuse_verdict(
        ml_prediction=0,
        confidence=90.0,
        evidence_data=ev_data_4,
        source_data=analyze_source(domain="Direct Input"),
        sensational_data=analyze_sensationalism(claim_4)
    )
    print(f"Final Verdict: {v_4['verdict']}")
    print(f"Rationale: {v_4['decision_rationale']}")
    assert v_4['verdict'] == 'UNCERTAIN', f"Expected UNCERTAIN, got {v_4['verdict']}"
    assert v_4['verdict'] != 'VERIFIED', "Mars city claim MUST NEVER be VERIFIED!"
    print(">>> CASE 4 VALIDATED PERFECTLY: CONTEXTUAL/IRRELEVANT, not SUPPORTING; Verdict is UNCERTAIN.")

    print("\n==================================================================")
    print(" ALL 4 SECTION 11 MANDATORY CASES VALIDATED 100% SUCCESSFULLY! ")
    print("==================================================================")

if __name__ == '__main__':
    run_section_11_validations()
