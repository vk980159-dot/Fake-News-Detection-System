import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.text_cleaner import clean_text
from services.verdict_engine import fuse_verdict
from services.evidence_service import (
    decompose_claim,
    classify_evidence_stance,
    verify_claim_evidence,
    evaluate_evidence_relevance,
    extract_rankings,
    extract_quantities,
    extract_dates,
    extract_modifiers
)

def run_contradiction_tests():
    print("==================================================================")
    print("RUNNING PROPOSITION-LEVEL ATTRIBUTE CONTRADICTION TEST SUITE")
    print("==================================================================")

    # -----------------------------------------------------------------
    # Test 1: Ranking Contradiction (first vs fourth / 4th)
    # Claim: "India was the first country to land a spacecraft on the Moon in 2023."
    # Evidence: "India became the 4th country to successfully land a spacecraft on the Moon."
    # -----------------------------------------------------------------
    print("\n--- Test 1: Ranking Contradiction (first vs 4th) ---")
    claim_1 = "India was the first country to land a spacecraft on the Moon in 2023."
    ev_1 = {
        'source': 'Reuters',
        'domain': 'reuters.com',
        'title': 'India becomes 4th country to land spacecraft on the Moon',
        'snippet': 'India became the 4th country to successfully land a spacecraft on the Moon.'
    }
    
    props_1 = decompose_claim(claim_1)
    assert len(props_1) > 0
    assert 1 in props_1[0].get('rankings', {}), f"Expected ranking 1, got {props_1[0].get('rankings')}"
    
    eval_1 = evaluate_evidence_relevance(claim_1, props_1, ev_1)
    print(f"Stance: {eval_1['stance']}")
    print(f"Conflict Type: {eval_1.get('conflict_type')}")
    print(f"Claim Attribute: {eval_1.get('claim_attribute')}")
    print(f"Evidence Attribute: {eval_1.get('evidence_attribute')}")
    print(f"Reason: {eval_1.get('reason')}")
    
    assert eval_1['stance'] == 'CONTRADICTING', f"Expected CONTRADICTING, got {eval_1['stance']}"
    assert eval_1.get('conflict_type') == 'Ranking contradiction'
    
    # End-to-end verdict check
    evidence_res_1 = {
        "claim": claim_1,
        "is_available": True,
        "status": "Contradicted",
        "supporting_evidence": [],
        "contradicting_evidence": [eval_1],
        "contextual_evidence": [],
        "all_evaluations": [eval_1],
        "supporting_count": 0,
        "contradicting_count": 1,
        "contextual_count": 0,
        "is_partially_supported": False
    }
    verdict_1 = fuse_verdict(
        ml_prediction=0,
        confidence=65.0,
        evidence_data=evidence_res_1,
        source_data={"reliability_tier": "Low", "overall_source_score": 50},
        sensational_data={"is_sensational": False}
    )
    print(f"Final Verdict: {verdict_1['verdict']}")
    assert verdict_1['verdict'] == 'LIKELY FAKE', f"Expected LIKELY FAKE, got {verdict_1['verdict']}"
    assert verdict_1['verdict'] != 'VERIFIED', "CRITICAL: Must NEVER be VERIFIED!"
    print(">>> Test 1 PASSED: Ranking contradiction correctly classified as CONTRADICTING and verdict is LIKELY FAKE")

    # -----------------------------------------------------------------
    # Test 2: Quantity Contradiction (10,000 vs 100)
    # -----------------------------------------------------------------
    print("\n--- Test 2: Quantity Contradiction (10,000 vs 100) ---")
    claim_2 = "10,000 citizens attended the protest in the capital city."
    ev_2 = {
        'source': 'BBC News',
        'domain': 'bbc.com',
        'title': 'Capital protest draws small crowd',
        'snippet': 'Police reported that approximately 100 people gathered for the protest in the capital city.'
    }
    props_2 = decompose_claim(claim_2)
    assert 10000 in props_2[0].get('parsed_quantities', {}), f"Expected 10000 in parsed_quantities, got {props_2[0].get('parsed_quantities')}"
    
    eval_2 = evaluate_evidence_relevance(claim_2, props_2, ev_2)
    print(f"Stance: {eval_2['stance']}")
    print(f"Conflict Type: {eval_2.get('conflict_type')}")
    print(f"Reason: {eval_2.get('reason')}")
    assert eval_2['stance'] == 'CONTRADICTING', f"Expected CONTRADICTING, got {eval_2['stance']}"
    assert eval_2.get('conflict_type') == 'Quantity contradiction'
    print(">>> Test 2 PASSED: Quantity contradiction correctly identified")

    # -----------------------------------------------------------------
    # Test 3: Temporal (Year) Contradiction (2025 vs 2024)
    # -----------------------------------------------------------------
    print("\n--- Test 3: Temporal Year Contradiction (2025 vs 2024) ---")
    claim_3 = "NASA launched the Europa Clipper mission in 2025."
    ev_3 = {
        'source': 'SpaceNews',
        'domain': 'spacenews.com',
        'title': 'NASA Europa Clipper launch confirmed',
        'snippet': 'NASA launched its Europa Clipper mission in 2024 to explore Jupiter\'s moon Europa.'
    }
    props_3 = decompose_claim(claim_3)
    assert '2025' in props_3[0].get('years', set()) or 2025 in props_3[0].get('years', set())
    
    eval_3 = evaluate_evidence_relevance(claim_3, props_3, ev_3)
    print(f"Stance: {eval_3['stance']}")
    print(f"Conflict Type: {eval_3.get('conflict_type')}")
    print(f"Reason: {eval_3.get('reason')}")
    assert eval_3['stance'] == 'CONTRADICTING', f"Expected CONTRADICTING, got {eval_3['stance']}"
    assert eval_3.get('conflict_type') == 'Temporal contradiction'
    print(">>> Test 3 PASSED: Temporal year contradiction correctly identified")

    # -----------------------------------------------------------------
    # Test 4: Temporal (Date) Contradiction (September 15 vs August 23)
    # -----------------------------------------------------------------
    print("\n--- Test 4: Temporal Date Contradiction (September 15 vs August 23) ---")
    claim_4 = "ISRO landed Chandrayaan-3 on the Moon on September 15, 2023."
    ev_4 = {
        'source': 'The Hindu',
        'domain': 'thehindu.com',
        'title': 'Chandrayaan-3 lunar landing date confirmed',
        'snippet': 'ISRO successfully landed Chandrayaan-3 on the Moon on August 23, 2023.'
    }
    props_4 = decompose_claim(claim_4)
    assert 'september 15' in props_4[0].get('dates', set()), f"Expected 'september 15', got {props_4[0].get('dates')}"
    
    eval_4 = evaluate_evidence_relevance(claim_4, props_4, ev_4)
    print(f"Stance: {eval_4['stance']}")
    print(f"Conflict Type: {eval_4.get('conflict_type')}")
    print(f"Reason: {eval_4.get('reason')}")
    assert eval_4['stance'] == 'CONTRADICTING', f"Expected CONTRADICTING, got {eval_4['stance']}"
    assert eval_4.get('conflict_type') == 'Temporal contradiction'
    print(">>> Test 4 PASSED: Temporal date contradiction correctly identified")

    # -----------------------------------------------------------------
    # Test 5: Modifier Contradiction (permanent vs temporary)
    # -----------------------------------------------------------------
    print("\n--- Test 5: Modifier Contradiction (permanent vs temporary) ---")
    claim_5 = "Scientists established a permanent base in Antarctica."
    ev_5 = {
        'source': 'Science Journal',
        'domain': 'science.org',
        'title': 'Antarctic Research Camp Set Up',
        'snippet': 'Scientists maintain a temporary research camp in Antarctica during the polar summer.'
    }
    props_5 = decompose_claim(claim_5)
    assert 'permanent' in props_5[0].get('modifiers', [])
    
    eval_5 = evaluate_evidence_relevance(claim_5, props_5, ev_5)
    print(f"Stance: {eval_5['stance']}")
    print(f"Conflict Type: {eval_5.get('conflict_type')}")
    print(f"Reason: {eval_5.get('reason')}")
    assert eval_5['stance'] == 'CONTRADICTING', f"Expected CONTRADICTING, got {eval_5['stance']}"
    assert eval_5.get('conflict_type') == 'Modifier contradiction'
    print(">>> Test 5 PASSED: Modifier contradiction correctly identified")

    # -----------------------------------------------------------------
    # Test 6A: Exclusivity Contradiction (only vs one of several)
    # -----------------------------------------------------------------
    print("\n--- Test 6A: Exclusivity Contradiction (only vs one of several) ---")
    claim_6a = "India was the only country to land a spacecraft on the Moon."
    ev_6a = {
        'source': 'Reuters',
        'domain': 'reuters.com',
        'title': 'Lunar exploration countries',
        'snippet': 'India is one of several countries to land on the Moon alongside the US, Russia, and China.'
    }
    props_6a = decompose_claim(claim_6a)
    eval_6a = evaluate_evidence_relevance(claim_6a, props_6a, ev_6a)
    print(f"Stance: {eval_6a['stance']}")
    print(f"Conflict Type: {eval_6a.get('conflict_type')}")
    print(f"Reason: {eval_6a.get('reason')}")
    assert eval_6a['stance'] == 'CONTRADICTING', f"Expected CONTRADICTING, got {eval_6a['stance']}"
    print(">>> Test 6A PASSED: Exclusivity conflict correctly identified as CONTRADICTING")

    # -----------------------------------------------------------------
    # Test 6B: Demotion to CONTEXTUAL when ranking is unconfirmed
    # Claim asserts ranking 'first country', but evidence merely says
    # 'India landed Chandrayaan-3 on the Moon in 2023' without confirming rank.
    # Must NOT be classified as SUPPORTING!
    # -----------------------------------------------------------------
    print("\n--- Test 6B: Demotion to CONTEXTUAL when ranking unconfirmed ---")
    claim_6b = "India was the first country to land a spacecraft on the Moon in 2023."
    ev_6b = {
        'source': 'Space News',
        'domain': 'spacenews.com',
        'title': 'Chandrayaan-3 lunar touchdown',
        'snippet': 'India landed its Chandrayaan-3 spacecraft on the Moon in 2023.'
    }
    props_6b = decompose_claim(claim_6b)
    eval_6b = evaluate_evidence_relevance(claim_6b, props_6b, ev_6b)
    print(f"Stance: {eval_6b['stance']}")
    print(f"Reason: {eval_6b.get('reason')}")
    assert eval_6b['stance'] == 'CONTEXTUAL', f"CRITICAL: Expected CONTEXTUAL, got {eval_6b['stance']}"
    print(">>> Test 6B PASSED: Unconfirmed ranking correctly demoted from SUPPORTING to CONTEXTUAL")

    # -----------------------------------------------------------------
    # Test 6C: First country vs First probe (Live Bug Regression)
    # Claim: "India was the first country to land a spacecraft on the Moon in 2023."
    # Evidence: "India on the moon! Chandrayaan-3 becomes 1st probe to land near lunar south pole."
    # Must NOT be classified as SUPPORTING! Must be CONTEXTUAL due to scope and sub-location mismatch.
    # -----------------------------------------------------------------
    print("\n--- Test 6C: Scope Mismatch (first country vs 1st probe) ---")
    claim_6c = "India was the first country to land a spacecraft on the Moon in 2023."
    ev_6c = {
        'source': 'Space.com',
        'domain': 'space.com',
        'title': 'India on the moon!',
        'snippet': 'India on the moon! Chandrayaan-3 becomes 1st probe to land near lunar south pole.'
    }
    props_6c = decompose_claim(claim_6c)
    eval_6c = evaluate_evidence_relevance(claim_6c, props_6c, ev_6c)
    print(f"Stance: {eval_6c['stance']}")
    print(f"Reason: {eval_6c.get('reason')}")
    assert eval_6c['stance'] == 'CONTEXTUAL', f"CRITICAL: Expected CONTEXTUAL, got {eval_6c['stance']}"
    
    # Verify end-to-end verdict cannot be VERIFIED
    ev_res_6c = {
        "claim": claim_6c,
        "is_available": True,
        "status": "Contextual",
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "contextual_evidence": [eval_6c],
        "all_evaluations": [eval_6c],
        "supporting_count": 0,
        "contradicting_count": 0,
        "contextual_count": 1,
        "is_partially_supported": False
    }
    verdict_6c = fuse_verdict(
        ml_prediction=0,
        confidence=78.13,
        evidence_data=ev_res_6c,
        source_data={"reliability_tier": "Low", "overall_source_score": 50},
        sensational_data={"is_sensational": False}
    )
    print(f"Final Verdict: {verdict_6c['verdict']}")
    assert verdict_6c['verdict'] != 'VERIFIED', "CRITICAL: Verdict must NEVER be VERIFIED for scope mismatch!"
    assert verdict_6c['verdict'] == 'UNCERTAIN', f"Expected UNCERTAIN, got {verdict_6c['verdict']}"
    print(">>> Test 6C PASSED: Scope mismatch correctly classified as CONTEXTUAL and verdict is UNCERTAIN")

    # -----------------------------------------------------------------
    # Test 6D: Global vs Location-specific achievement
    # Claim: "India achieved the first spacecraft landing on the Moon in 2023."
    # Evidence: "Chandrayaan-3 achieved the first landing near the lunar south pole."
    # Must NOT be classified as SUPPORTING! Must be CONTEXTUAL due to sub-location restriction.
    # -----------------------------------------------------------------
    print("\n--- Test 6D: Global Achievement vs Sub-location Achievement ---")
    claim_6d = "India achieved the first spacecraft landing on the Moon in 2023."
    ev_6d = {
        'source': 'Nature',
        'domain': 'nature.com',
        'title': 'Lunar South Pole Landing',
        'snippet': 'Chandrayaan-3 achieved the first landing near the lunar south pole.'
    }
    props_6d = decompose_claim(claim_6d)
    eval_6d = evaluate_evidence_relevance(claim_6d, props_6d, ev_6d)
    print(f"Stance: {eval_6d['stance']}")
    print(f"Reason: {eval_6d.get('reason')}")
    assert eval_6d['stance'] == 'CONTEXTUAL', f"CRITICAL: Expected CONTEXTUAL, got {eval_6d['stance']}"
    print(">>> Test 6D PASSED: Location-restricted first correctly classified as CONTEXTUAL for global landing claim")

    # -----------------------------------------------------------------
    # Test 6E: Country Ranking vs Mission Ranking
    # Claim: "India was the 1st country to land on the Moon."
    # Evidence: "Chandrayaan-3 was India's 3rd lunar exploration mission."
    # Must NOT be classified as SUPPORTING or CONTRADICTING! Must be CONTEXTUAL.
    # -----------------------------------------------------------------
    print("\n--- Test 6E: Country Ranking vs Mission Ranking ---")
    claim_6e = "India was the 1st country to land on the Moon."
    ev_6e = {
        'source': 'ISRO',
        'domain': 'isro.gov.in',
        'title': 'Mission Profile',
        'snippet': "Chandrayaan-3 was India's 3rd lunar exploration mission."
    }
    props_6e = decompose_claim(claim_6e)
    eval_6e = evaluate_evidence_relevance(claim_6e, props_6e, ev_6e)
    print(f"Stance: {eval_6e['stance']}")
    print(f"Reason: {eval_6e.get('reason')}")
    assert eval_6e['stance'] == 'CONTEXTUAL', f"CRITICAL: Expected CONTEXTUAL, got {eval_6e['stance']}"
    print(">>> Test 6E PASSED: Mission sequence vs country ranking correctly classified as CONTEXTUAL")

    # -----------------------------------------------------------------
    # Test 6F: Exact Matching Country-Level Ranking Evidence -> SUPPORTING
    # Claim: "India became the fourth country to successfully land a spacecraft on the Moon."
    # Evidence: "India has become the 4th country to successfully land on the Moon after US, Russia and China."
    # -----------------------------------------------------------------
    print("\n--- Test 6F: Exact Matching Country-Level Ranking ---")
    claim_6f = "India became the fourth country to successfully land a spacecraft on the Moon."
    ev_6f = {
        'source': 'BBC News',
        'domain': 'bbc.com',
        'title': 'India joins elite lunar club',
        'snippet': 'India has become the 4th country to successfully land on the Moon after US, Russia and China.'
    }
    props_6f = decompose_claim(claim_6f)
    eval_6f = evaluate_evidence_relevance(claim_6f, props_6f, ev_6f)
    print(f"Stance: {eval_6f['stance']}")
    print(f"Reason: {eval_6f.get('reason')}")
    assert eval_6f['stance'] == 'SUPPORTING', f"Expected SUPPORTING, got {eval_6f['stance']}"
    print(">>> Test 6F PASSED: Exact matching country ranking correctly classified as SUPPORTING")

    # -----------------------------------------------------------------
    # Test 7: Genuine Chandrayaan-3 Landing Claim -> SUPPORTING & VERIFIED
    # -----------------------------------------------------------------
    print("\n--- Test 7: Genuine Chandrayaan-3 Corroboration ---")
    claim_7 = "India landed Chandrayaan-3 on the Moon on August 23, 2023."
    ev_7 = {
        'source': 'ISRO Official',
        'domain': 'isro.gov.in',
        'title': 'Chandrayaan-3 mission details',
        'snippet': 'India successfully landed Chandrayaan-3 on the Moon on August 23, 2023 near lunar south pole.'
    }
    props_7 = decompose_claim(claim_7)
    eval_7 = evaluate_evidence_relevance(claim_7, props_7, ev_7)
    print(f"Stance: {eval_7['stance']}")
    print(f"Reason: {eval_7.get('reason')}")
    assert eval_7['stance'] == 'SUPPORTING', f"Expected SUPPORTING, got {eval_7['stance']}"
    
    evidence_res_7 = {
        "claim": claim_7,
        "is_available": True,
        "status": "Supported",
        "supporting_evidence": [eval_7],
        "contradicting_evidence": [],
        "contextual_evidence": [],
        "all_evaluations": [eval_7],
        "supporting_count": 1,
        "contradicting_count": 0,
        "contextual_count": 0,
        "is_partially_supported": False
    }
    verdict_7 = fuse_verdict(
        ml_prediction=1,
        confidence=88.0,
        evidence_data=evidence_res_7,
        source_data={"reliability_tier": "High", "overall_source_score": 90},
        sensational_data={"is_sensational": False}
    )
    print(f"Final Verdict: {verdict_7['verdict']}")
    assert verdict_7['verdict'] == 'VERIFIED', f"Expected VERIFIED, got {verdict_7['verdict']}"
    print(">>> Test 7 PASSED: Genuine factual claim corroborated as SUPPORTING and VERIFIED")

    print("\n==================================================================")
    print("ALL ATTRIBUTE CONTRADICTION AND SCOPE ENTAILMENT TESTS PASSED!")
    print("==================================================================")

if __name__ == '__main__':
    run_contradiction_tests()
