import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.text_cleaner import clean_text, analyze_sensationalism
from utils.source_analyzer import analyze_source
from services.verdict_engine import fuse_verdict
from services.evidence_service import (
    decompose_claim,
    classify_evidence_stance,
    verify_claim_evidence,
    evaluate_evidence_relevance,
    is_authoritative_source,
    is_reference_source
)

def run_regression_suite():
    print("==================================================")
    print("RUNNING PROPOSITION ENTAILMENT REGRESSION SUITE")
    print("==================================================")

    model = pickle.load(open('fake_news_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    # -------------------------------------------------------------
    # Test 1: Wikipedia Source Labeling
    # -------------------------------------------------------------
    print("\n--- Test 1: Wikipedia Source Labeling ---")
    assert is_reference_source('wikipedia.org', 'Wikipedia') == True
    assert is_authoritative_source('wikipedia.org', 'Wikipedia') == False
    assert is_authoritative_source('isro.gov.in', 'ISRO') == True
    assert is_authoritative_source('reuters.com', 'Reuters') == True
    print(">>> Test 1 PASSED: Wikipedia is Reference Source, not Authoritative Source")

    # -------------------------------------------------------------
    # Test A: Chandrayaan-3 claim + landing source -> SUPPORTING
    # -------------------------------------------------------------
    print("\n--- Test A: Chandrayaan-3 claim + landing source ---")
    claim_chandrayaan = "India landed Chandrayaan-3 on the Moon."
    source_landing = {
        'source': 'Space Daily',
        'domain': 'spacedaily.com',
        'title': 'India landed its Chandrayaan-3 spacecraft near Moon south pole in August 2023',
        'snippet': 'India landed its Chandrayaan-3 spacecraft near Moon south pole'
    }
    stance_a = classify_evidence_stance(claim_chandrayaan, source_landing)
    print(f"Stance: {stance_a}")
    assert stance_a == 'SUPPORTING', f"Expected SUPPORTING, got {stance_a}"
    print(">>> Test A PASSED: Landing report correctly entails Chandrayaan-3 landing claim")

    # -------------------------------------------------------------
    # Test B: Moon colony claim + Chandrayaan-3 landing source -> CONTEXTUAL
    # -------------------------------------------------------------
    print("\n--- Test B: Moon colony claim + Chandrayaan-3 landing source ---")
    claim_moon_colony = "India has successfully built a permanent human colony on the Moon, and the first Indian astronauts are already living there permanently. ISRO announced that the colony became operational in 2025."
    stance_b = classify_evidence_stance(claim_moon_colony, source_landing)
    print(f"Stance: {stance_b}")
    print(f"Entailment reason: {source_landing.get('reason')}")
    assert stance_b == 'CONTEXTUAL', f"CRITICAL REGRESSION! Expected CONTEXTUAL, got {stance_b}"
    print(">>> Test B PASSED: Chandrayaan-3 landing source is strictly CONTEXTUAL for Moon colony claim")

    # -------------------------------------------------------------
    # Test C: Moon colony claim + refutation source -> CONTRADICTING
    # -------------------------------------------------------------
    print("\n--- Test C: Moon colony claim + refutation source ---")
    source_refutation = {
        'source': 'Reuters Fact Check',
        'domain': 'reuters.com',
        'title': 'Fact Check: Claims that India established a permanent moon colony are false and fabricated',
        'snippet': 'There is no human colony on the moon and ISRO has not built any permanent lunar settlement.'
    }
    stance_c = classify_evidence_stance(claim_moon_colony, source_refutation)
    print(f"Stance: {stance_c}")
    print(f"Entailment reason: {source_refutation.get('reason')}")
    assert stance_c == 'CONTRADICTING', f"Expected CONTRADICTING, got {stance_c}"
    print(">>> Test C PASSED: Fact check report correctly classified as CONTRADICTING")

    # -------------------------------------------------------------
    # Test D: Compound claim (one true, one false/unsupported) -> MISLEADING
    # -------------------------------------------------------------
    print("\n--- Test D: Compound claim -> MISLEADING ---")
    compound_claim = "India landed Chandrayaan-3 on the moon and immediately established a permanent human colony."
    compound_props = decompose_claim(compound_claim)
    print("Compound Claim Propositions:")
    for idx, p in enumerate(compound_props):
        print(f"  P{idx+1}: {p['text']} (Actions: {p['actions']}, Objects: {p['objects']})")

    # Landing source only supports P1, not P2
    eval_landing = evaluate_evidence_relevance(compound_claim, compound_props, source_landing)
    print("Evaluation of landing source against compound claim:")
    print(f"  Stance: {eval_landing['stance']}")
    print(f"  Matched: {eval_landing['matched_proposition']}")
    print(f"  Reason: {eval_landing['reason']}")
    assert eval_landing['stance'] == 'SUPPORTING'

    # Build evidence dictionary simulating P1 supported, P2 unsupported
    cleaned_comp = clean_text(compound_claim)
    vec_comp = vectorizer.transform([cleaned_comp])
    ml_comp = int(model.predict(vec_comp)[0])
    conf_comp = float(max(model.predict_proba(vec_comp)[0]) * 100.0)
    sens_comp = analyze_sensationalism(compound_claim)
    src_comp = analyze_source(domain="Direct Input")

    # 1 supporting item for P1, 0 for P2 -> is_partially_supported is True
    ev_compound = {
        'is_available': True,
        'status': 'Available',
        'supporting_evidence': [{
            'source': 'Space Daily',
            'domain': 'spacedaily.com',
            'title': source_landing['title'],
            'url': 'https://spacedaily.com/landing',
            'snippet': source_landing['snippet'],
            'is_authoritative': True,
            'stance': 'SUPPORTING',
            'matched_proposition': eval_landing['matched_proposition']
        }],
        'contradicting_evidence': [],
        'contextual_evidence': [],
        'is_partially_supported': True
    }
    res_d = fuse_verdict(ml_comp, conf_comp, ev_compound, src_comp, sens_comp)
    print(f"Compound Claim Final Verdict: {res_d['verdict']}")
    print(f"Compound Claim Rationale: {res_d['decision_rationale']}")
    assert res_d['verdict'] == 'MISLEADING', f"Expected MISLEADING, got {res_d['verdict']}"
    assert 'partially supported' in res_d['decision_rationale']
    print(">>> Test D PASSED: Partially supported compound claim yields MISLEADING")

    # -------------------------------------------------------------
    # Test E: Full Moon Colony Claim End-to-End Simulation
    # -------------------------------------------------------------
    print("\n--- Test E: Moon Colony Claim End-to-End Simulation ---")
    chandrayaan_articles = [
        {'source': 'Space Daily', 'domain': 'spacedaily.com', 'title': 'India landed its Chandrayaan-3 spacecraft near Moon south pole', 'snippet': 'Chandrayaan-3 spacecraft landing near lunar south pole'},
        {'source': 'Nature', 'domain': 'nature.com', 'title': 'Chandrayaan-3 lands on the Moon south pole', 'snippet': 'historic landing near lunar south polar region'},
        {'source': 'BBC', 'domain': 'bbc.com', 'title': 'Chandrayaan-3: Pragyan rover deployed on Moon', 'snippet': 'Pragyan rover rolled out from lander'},
        {'source': 'Wikipedia', 'domain': 'wikipedia.org', 'title': 'Wikipedia: Chandrayaan-3', 'snippet': 'Vikram lander and Pragyan rover successfully landed in August 2023'}
    ]

    moon_props = decompose_claim(claim_moon_colony)
    supp_moon = []
    cont_moon = []
    ctx_moon = []

    for art in chandrayaan_articles:
        st = classify_evidence_stance(claim_moon_colony, art)
        if st == 'SUPPORTING':
            supp_moon.append(art)
        elif st == 'CONTRADICTING':
            cont_moon.append(art)
        else:
            ctx_moon.append(art)

    print(f"Moon colony claim with Chandrayaan-3 articles:")
    print(f"  Supporting: {len(supp_moon)}")
    print(f"  Contradicting: {len(cont_moon)}")
    print(f"  Contextual: {len(ctx_moon)}")
    assert len(supp_moon) == 0, f"REGRESSION: Moon colony claim got {len(supp_moon)} supporting sources from Chandrayaan-3 articles!"
    assert len(ctx_moon) == len(chandrayaan_articles)

    ev_moon = {
        'is_available': True,
        'status': 'Insufficient',
        'supporting_evidence': supp_moon,
        'contradicting_evidence': cont_moon,
        'contextual_evidence': ctx_moon,
        'is_partially_supported': False
    }

    cleaned_moon = clean_text(claim_moon_colony)
    vec_moon = vectorizer.transform([cleaned_moon])
    ml_moon = int(model.predict(vec_moon)[0])
    conf_moon = float(max(model.predict_proba(vec_moon)[0]) * 100.0)
    sens_moon = analyze_sensationalism(claim_moon_colony)
    src_moon = analyze_source(domain="Direct Input")

    res_moon = fuse_verdict(ml_moon, conf_moon, ev_moon, src_moon, sens_moon)
    print(f"Moon Colony Claim Verdict: {res_moon['verdict']}")
    print(f"Moon Colony Claim Rationale: {res_moon['decision_rationale']}")
    assert res_moon['verdict'] == 'UNCERTAIN', f"Expected UNCERTAIN, got {res_moon['verdict']}"
    assert res_moon['verdict'] != 'VERIFIED', "Moon colony claim must NEVER be VERIFIED!"
    print(">>> Test E PASSED: Moon colony claim yields UNCERTAIN with 0 supporting sources (NEVER VERIFIED)")

    # -------------------------------------------------------------
    # Test F: Chandrayaan-3 Genuine Claim End-to-End Simulation
    # -------------------------------------------------------------
    print("\n--- Test F: Chandrayaan-3 Genuine Claim End-to-End Simulation ---")
    genuine_claim = "India successfully landed Chandrayaan-3 near the lunar south pole on August 23, 2023."
    genuine_props = decompose_claim(genuine_claim)
    supp_gen = []
    cont_gen = []
    ctx_gen = []

    for art in chandrayaan_articles:
        st = classify_evidence_stance(genuine_claim, art)
        if st == 'SUPPORTING':
            supp_gen.append(art)
        elif st == 'CONTRADICTING':
            cont_gen.append(art)
        else:
            ctx_gen.append(art)

    print(f"Genuine Chandrayaan-3 claim with articles:")
    print(f"  Supporting: {len(supp_gen)}")
    print(f"  Contradicting: {len(cont_gen)}")
    print(f"  Contextual: {len(ctx_gen)}")
    assert len(supp_gen) >= 2, f"Expected >= 2 supporting, got {len(supp_gen)}"

    ev_gen = {
        'is_available': True,
        'status': 'Strong',
        'supporting_evidence': supp_gen,
        'contradicting_evidence': cont_gen,
        'contextual_evidence': ctx_gen,
        'is_partially_supported': False
    }

    cleaned_gen = clean_text(genuine_claim)
    vec_gen = vectorizer.transform([cleaned_gen])
    ml_gen = int(model.predict(vec_gen)[0])
    conf_gen = float(max(model.predict_proba(vec_gen)[0]) * 100.0)
    sens_gen = analyze_sensationalism(genuine_claim)
    src_gen = analyze_source(url='https://www.isro.gov.in/Chandrayaan3.html', domain='isro.gov.in')

    res_gen = fuse_verdict(ml_gen, conf_gen, ev_gen, src_gen, sens_gen)
    print(f"Genuine Chandrayaan-3 Verdict: {res_gen['verdict']}")
    print(f"Genuine Chandrayaan-3 Rationale: {res_gen['decision_rationale']}")
    assert res_gen['verdict'] == 'VERIFIED', f"Expected VERIFIED, got {res_gen['verdict']}"
    assert 'Available public evidence supports the central claim' in res_gen['decision_rationale']
    print(">>> Test F PASSED: Genuine Chandrayaan-3 claim yields VERIFIED")

    print("\n==================================================")
    print(" ALL PROPOSITION REGRESSION TESTS PASSED! ")
    print("==================================================")

if __name__ == '__main__':
    run_regression_suite()
