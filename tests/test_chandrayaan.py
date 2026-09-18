import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.text_cleaner import clean_text, analyze_sensationalism
from utils.source_analyzer import analyze_source
from services.verdict_engine import fuse_verdict
from services.ai_reasoning import generate_ai_reasoning
from services.evidence_service import verify_claim_evidence, classify_evidence_stance

def test_chandrayaan_suite():
    print("==================================================")
    print("RUNNING CHANDRAYAAN-3 DEDICATED TEST SUITE")
    print("==================================================")

    model = pickle.load(open('fake_news_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    chandrayaan_text = (
        "Chandrayaan-3 has successfully soft-landed on the moon. The Ch-3 Rover ramped down from the Lander "
        "and India took a walk on the moon! August 23, 2023. Vikram lander module descent near lunar south polar region. "
        "Pragyan rover was successfully deployed."
    )
    chandrayaan_title = "Chandrayaan-3"

    cleaned = clean_text(chandrayaan_text)
    vec = vectorizer.transform([cleaned])
    ml_pred = int(model.predict(vec)[0])
    conf = float(max(model.predict_proba(vec)[0]) * 100.0)
    source_analysis = analyze_source(url='https://www.isro.gov.in/Chandrayaan3.html', domain='isro.gov.in')
    sens = analyze_sensationalism(chandrayaan_text)

    # -------------------------------------------------------------
    # Test A: Strong supporting evidence -> VERIFIED
    # -------------------------------------------------------------
    print("\n--- Test A: Strong Supporting Evidence ---")
    ev_a = {
        'is_available': True,
        'status': 'Strong',
        'supporting_evidence': [
            {
                'source': 'Space Daily',
                'domain': 'spacedaily.com',
                'title': 'India landed its Chandrayaan-3 spacecraft near the Moon south pole in August 2023',
                'url': 'https://spacedaily.com/chandrayaan3',
                'snippet': 'India landed its Chandrayaan-3 spacecraft near the Moon south pole in August 2023',
                'is_authoritative': True,
                'stance': 'SUPPORTING'
            },
            {
                'source': 'Nature',
                'domain': 'nature.com',
                'title': 'Chandrayaan-3 lands on the Moon south pole',
                'url': 'https://nature.com/articles/chandrayaan3',
                'snippet': 'historic landing near lunar south polar region',
                'is_authoritative': True,
                'stance': 'SUPPORTING'
            },
            {
                'source': 'Wikipedia',
                'domain': 'wikipedia.org',
                'title': 'Wikipedia: Chandrayaan-3',
                'url': 'https://en.wikipedia.org/wiki/Chandrayaan-3',
                'snippet': 'Vikram, a lunar lander, and Pragyan, a lunar rover successfully landed on 23 August 2023',
                'is_authoritative': True,
                'stance': 'SUPPORTING'
            }
        ],
        'contradicting_evidence': [],
        'contextual_evidence': []
    }
    res_a = fuse_verdict(ml_pred, conf, ev_a, source_analysis, sens)
    print("Test A Verdict:", res_a['verdict'])
    print("Test A Rationale:", res_a['decision_rationale'])
    assert res_a['verdict'] == 'VERIFIED', f"Test A Expected VERIFIED, got {res_a['verdict']}"
    assert 'Available public evidence supports the central claim' in res_a['decision_rationale']
    print(">>> Test A PASSED: Strong supporting evidence yields VERIFIED")

    # -------------------------------------------------------------
    # Test B: No evidence (0/0) -> UNCERTAIN
    # -------------------------------------------------------------
    print("\n--- Test B: No Evidence (0/0) ---")
    ev_b = {
        'is_available': False,
        'status': 'Insufficient',
        'supporting_evidence': [],
        'contradicting_evidence': [],
        'contextual_evidence': []
    }
    res_b = fuse_verdict(ml_pred, conf, ev_b, source_analysis, sens)
    print("Test B Verdict:", res_b['verdict'])
    print("Test B Rationale:", res_b['decision_rationale'])
    assert res_b['verdict'] == 'UNCERTAIN', f"Test B Expected UNCERTAIN, got {res_b['verdict']}"
    assert 'Because the available evidence is insufficient, the final verdict is UNCERTAIN' in res_b['decision_rationale']
    print(">>> Test B PASSED: 0/0 evidence strictly yields UNCERTAIN")

    # -------------------------------------------------------------
    # Test C: Strong contradiction -> LIKELY FAKE
    # -------------------------------------------------------------
    print("\n--- Test C: Strong Contradiction ---")
    ev_c = {
        'is_available': True,
        'status': 'Contradicting',
        'supporting_evidence': [],
        'contradicting_evidence': [
            {
                'source': 'Reuters Fact Check',
                'domain': 'reuters.com',
                'title': 'Fact Check: Disproven and fabricated space claim',
                'url': 'https://reuters.com/factcheck/space',
                'snippet': 'Independent analysis debunked and refuted the assertion as completely false.',
                'is_authoritative': True,
                'stance': 'CONTRADICTING'
            },
            {
                'source': 'AP Fact Check',
                'domain': 'apnews.com',
                'title': 'Hoax report refuted by official radar telemetry',
                'url': 'https://apnews.com/article/factcheck',
                'snippet': 'Officials confirmed the report was entirely fabricated.',
                'is_authoritative': True,
                'stance': 'CONTRADICTING'
            }
        ],
        'contextual_evidence': []
    }
    res_c = fuse_verdict(ml_pred, conf, ev_c, source_analysis, sens)
    print("Test C Verdict:", res_c['verdict'])
    print("Test C Rationale:", res_c['decision_rationale'])
    assert res_c['verdict'] == 'LIKELY FAKE', f"Test C Expected LIKELY FAKE, got {res_c['verdict']}"
    assert 'Available public evidence' in res_c['decision_rationale'] or 'contradict' in res_c['decision_rationale']
    print(">>> Test C PASSED: Strong contradiction yields LIKELY FAKE")

    # -------------------------------------------------------------
    # Test D: Stance classification for specific Chandrayaan propositions
    # -------------------------------------------------------------
    print("\n--- Test D: Specific Factual Propositions Classification ---")
    propositions = [
        ('India landed Chandrayaan-3', {'source': 'Space Daily', 'domain': 'spacedaily.com', 'title': 'India landed its Chandrayaan-3 spacecraft near Moon south pole in August 2023', 'snippet': 'India landed its Chandrayaan-3 spacecraft near Moon south pole'}),
        ('Landing August 23 2023', {'source': 'Wikipedia', 'domain': 'wikipedia.org', 'title': 'Wikipedia: Chandrayaan programme', 'snippet': 'Chandrayaan-3 successfully landed on the Moon on 23 August 2023'}),
        ('Vikram lander touchdown', {'source': 'NDTV', 'domain': 'ndtv.com', 'title': 'Chandrayaan-3 Vikram lander touchdown successful on lunar surface', 'snippet': 'Vikram lander touched down safely'}),
        ('Pragyan rover deployed', {'source': 'BBC', 'domain': 'bbc.com', 'title': 'Chandrayaan-3: Pragyan rover deployed on Moon', 'snippet': 'Pragyan rover rolled out from lander'}),
        ('Near lunar south polar region', {'source': 'Nature', 'domain': 'nature.com', 'title': 'Chandrayaan-3 lands on the Moon south pole', 'snippet': 'historic landing near lunar south polar region'})
    ]

    for label, item in propositions:
        stance = classify_evidence_stance(chandrayaan_text, item, chandrayaan_title)
        assert stance == 'SUPPORTING', f"Proposition {label} failed: expected SUPPORTING, got {stance}"
        print(f"  [OK] {label} classified as SUPPORTING")

    print("\n==================================================")
    print(" ALL CHANDRAYAAN-3 TESTS PASSED PERFECTLY! ")
    print("==================================================")

if __name__ == '__main__':
    test_chandrayaan_suite()
