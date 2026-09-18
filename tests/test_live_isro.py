import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.content_extractor import extract_from_url
from utils.source_analyzer import analyze_source
from services.evidence_service import verify_claim_evidence
from services.verdict_engine import fuse_verdict
from services.ai_reasoning import generate_ai_reasoning
from utils.text_cleaner import clean_text, analyze_sensationalism

def run_live_isro_test():
    print("==================================================")
    print("RUNNING LIVE ISRO CHANDRAYAAN-3 PIPELINE TEST")
    print("==================================================")

    url = "https://www.isro.gov.in/Chandrayaan3.html"
    ext_res = extract_from_url(url)
    assert ext_res["success"], f"Extraction failed: {ext_res.get('error')}"

    print(f"Domain: {ext_res['domain']}")
    print(f"Title: {ext_res['title']}")
    print(f"Author: {ext_res['author']}")
    print(f"Date: {ext_res['publish_date']}")
    print(f"Word count: {len(ext_res['text'].split())}")

    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    raw_content = ext_res["text"]
    cleaned = clean_text(raw_content)
    news_vec = vectorizer.transform([cleaned])
    ml_pred = int(model.predict(news_vec)[0])
    prob_dist = model.predict_proba(news_vec)[0]
    confidence = float(max(prob_dist) * 100.0)

    sensational_signals = analyze_sensationalism(raw_content)

    source_analysis = analyze_source(
        url=url,
        domain=ext_res.get("domain", ""),
        author=ext_res.get("author", ""),
        publish_date=ext_res.get("publish_date", "")
    )

    evidence_data = verify_claim_evidence(raw_content, ext_res.get("title", ""))

    verdict_res = fuse_verdict(
        ml_prediction=ml_pred,
        confidence=confidence,
        evidence_data=evidence_data,
        source_data=source_analysis,
        sensational_data=sensational_signals
    )

    print("\n--------------------------------------------------")
    print("LIVE PIPELINE RESULTS:")
    print("--------------------------------------------------")
    print(f"ML Prediction: {'Real News' if ml_pred == 1 else 'Fake News'}")
    print(f"Model Confidence: {confidence:.2f}%")
    print(f"Sensational Detected: {sensational_signals['is_sensational']}")
    print(f"Source Reliability: {source_analysis['reliability_level']}")
    print(f"Source Metadata Status: {source_analysis['metadata_status']}")
    print(f"Evidence Query: {evidence_data['query']}")
    print(f"Evidence Status: {evidence_data['status']}")
    print(f"Supporting Evidence Count: {len(evidence_data['supporting_evidence'])}")
    print(f"Contradicting Evidence Count: {len(evidence_data['contradicting_evidence'])}")
    print(f"Contextual Evidence Count: {len(evidence_data['contextual_evidence'])}")
    print(f"FINAL VERDICT: {verdict_res['verdict']}")
    print(f"Decision Rationale: {verdict_res['decision_rationale']}")

    print("\nSupporting Sources Found:")
    for s in evidence_data['supporting_evidence']:
        print(f"- [{s['source']}] ({s.get('domain')}) {s['title'][:70]}")

    assert verdict_res['verdict'] == 'VERIFIED', f"Expected VERIFIED, got {verdict_res['verdict']}"
    assert len(evidence_data['supporting_evidence']) >= 1, "Expected at least 1 supporting evidence source"
    print("\n>>> LIVE ISRO TEST PASSED WITH VERDICT: VERIFIED! <<<")

if __name__ == '__main__':
    run_live_isro_test()
