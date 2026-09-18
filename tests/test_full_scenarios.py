import os
import sys
import pickle
import io
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.text_cleaner import clean_text, detect_category, get_text_stats, analyze_sensationalism
from utils.source_analyzer import analyze_source, extract_domain
from utils.pdf_generator import generate_pdf_report
from services.content_extractor import extract_from_url, extract_from_text, extract_from_claim
from services.ocr_service import extract_text_from_image
from services.evidence_service import verify_claim_evidence
from services.ai_reasoning import generate_ai_reasoning, query_gemini_synthesis, get_gemini_api_key
from services.verdict_engine import fuse_verdict

def run_all_scenarios():
    print("==================================================")
    print("RUNNING FULL VERIFICATION SCENARIOS TEST SUITE")
    print("==================================================")

    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    # 1. Plain Fake News Text
    print("\n--- Test 1: Plain Fake News Text ---")
    fake_text = (
        "BREAKING: Secret underground military bunker discovered with alien technology! "
        "Government insiders leaked shocking classified evidence proving total conspiracy and cover-up!"
    )
    cleaned = clean_text(fake_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    conf = max(proba) * 100
    sens = analyze_sensationalism(fake_text)
    source = analyze_source(domain="Direct Input")
    ev = verify_claim_evidence(fake_text)
    verdict = fuse_verdict(pred, conf, ev, source, sens)
    print(f"Prediction: {'Real' if pred == 1 else 'Fake'}, Confidence: {conf:.2f}%, Sensational: {sens['is_sensational']}")
    print(f"Final Verdict: {verdict['verdict']}")
    assert verdict["verdict"] in ("LIKELY FAKE", "UNCERTAIN", "MISLEADING")
    print("Test 1 PASSED")

    # 2. Plain Real News Text
    print("\n--- Test 2: Plain Real News Text ---")
    real_text = (
        "LONDON (Reuters) - British inflation unexpectedly held steady at 4.0 percent in January, "
        "defying forecasts of an uptick according to official data released by the Office for National Statistics."
    )
    cleaned = clean_text(real_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    conf = max(proba) * 100
    sens = analyze_sensationalism(real_text)
    source = analyze_source(domain="reuters.com", author="David Milliken", publish_date="2026-02-14")
    ev = verify_claim_evidence(real_text, title="British Inflation Steady at 4.0 Percent")
    verdict = fuse_verdict(pred, conf, ev, source, sens)
    print(f"Prediction: {'Real' if pred == 1 else 'Fake'}, Confidence: {conf:.2f}%")
    print(f"Final Verdict: {verdict['verdict']}")
    assert pred == 1, "Expected model to predict Real for Reuters text"
    assert verdict["verdict"] in ("VERIFIED", "UNCERTAIN")
    print("Test 2 PASSED")

    # 3. Valid URL handling
    print("\n--- Test 3: URL Extraction ---")
    url_res = extract_from_url("https://www.bbc.com/news")
    print(f"URL Success: {url_res['success']}, Domain: {url_res['domain']}")
    assert url_res["domain"] == "bbc.com"
    print("Test 3 PASSED")

    # 4. Invalid URL handling
    print("\n--- Test 4: Invalid URL Graceful Handling ---")
    invalid_res = extract_from_url("invalid-url-not-http")
    assert invalid_res["success"] is False
    assert "invalid" in invalid_res["error"].lower()
    print("Test 4 PASSED (Invalid URL handled gracefully without crashing)")

    # 5. Image with readable text (OCR)
    print("\n--- Test 5: Image with Readable Text (OCR) ---")
    img_valid = Image.new("RGB", (400, 70), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_valid)
    draw.text((20, 20), "CHANDRAYAAN MOON MISSION SUCCESS", fill=(0, 0, 0))
    ocr_res = extract_text_from_image(img_valid)
    assert ocr_res["success"] is True
    print(f"Extracted: '{ocr_res['text'].strip()}'")
    print("Test 5 PASSED")

    # 6. Image with no readable text (Blank image)
    print("\n--- Test 6: Image with No Readable Text (Blank Image) ---")
    img_blank = Image.new("RGB", (200, 100), color=(255, 255, 255))
    ocr_blank_res = extract_text_from_image(img_blank)
    assert ocr_blank_res["success"] is False
    assert "no readable text" in ocr_blank_res["error"].lower()
    print("Test 6 PASSED (Blank image handled cleanly without crashing)")

    # 7. Claim where evidence exists
    print("\n--- Test 7: Claim where Public Evidence Exists ---")
    claim_known = "NASA James Webb Space Telescope discovers ancient galaxies"
    ev_known = verify_claim_evidence(claim_known)
    print(f"Status: {ev_known['status']}, Sources found: {ev_known['total_sources_found']}")
    assert ev_known["total_sources_found"] > 0
    print("Test 7 PASSED")

    # 8. Claim where evidence is insufficient
    print("\n--- Test 8: Claim with Insufficient / Obscure Evidence ---")
    claim_obscure = "Xylophone zebra 99824xyz secret purple mountain"
    ev_obscure = verify_claim_evidence(claim_obscure)
    print(f"Status: {ev_obscure['status']}, Sources found: {ev_obscure['total_sources_found']}")
    assert ev_obscure["status"] in ["Insufficient", "INSUFFICIENT"]
    print("Test 8 PASSED (Accurately flagged as Insufficient)")

    # 9. Missing API credentials fallback
    print("\n--- Test 9: Missing API Credentials Fallback ---")
    # Verify safe key getter does not raise StreamlitSecretNotFoundError
    safe_key = get_gemini_api_key()
    print(f"Safe key retrieval (no secrets.toml): {safe_key}")
    llm_res = query_gemini_synthesis(
        claim_text="Sample text",
        reasons=["Reason 1", "Reason 2"],
        verdict="VERIFIED",
        api_key=None
    )
    assert llm_res is None # Gracefully falls back without error!
    print("Test 9 PASSED (Fallback safely operates without external LLM API key)")

    # 10. PDF Generation & History
    print("\n--- Test 10: PDF Dossier & History Integration ---")
    history_record = {
        "Timestamp": "2026-09-13 22:00:00",
        "InputType": "Claim",
        "Title": "James Webb Discovery",
        "Category": "Technology",
        "Result": "Real News",
        "Confidence": "96.4%",
        "Verdict": "VERIFIED",
        "Domain": "nasa.gov"
    }
    pdf_data = {
        "input_type": history_record["InputType"],
        "title": history_record["Title"],
        "text_summary": "NASA scientists confirmed new images from Webb telescope.",
        "ml_prediction": history_record["Result"],
        "confidence": 96.4,
        "verdict": history_record["Verdict"],
        "evidence_status": "Supporting",
        "domain": history_record["Domain"],
        "author": "Dr. Astrophysicist",
        "publish_date": "2026-08-15",
        "supporting_evidence": [{"title": "Webb Deep Field", "source": "NASA", "url": "https://nasa.gov"}],
        "contradicting_evidence": [],
        "reasoning": ["Model confidence is 96.40%.", "1 corroborating source identified."]
    }
    pdf_bytes = generate_pdf_report(pdf_data)
    assert len(pdf_bytes) > 2000
    print("Test 10 PASSED (PDF generated cleanly)")

    print("\n==================================================")
    print(" ALL 10 COMPREHENSIVE SCENARIOS PASSED WITH ZERO ERRORS! ")
    print("==================================================")

if __name__ == "__main__":
    run_all_scenarios()
