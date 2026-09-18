import os
import sys
import pickle
from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.text_cleaner import clean_text, detect_category, get_text_stats, analyze_sensationalism
from utils.source_analyzer import analyze_source, extract_domain
from utils.pdf_generator import generate_pdf_report
from services.content_extractor import extract_from_url, extract_from_text, extract_from_claim
from services.ocr_service import extract_text_from_image
from services.evidence_service import verify_claim_evidence
from services.ai_reasoning import generate_ai_reasoning
from services.verdict_engine import fuse_verdict

def test_text_cleaner():
    print("Testing text_cleaner...")
    raw = "BREAKING: Check this out https://fakenews.com/story! $10,000,000 lottery winner!!!"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "$" not in cleaned
    assert "breaking" in cleaned
    print("  clean_text PASSED")

    cat = detect_category("The prime minister and parliament announced election policy.")
    assert cat == "Politics", f"Expected Politics, got {cat}"
    print("  detect_category PASSED")

    stats = get_text_stats("One two three four five. Six seven eight nine ten.")
    assert stats["word_count"] == 10
    assert stats["sentence_count"] == 2
    print("  get_text_stats PASSED")

    sens = analyze_sensationalism("SHOCKING CONSPIRACY EXPOSED! THEY DONT WANT YOU TO KNOW!!!")
    assert sens["is_sensational"] is True
    assert len(sens["warning_signals"]) > 0
    print("  analyze_sensationalism PASSED")

def test_ml_model_loading():
    print("\nTesting ML Model and Vectorizer...")
    assert os.path.exists("fake_news_model.pkl"), "fake_news_model.pkl not found"
    assert os.path.exists("vectorizer.pkl"), "vectorizer.pkl not found"

    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    sample_real = (
        "WASHINGTON (Reuters) - The U.S. Senate on Thursday overwhelmingly approved legislation "
        "imposing new sanctions on Russia, Iran and North Korea, sending the measure to the White House."
    )
    cleaned = clean_text(sample_real)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    conf = max(proba) * 100
    print(f"  Sample Reuters News -> Prediction: {'Real' if pred == 1 else 'Fake'} ({conf:.2f}% confidence)")
    assert pred in (0, 1)
    assert 50.0 <= conf <= 100.0
    print("  ML model inference PASSED")

def test_ocr_service():
    print("\nTesting OCR Service...")
    img = Image.new("RGB", (320, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((15, 25), "ELECTION RESULTS 2026", fill=(0, 0, 0))
    res = extract_text_from_image(img)
    assert res["success"] is True, f"OCR failed: {res.get('error')}"
    assert "ELECTION" in res["text"].upper() or "RESULTS" in res["text"].upper()
    print(f"  Extracted OCR text: {repr(res['text'].strip())}")
    print("  OCR Service PASSED")

def test_source_analyzer():
    print("\nTesting Source Analyzer...")
    domain = extract_domain("https://www.reuters.com/world/us-politics-report")
    assert domain == "reuters.com"

    analysis = analyze_source(
        domain="reuters.com",
        author="John Doe",
        publish_date="2026-05-12",
        has_evidence=True
    )
    assert analysis["reliability_level"] == "High"
    assert "Complete" in analysis["metadata_status"]

    analysis_direct = analyze_source(domain="Direct Input")
    assert analysis_direct["domain"] == "Direct Input"
    assert analysis_direct["author"] == "Not Specified"
    assert analysis_direct["publish_date"] == "Not Specified"
    assert analysis_direct["metadata_status"] == "Not Available"
    print("  Source Analyzer PASSED")

def test_evidence_service():
    print("\nTesting Evidence Service...")
    # Real query with known historical presence
    ev = verify_claim_evidence(
        claim_text="NASA Artemis moon landing mission astronauts preparation",
        title="NASA Artemis Moon Mission"
    )
    print(f"  Evidence Search Status: {ev['status']}, Total Sources: {ev['total_sources_found']}")
    assert "supporting_evidence" in ev
    assert "contradicting_evidence" in ev
    print("  Evidence Service PASSED")

def test_verdict_engine():
    print("\nTesting Verdict Engine Fusion Matrix...")
    # Test Branch 1: Debunked claim with contradicting evidence
    res_fake = fuse_verdict(
        ml_prediction=0,
        confidence=92.0,
        evidence_data={"supporting_evidence": [], "contradicting_evidence": [{"title": "Debunked hoax"}]},
        source_data={"reliability_level": "Low"},
        sensational_data={"is_sensational": True}
    )
    assert res_fake["verdict"] == "LIKELY FAKE", f"Expected LIKELY FAKE, got {res_fake['verdict']}"

    # Test Branch 2: Corroborated claim with supporting evidence and no contradiction
    res_verified = fuse_verdict(
        ml_prediction=1,
        confidence=95.0,
        evidence_data={"supporting_evidence": [{"title": "Official Announcement"}], "contradicting_evidence": []},
        source_data={"reliability_level": "High"},
        sensational_data={"is_sensational": False}
    )
    assert res_verified["verdict"] == "VERIFIED", f"Expected VERIFIED, got {res_verified['verdict']}"

    # Test Branch 3: Sensational exaggeration on top of real event
    res_misleading = fuse_verdict(
        ml_prediction=0,
        confidence=89.0,
        evidence_data={"supporting_evidence": [{"title": "Official Announcement"}], "contradicting_evidence": []},
        source_data={"reliability_level": "Medium"},
        sensational_data={"is_sensational": True}
    )
    assert res_misleading["verdict"] == "MISLEADING", f"Expected MISLEADING, got {res_misleading['verdict']}"

    # Test Branch 4: Insufficient evidence
    res_uncertain = fuse_verdict(
        ml_prediction=1,
        confidence=62.0,
        evidence_data={"supporting_evidence": [], "contradicting_evidence": [], "is_available": False},
        source_data={"reliability_level": "Low"},
        sensational_data={"is_sensational": False}
    )
    assert res_uncertain["verdict"] == "UNCERTAIN", f"Expected UNCERTAIN, got {res_uncertain['verdict']}"
    print("  Verdict Engine 4-state fusion PASSED")

def test_pdf_generation():
    print("\nTesting PDF Report Generation...")
    pdf_data = {
        "input_type": "News Text",
        "title": "NASA Moon Mission Update",
        "text_summary": "NASA scientists confirmed the trajectory for the upcoming lunar landing mission.",
        "ml_prediction": "Real News",
        "confidence": 94.5,
        "verdict": "VERIFIED",
        "evidence_status": "Supporting",
        "domain": "reuters.com",
        "author": "Jane Smith",
        "publish_date": "2026-09-01",
        "supporting_evidence": [{"title": "NASA Press Briefing", "source": "Reuters", "url": "https://reuters.com"}],
        "contradicting_evidence": [],
        "reasoning": ["ML model classified article as likely real with 94.50% model confidence.", "Supporting evidence corroborated the event."]
    }
    pdf_bytes = generate_pdf_report(pdf_data)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
    print(f"  Generated PDF size: {len(pdf_bytes)} bytes")
    print("  PDF Generation PASSED")

if __name__ == "__main__":
    test_text_cleaner()
    test_ml_model_loading()
    test_ocr_service()
    test_source_analyzer()
    test_evidence_service()
    test_verdict_engine()
    test_pdf_generation()
    print("\n==========================================")
    print(" ALL PIPELINE TESTS PASSED SUCCESSFULLY! ")
    print("==========================================")
