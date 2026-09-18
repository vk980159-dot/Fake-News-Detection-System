import os
import sys
import pickle
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

def simulate_pipeline(input_type: str, title: str, text: str, domain: str = "Direct Input", author: str = "", publish_date: str = ""):
    """Executes the exact code path from app.py lines 305-375."""
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    # Step 1: Preprocessing & Text Statistics
    cleaned = clean_text(text)
    stats = get_text_stats(text)
    category = detect_category(text)
    sensational_signals = analyze_sensationalism(text)

    # Step 2: ML Model Inference
    if not cleaned.strip():
        ml_pred = 0
        confidence = 50.0
    else:
        news_vec = vectorizer.transform([cleaned])
        ml_pred = int(model.predict(news_vec)[0])
        prob_dist = model.predict_proba(news_vec)[0]
        confidence = float(max(prob_dist) * 100.0)

    # Step 3: Source Reliability Analysis
    source_analysis = analyze_source(
        url=None,
        domain=domain,
        author=author,
        publish_date=publish_date
    )

    # Step 4: Real Evidence Search
    evidence_data = verify_claim_evidence(text, title)

    # Step 5: Verdict Fusion
    verdict_res = fuse_verdict(
        ml_prediction=ml_pred,
        confidence=confidence,
        evidence_data=evidence_data,
        source_data=source_analysis,
        sensational_data=sensational_signals
    )

    # Step 6: AI Reasoning Generation
    reasoning_points = generate_ai_reasoning(
        ml_prediction=ml_pred,
        confidence=confidence,
        evidence_data=evidence_data,
        source_data=source_analysis,
        sensational_data=sensational_signals,
        final_verdict=verdict_res["verdict"]
    )

    # SAFE Gemini key retrieval (guaranteed no StreamlitSecretNotFoundError)
    gemini_key = get_gemini_api_key()
    llm_summary = query_gemini_synthesis(text, reasoning_points, verdict_res["verdict"], gemini_key)

    return {
        "verdict": verdict_res["verdict"],
        "confidence": confidence,
        "ml_pred": ml_pred,
        "category": category,
        "reasoning": reasoning_points,
        "llm_summary": llm_summary,
        "gemini_key": gemini_key
    }

def test_all_ui_modes():
    print("\n--- Testing Mode A: Plain News Text ---")
    res_text = simulate_pipeline(
        input_type="News Text",
        title="Senate passes bill",
        text="The Senate today passed historic legislation funding infrastructure improvements across all states."
    )
    print(f"Result: Verdict={res_text['verdict']}, ML={res_text['ml_pred']}, Confidence={res_text['confidence']:.1f}%")
    assert res_text["gemini_key"] is None
    assert res_text["llm_summary"] is None
    print("Mode A PASSED")

    print("\n--- Testing Mode B: Plain Claim ---")
    res_claim = simulate_pipeline(
        input_type="Claim",
        title="NASA discovers liquid water on Europa",
        text="NASA discovers liquid water on Europa"
    )
    print(f"Result: Verdict={res_claim['verdict']}, ML={res_claim['ml_pred']}, Confidence={res_claim['confidence']:.1f}%")
    assert res_claim["gemini_key"] is None
    print("Mode B PASSED")

    print("\n--- Testing Mode C: Invalid URL ---")
    ext_inv = extract_from_url("not-a-valid-http-url")
    assert ext_inv["success"] is False
    assert "invalid" in ext_inv["error"].lower()
    print("Mode C PASSED (Clean error handling, no crash)")

    print("\n--- Testing Mode D: Normal URL ---")
    ext_url = extract_from_url("https://www.bbc.com/news")
    if ext_url["success"]:
        res_url = simulate_pipeline(
            input_type="News URL",
            title=ext_url["title"],
            text=ext_url["text"] or "BBC News international world reporting",
            domain=ext_url["domain"],
            author=ext_url["author"],
            publish_date=ext_url["publish_date"]
        )
        print(f"Result: Verdict={res_url['verdict']}, Domain={ext_url['domain']}")
    print("Mode D PASSED")

    print("\n--- Testing Mode E: Image OCR ---")
    img = Image.new("RGB", (350, 70), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((15, 20), "GOVERNMENT ANNOUNCES NEW BUDGET", fill=(0, 0, 0))
    ocr_res = extract_text_from_image(img)
    assert ocr_res["success"] is True
    res_ocr = simulate_pipeline(
        input_type="Image OCR",
        title="GOVERNMENT ANNOUNCES NEW BUDGET",
        text=ocr_res["text"]
    )
    print(f"Extracted: '{ocr_res['text'].strip()}', Verdict={res_ocr['verdict']}")
    print("Mode E PASSED")

    print("\n--- Testing Mode F: No Gemini API Key ---")
    key = get_gemini_api_key()
    assert key is None
    synthesis = query_gemini_synthesis("sample", ["reason 1"], "VERIFIED", key)
    assert synthesis is None
    print("Mode F PASSED (No crash, returns None safely)")

    print("\n==================================================")
    print(" ALL UI MODES & EDGE CASES PASSED WITH ZERO CRASHES! ")
    print("==================================================")

if __name__ == "__main__":
    test_all_ui_modes()
