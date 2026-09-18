import sys
import os
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.text_cleaner import clean_text, detect_language
from utils.source_analyzer import classify_source_tier, is_safe_url, analyze_source
from services.evidence_service import (
    decompose_claim,
    evaluate_evidence_relevance,
    verify_claim_evidence
)
from services.verdict_engine import fuse_verdict
from services.ai_reasoning import generate_structured_reasoning, generate_ai_reasoning
from utils.pdf_generator import generate_pdf_report


class TestTruthLensAI(unittest.TestCase):

    def test_01_multilingual_detection(self):
        """Test language detection across all 6 supported languages."""
        cases = [
            ("India successfully landed Chandrayaan-3 on the Moon.", "EN", "English"),
            ("चंद्रयान-3 चंद्रमा के दक्षिणी ध्रुव पर सफलतापूर्वक उतरा।", "HI", "Hindi"),
            ("El telescopio James Webb capturó imágenes del espacio profundo.", "ES", "Spanish"),
            ("Le rover de la NASA a découvert des traces d'eau sur Mars.", "FR", "French"),
            ("Die Europäische Weltraumorganisation plant eine neue Mission zum Mond.", "DE", "German"),
            ("أعلنت وكالة الفضاء عن هبوط المركبة على سطح القمر بنجاح.", "AR", "Arabic"),
        ]
        for text, expected_code, expected_lang in cases:
            res = detect_language(text)
            self.assertEqual(res["lang_code"], expected_code, f"Failed for {text}")
            self.assertEqual(res["language"], expected_lang)

    def test_02_source_tier_classification_and_wikipedia(self):
        """Test source tier taxonomy and Wikipedia reference status."""
        # Tier 1: Gov & Science
        t1_gov = classify_source_tier("isro.gov.in")
        self.assertEqual(t1_gov["tier"], 1)
        self.assertEqual(t1_gov["tier_label"], "Tier 1")
        self.assertTrue(t1_gov["is_authoritative"])

        t1_sci = classify_source_tier("nature.com")
        self.assertEqual(t1_sci["tier"], 1)
        self.assertTrue(t1_sci["is_authoritative"])

        # Tier 2: Established News
        t2_news = classify_source_tier("reuters.com")
        self.assertEqual(t2_news["tier"], 2)
        self.assertEqual(t2_news["tier_label"], "Tier 2")
        self.assertEqual(t2_news["reliability_level"], "High")

        # Tier 4: Direct input / Unverified
        t4_direct = classify_source_tier("direct input")
        self.assertEqual(t4_direct["tier"], 4)
        self.assertFalse(t4_direct["is_authoritative"])

        # Wikipedia: Must be Reference Source, NEVER Authoritative Source
        wiki = classify_source_tier("en.wikipedia.org", "Wikipedia")
        self.assertEqual(wiki["tier"], 0)
        self.assertEqual(wiki["tier_label"], "Reference Source")
        self.assertFalse(wiki["is_authoritative"], "Wikipedia must NOT be classified as authoritative!")
        self.assertTrue(wiki["is_reference_source"])

    def test_03_ssrf_protection(self):
        """Test URL blocking for private and local IP ranges."""
        blocked_urls = [
            "http://localhost/admin",
            "http://localhost:8000/api",
            "http://127.0.0.1:5000",
            "http://10.0.0.1/sensitive",
            "http://192.168.1.1/router",
            "http://172.16.0.1",
            "http://169.254.169.254/latest/meta-data",
            "ftp://example.com/file",
            "file:///etc/passwd"
        ]
        for url in blocked_urls:
            self.assertFalse(is_safe_url(url), f"SSRF vulnerability: {url} should be blocked")

        allowed_urls = [
            "https://www.isro.gov.in/Chandrayaan3.html",
            "https://www.bbc.com/news/world-asia-india-66594520",
            "https://en.wikipedia.org/wiki/Chandrayaan-3"
        ]
        for url in allowed_urls:
            self.assertTrue(is_safe_url(url), f"Safe URL falsely blocked: {url}")

    def test_04_mars_vs_lunar_gateway_false_claim(self):
        """
        CRITICAL TEST:
        Claim: Permanent city on Mars by 2027 with 10,000 citizens.
        Source: NASA Lunar Gateway corrosion.
        The source MUST NOT be classified as SUPPORTING.
        Final verdict MUST be UNCERTAIN (never VERIFIED).
        """
        claim = "NASA has confirmed that India will build the world's first permanent city on Mars by 2027, and 10,000 Indian citizens have already been selected to live there."
        source_title = "Corrosion found in NASA Lunar Gateway power and propulsion element"
        source_snippet = "Engineers detected minor corrosion issues on the Lunar Gateway space station modules orbiting the Moon."
        source_url = "https://spacenews.com/corrosion-lunar-gateway"

        claim_props = decompose_claim(claim)
        claim_locations = set().union(*[p["locations"] for p in claim_props if p.get("locations")])
        self.assertIn("mars", claim_locations)

        eval_res = evaluate_evidence_relevance(
            claim_text=claim,
            propositions=claim_props,
            item={
                "title": source_title,
                "snippet": source_snippet,
                "url": source_url,
                "source": "SpaceNews",
                "domain": "spacenews.com"
            }
        )

        # Stance must be CONTEXTUAL or IRRELEVANT, NEVER SUPPORTING
        self.assertNotEqual(eval_res["stance"], "SUPPORTING", "Lunar Gateway article must NOT support Mars city claim!")
        self.assertIn(eval_res["stance"], ["CONTEXTUAL", "IRRELEVANT"])

        # If this is the only retrieved evidence, verdict must be UNCERTAIN
        verdict_res = fuse_verdict(
            ml_prediction=0,
            confidence=95.0, # High ML confidence alone must NEVER override rule 1
            evidence_data={
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "contextual_evidence": [eval_res],
                "evidence_strength": "INSUFFICIENT"
            },
            source_data={"reliability_level": "Low", "has_domain": False},
            sensational_data={"is_sensational": False}
        )

        self.assertEqual(verdict_res["verdict"], "UNCERTAIN")
        self.assertNotEqual(verdict_res["verdict"], "VERIFIED")
        self.assertNotEqual(verdict_res["verdict"], "LIKELY FAKE")

    def test_05_chandrayaan3_genuine_claim(self):
        """
        Chandrayaan-3 genuine landing claim.
        Sources confirming landing on Moon on August 23, 2023 must be classified as SUPPORTING.
        Final verdict MUST be VERIFIED.
        """
        claim = "ISRO's Chandrayaan-3 successfully completed a soft landing on the lunar south pole on August 23, 2023."
        claim_props = decompose_claim(claim)

        source_title = "India makes historic moon landing as Chandrayaan-3 successfully touches down near lunar south pole"
        source_snippet = "ISRO confirmed the Vikram lander of Chandrayaan-3 landed successfully on August 23, 2023."
        source_url = "https://www.bbc.com/news/world-asia-india-66594520"

        eval_res = evaluate_evidence_relevance(
            claim_text=claim,
            propositions=claim_props,
            item={
                "title": source_title,
                "snippet": source_snippet,
                "url": source_url,
                "source": "BBC News",
                "domain": "bbc.com"
            }
        )

        self.assertEqual(eval_res["stance"], "SUPPORTING")

        # Fusion test with 2 independent supporting sources
        verdict_res = fuse_verdict(
            ml_prediction=0, # Even if legacy ML predicted fake due to vocabulary
            confidence=60.0,
            evidence_data={
                "supporting_evidence": [
                    {"title": source_title, "url": source_url, "domain": "bbc.com"},
                    {"title": "Chandrayaan-3 Lunar Landing Success", "url": "https://isro.gov.in", "domain": "isro.gov.in"}
                ],
                "contradicting_evidence": [],
                "evidence_strength": "STRONG"
            },
            source_data={"reliability_level": "High", "has_domain": True},
            sensational_data={"is_sensational": False}
        )

        self.assertEqual(verdict_res["verdict"], "VERIFIED")

    def test_06_absence_of_evidence_rule(self):
        """
        Core Rule:
        If supporting == 0 and contradicting == 0 -> verdict MUST be UNCERTAIN,
        even if ML confidence is 99.9%.
        """
        verdict_res = fuse_verdict(
            ml_prediction=0,
            confidence=99.9,
            evidence_data={"supporting_evidence": [], "contradicting_evidence": []},
            source_data={"reliability_level": "Low", "has_domain": False},
            sensational_data={"is_sensational": False}
        )
        self.assertEqual(verdict_res["verdict"], "UNCERTAIN")
        self.assertIn("ML confidence reflects learned textual patterns, not factual truth", verdict_res["decision_rationale"])

    def test_07_contradiction_verdict(self):
        """
        Contradictory/debunking source -> LIKELY FAKE.
        """
        verdict_res = fuse_verdict(
            ml_prediction=0,
            confidence=75.0,
            evidence_data={
                "supporting_evidence": [],
                "contradicting_evidence": [
                    {"title": "Fact Check: No human colony exists on Moon", "url": "https://snopes.com", "domain": "snopes.com"}
                ],
                "evidence_strength": "Contradicting"
            },
            source_data={"reliability_level": "Low", "has_domain": False},
            sensational_data={"is_sensational": False}
        )
        self.assertEqual(verdict_res["verdict"], "LIKELY FAKE")

    def test_08_structured_plain_language_explainability(self):
        """Test structured 3-part explainability breakdown."""
        structured = generate_structured_reasoning(
            ml_prediction=1,
            confidence=88.5,
            evidence_data={
                "supporting_evidence": [{"source": "BBC", "domain": "bbc.com"}],
                "contradicting_evidence": []
            },
            source_data={"domain": "bbc.com", "has_domain": True, "metadata_status": "Complete"},
            sensational_data={"warning_signals": []},
            final_verdict="VERIFIED"
        )
        self.assertIn("ml_explanation", structured)
        self.assertIn("evidence_explanation", structured)
        self.assertIn("verdict_explanation", structured)
        self.assertIn("bullet_points", structured)
        self.assertTrue(len(structured["bullet_points"]) >= 3)

    def test_09_pdf_report_generation(self):
        """Test PDF dossier generation with TruthLens AI branding and disclaimers."""
        pdf_payload = {
            "input_type": "News URL",
            "title": "Chandrayaan-3 Mission Landing",
            "text_summary": "ISRO successfully touched down on the lunar south pole.",
            "ml_prediction": "Real News",
            "confidence": 88.2,
            "verdict": "VERIFIED",
            "evidence_status": "Available",
            "domain": "isro.gov.in",
            "author": "ISRO Press",
            "publish_date": "2023-08-23",
            "supporting_evidence": [
                {
                    "source": "ISRO Official",
                    "source_tier": "Tier 1: Government & Official",
                    "title": "Chandrayaan-3 Update",
                    "url": "https://isro.gov.in/update"
                }
            ],
            "contradicting_evidence": [],
            "reasoning": ["Available public evidence corroborates the soft landing."]
        }
        pdf_bytes = generate_pdf_report(pdf_payload)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(len(pdf_bytes) > 1000, "PDF should not be empty")
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "Valid PDF magic number expected")


if __name__ == "__main__":
    unittest.main()
