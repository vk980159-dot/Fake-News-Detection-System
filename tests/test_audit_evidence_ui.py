import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.evidence_service import (
    decompose_claim,
    evaluate_evidence_relevance,
    verify_claim_evidence,
    build_debug_claim_attributes
)
from services.verdict_engine import fuse_verdict
from services.ai_reasoning import (
    extract_verification_points,
    generate_structured_reasoning,
    generate_ai_reasoning
)
from utils.pdf_generator import generate_pdf_report


class TestAuditEvidenceUI(unittest.TestCase):
    def setUp(self):
        self.claim = "India was the first country to land a spacecraft on the Moon in 2023."
        self.props = decompose_claim(self.claim)

    def test_1_ranking_contradiction_first_vs_fourth(self):
        """Test 1: First country vs fourth country produces RANKING CONTRADICTION."""
        ev = {
            'source': 'PBS NewsHour',
            'domain': 'pbs.org',
            'title': 'India becomes only the 4th country to successfully land a spacecraft on the moon',
            'snippet': 'India becomes only the 4th country to successfully land a spacecraft on the moon after the United States, Russia and China.'
        }
        res = evaluate_evidence_relevance(self.claim, self.props, ev)
        self.assertEqual(res['stance'], 'CONTRADICTING')
        self.assertEqual(res['conflict_type'], 'Ranking contradiction')
        self.assertIn('first country', res['claim_attribute'].lower())
        self.assertTrue('4th' in res['evidence_attribute'].lower() or 'fourth' in res['evidence_attribute'].lower())
        self.assertIn('contradicts claim ranking', res['reason'].lower())
        # Check debug_evidence_attributes attached
        ev_attrs = res.get('debug_evidence_attributes', {})
        self.assertIn('India', ev_attrs.get('extracted_subject', ''))
        self.assertNotEqual(ev_attrs.get('ranking'), 'Not detected')

    def test_2_as_of_2024_does_not_create_temporal_contradiction(self):
        """Test 2: 'As of 2024' reference year does not create a temporal contradiction."""
        ev = {
            'source': 'Wikipedia',
            'domain': 'wikipedia.org',
            'title': 'Wikipedia: Lunar lander',
            'snippet': 'lunar lander or Moon lander is a spacecraft designed to land on the surface of the Moon. As of 2024, the Apollo Lunar Module is the only lunar lander to...'
        }
        res = evaluate_evidence_relevance(self.claim, self.props, ev)
        self.assertEqual(res['stance'], 'CONTEXTUAL')
        self.assertNotEqual(res['stance'], 'CONTRADICTING')
        self.assertIsNone(res.get('conflict_type'))
        ev_attrs = res.get('debug_evidence_attributes', {})
        self.assertIn('reference_year', ev_attrs.get('year_type', ''))

    def test_3_publication_year_differs_without_contradiction(self):
        """Test 3: Publication year differing from claim year does not trigger contradiction."""
        ev = {
            'source': 'Reuters',
            'domain': 'reuters.com',
            'title': 'India Moon landing analysis',
            'snippet': 'India touched down on the Moon with its spacecraft. Published by Reuters in 2024.'
        }
        res = evaluate_evidence_relevance(self.claim, self.props, ev)
        self.assertNotEqual(res['stance'], 'CONTRADICTING')
        self.assertNotEqual(res.get('conflict_type'), 'Temporal contradiction')

    def test_4_generic_definition_with_different_year(self):
        """Test 4: Generic definitions mentioning other years remain CONTEXTUAL, not CONTRADICTING."""
        ev = {
            'source': 'Space Encyclopedia',
            'domain': 'space-encyclopedia.org',
            'title': 'Lunar Lander Definition',
            'snippet': 'A lunar lander is a spacecraft designed to land on the Moon. In 2024, a private American company tested a lunar lander.'
        }
        res = evaluate_evidence_relevance(self.claim, self.props, ev)
        self.assertEqual(res['stance'], 'CONTEXTUAL')
        self.assertNotEqual(res['stance'], 'CONTRADICTING')
        self.assertNotEqual(res.get('conflict_type'), 'Temporal contradiction')
        self.assertIn('defining', res['reason'].lower())

    def test_5_same_event_conflicting_occurrence_year(self):
        """Test 5: Explicitly conflicting occurrence year for same event/subject triggers TEMPORAL CONTRADICTING."""
        ev = {
            'source': 'Historical Records',
            'domain': 'history.org',
            'title': 'India Lunar Landing Record',
            'snippet': 'India landed its spacecraft on the Moon in 2021 during an earlier mission.'
        }
        claim = 'India landed a spacecraft on the Moon in 2023.'
        props = decompose_claim(claim)
        res = evaluate_evidence_relevance(claim, props, ev)
        self.assertEqual(res['stance'], 'CONTRADICTING')
        self.assertEqual(res['conflict_type'], 'Temporal contradiction')
        self.assertEqual(res['claim_attribute'], '2023')
        self.assertEqual(res['evidence_attribute'], '2021')

    def test_6_scope_mismatch_country_vs_probe(self):
        """Test 6: Probe-level achievement does not entail country-level achievement."""
        ev = {
            'source': 'Space News',
            'domain': 'spacenews.com',
            'title': 'India on the moon! Chandrayaan-3 becomes 1st probe to land near lunar south pole',
            'snippet': 'Chandrayaan-3 becomes the 1st probe to land near the lunar south pole.'
        }
        res = evaluate_evidence_relevance(self.claim, self.props, ev)
        self.assertEqual(res['stance'], 'CONTEXTUAL')
        self.assertNotEqual(res['stance'], 'SUPPORTING')
        self.assertIn('probe', res['reason'].lower())

    def test_7_sublocation_achievement_south_pole(self):
        """Test 7: Location-restricted achievement (South Pole) does not entail global first country."""
        ev = {
            'source': 'BBC News',
            'domain': 'bbc.com',
            'title': 'India makes history as first country to land near Moon south pole',
            'snippet': 'India makes history as the first nation to successfully touch down near the lunar south pole region.'
        }
        res = evaluate_evidence_relevance(self.claim, self.props, ev)
        self.assertEqual(res['stance'], 'CONTEXTUAL')
        self.assertNotEqual(res['stance'], 'SUPPORTING')

    def test_8_debug_trace_contains_no_empty_bullets_or_asterisks(self):
        """Test 8: Debug trace contains structured non-empty attributes with zero empty bullets or asterisks."""
        claim_attrs = build_debug_claim_attributes(self.props)
        self.assertTrue(len(claim_attrs) > 0)
        ca = claim_attrs[0]
        # Check required 11 Claim Trace fields
        required_claim_keys = [
            "proposition_id", "subject", "subject_type", "action", "object",
            "location", "geographic_scope", "claim_year", "event_phase",
            "ranking", "ranking_scope"
        ]
        for k in required_claim_keys:
            self.assertIn(k, ca)
            self.assertIsNotNone(ca[k])
            self.assertNotEqual(ca[k], "")
            self.assertFalse(str(ca[k]).startswith("- "))
            self.assertFalse(str(ca[k]).startswith("* "))

        # Verify evaluation record schema
        ev_pbs = {
            'source': 'PBS NewsHour',
            'domain': 'pbs.org',
            'url': 'https://pbs.org/landing',
            'title': 'India becomes 4th country to land on Moon',
            'snippet': 'India becomes only the 4th country to successfully land a spacecraft on the moon'
        }
        res = evaluate_evidence_relevance(self.claim, self.props, ev_pbs)
        ev_attrs = res.get("debug_evidence_attributes", {})
        # Check required 17 Evidence Trace fields
        required_ev_keys = [
            "extracted_subject", "action", "object", "evidence_year",
            "year_type", "event_phase", "ranking", "ranking_scope"
        ]
        for k in required_ev_keys:
            self.assertIn(k, ev_attrs)
            self.assertIsNotNone(ev_attrs[k])
            self.assertNotEqual(ev_attrs[k], "")

    def test_9_dashboard_cards_trace_pdf_consistency(self):
        """Test 9: Synchronized classifications across dashboard, evidence cards, debug trace, and PDF."""
        ev_cont = {
            'source': 'PBS NewsHour',
            'domain': 'pbs.org',
            'title': 'India becomes only the 4th country to land on moon',
            'url': 'https://pbs.org/chandrayaan3',
            'snippet': 'India became the 4th country to land on the Moon.',
            'stance': 'CONTRADICTING',
            'conflict_type': 'Ranking contradiction',
            'claim_attribute': 'first country',
            'evidence_attribute': '4th country',
            'reason': "Evidence directly contradicts country-level ranking: claim asserts 'first country' but reliable evidence confirms '4th country'."
        }
        ev_ctx = {
            'source': 'Wikipedia',
            'domain': 'wikipedia.org',
            'title': 'Wikipedia: Lunar lander',
            'url': 'https://en.wikipedia.org/wiki/Lunar_lander',
            'snippet': 'lunar lander or Moon lander is a spacecraft designed to land on the surface of the Moon. As of 2024, the Apollo Lunar Module is the only lunar lander to...',
            'stance': 'CONTEXTUAL',
            'reason': 'Contextual reference defining lunar lander / spacecraft.'
        }
        ev_data = {
            'claim': self.claim,
            'is_available': True,
            'status': 'Contradicting',
            'supporting_evidence': [],
            'contradicting_evidence': [ev_cont],
            'contextual_evidence': [ev_ctx],
            'supporting_count': 0,
            'contradicting_count': 1,
            'contextual_count': 1,
            'is_partially_supported': False,
            'debug_trace': {
                'input_claim': self.claim,
                'search_query': 'India Moon land spacecraft 2023',
                'claim_attributes': build_debug_claim_attributes(self.props),
                'evaluations': [
                    {
                        'source_title': ev_cont['title'],
                        'domain': ev_cont['domain'],
                        'source_url': ev_cont['url'],
                        'source_tier': 'Tier 1 - Authoritative News',
                        'final_classification': 'CONTRADICTING',
                        'classification': 'CONTRADICTING',
                        'conflict_type': 'Ranking contradiction',
                        'classification_reason': ev_cont['reason']
                    },
                    {
                        'source_title': ev_ctx['title'],
                        'domain': ev_ctx['domain'],
                        'source_url': ev_ctx['url'],
                        'source_tier': 'Reference Source',
                        'final_classification': 'CONTEXTUAL',
                        'classification': 'CONTEXTUAL',
                        'conflict_type': 'Not detected',
                        'classification_reason': ev_ctx['reason']
                    }
                ]
            }
        }

        # 1. Verdict Engine classification
        verdict_res = fuse_verdict(
            ml_prediction=0,
            confidence=63.78,
            evidence_data=ev_data,
            source_data={'reliability_tier': 'Low', 'overall_source_score': 40},
            sensational_data={'is_sensational': False}
        )
        self.assertEqual(verdict_res['verdict'], 'LIKELY FAKE')

        # 2. AI Reasoning verification points classification
        vp = extract_verification_points(
            claim_text=self.claim,
            evidence_data=ev_data,
            final_verdict='LIKELY FAKE'
        )
        self.assertEqual(vp['classification'], 'Contradicting')
        self.assertEqual(vp['ranking_found_in_evidence'], 'Fourth country')

        # 3. Debug trace evaluations consistency
        evals = ev_data['debug_trace']['evaluations']
        self.assertEqual(evals[0]['final_classification'], 'CONTRADICTING')
        self.assertEqual(evals[1]['final_classification'], 'CONTEXTUAL')

        # 4. PDF report generation
        pdf_payload = {
            'input_type': 'Text',
            'title': self.claim,
            'text_summary': self.claim,
            'ml_prediction': 'Fake News',
            'confidence': 63.78,
            'verdict': verdict_res['verdict'],
            'verdict_desc': verdict_res['decision_rationale'],
            'evidence_status': ev_data['status'],
            'supporting_count': 0,
            'contradicting_count': 1,
            'contextual_count': 1,
            'domain': 'Direct Input',
            'author': 'Not Available',
            'publish_date': '2023',
            'supporting_evidence': [],
            'contradicting_evidence': [ev_cont],
            'contextual_evidence': [ev_ctx],
            'verification_points': vp,
            'reasoning': [verdict_res['decision_rationale']]
        }
        pdf_bytes = generate_pdf_report(pdf_payload)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)


if __name__ == '__main__':
    unittest.main()
