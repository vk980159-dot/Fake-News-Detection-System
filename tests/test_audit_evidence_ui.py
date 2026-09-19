import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.evidence_service import (
    decompose_claim,
    evaluate_evidence_relevance,
    verify_claim_evidence
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
        ev = {
            'source': 'Reuters',
            'domain': 'reuters.com',
            'title': 'India becomes 4th country to successfully land a spacecraft on the Moon',
            'snippet': 'India became the 4th country to successfully land a spacecraft on the Moon after US, Soviet Union and China.'
        }
        res = evaluate_evidence_relevance(self.claim, self.props, ev)
        self.assertEqual(res['stance'], 'CONTRADICTING')
        self.assertEqual(res['conflict_type'], 'Ranking contradiction')
        self.assertIn('first country', res['claim_attribute'].lower())
        self.assertIn('4th country', res['evidence_attribute'].lower())
        self.assertIn('contradicts claim ranking', res['reason'].lower())

    def test_2_scope_mismatch_country_vs_probe(self):
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

    def test_3_sublocation_achievement_south_pole(self):
        ev = {
            'source': 'BBC News',
            'domain': 'bbc.com',
            'title': 'India makes history as first country to land near Moon south pole',
            'snippet': 'India makes history as the first nation to successfully touch down near the lunar south pole region.'
        }
        res = evaluate_evidence_relevance(self.claim, self.props, ev)
        self.assertEqual(res['stance'], 'CONTEXTUAL')
        self.assertNotEqual(res['stance'], 'SUPPORTING')
        self.truth = ('location' in res['reason'].lower() or 'south pole' in res['reason'].lower() or 'regional' in res['reason'].lower())
        self.assertTrue(self.truth)

    def test_4_matching_country_level_ranking(self):
        claim_4 = 'India was the fourth country to land a spacecraft on the Moon.'
        props_4 = decompose_claim(claim_4)
        ev_4 = {
            'source': 'Associated Press',
            'domain': 'Enapnews.com',
            'title': 'India lands spacecraft on Moon, joining elite club as 4th nation',
            'snippet': 'India became the 4th country to land a spacecraft on the Moon, joining the US, Russia and China.'
        }
        res_4 = evaluate_evidence_relevance(claim_4, props_4, ev_4)
        self.assertEqual(res_4['stance'], 'SUPPORTING')
        self.assertIn('Directly entails', res_4["reason"])

    def test_5_related_lunar_article_without_same_event(self):
        ev_5 = {
            'source': 'Astronomy Today',
            'domain': 'astronomy.com',
            'title': 'Lunar geology and mineral resources of the Moon',
            'snippet': 'Studies of the lunar surface reveal vast reserves of water ice and valuable titanium deposits on the Moon.'
        }
        res_5 = evaluate_evidence_relevance(self.claim, self.props, ev_5)
        self.assertEqual(res_5['stance'], 'CONTEXTUAL')
        self.assertNotEqual(res_5['stance'], 'SUPPORTING')

    def test_6_evidence_exists_but_supporting_count_is_zero(self):
        ev_cont = {
            'source': 'Reuters',
            'domain': 'reuters.com',
            'title': 'India becomes 4th country to successfully land a spacecraft on the Moon',
            'snippet': 'India became the 4th country to successfully land on the Moon.',
            'stance': 'CONTRADICTING',
            'conflict_type': 'Ranking contradiction',
            'claim_attribute': '1st country',
            'evidence_attribute': '4th country',
            'reason': "Evidence directly conflicts with country-level ranking: claim asserts '1st country' but evidence confirms '4th country'."
        }
        ev_data = {
            'claim': self.claim,
            'is_available': True,
            'status': 'Contradicting',
            'supporting_evidence': [],
            'contradicting_evidence': [ev_cont],
            'contextual_evidence': [
                {'source': 'BBC', 'title': 'Chandrayaan-3 landing', 'domain': 'bbc.com', 'snippet': 'Moon landing', 'stance': 'CONTEXTUAL', 'reason': 'Contextual'}
            ],
            'supporting_count': 0,
            'contradicting_count': 1,
            'contextual_count': 1,
            'is_partially_supported': False
        }

        verdict_res = fuse_verdict(
            ml_prediction=0,
            confidence=63.78,
            evidence_data=ev_data,
            source_data={'reliability_tier': 'Low', 'overall_source_score': 40},
            sensational_data={'is_sensational': False}
        )

        self.assertEqual(verdict_res['verdict'], 'LIKELY FAKE')
        self.assertIn('fourth country', verdict_res['decision_rationale'].lower())
        self.assertIn('first country', verdict_res['decision_rationale'].lower())

        sr = generate_structured_reasoning(
            ml_prediction=0,
            confidence=63.78,
            evidence_data=ev_data,
            source_data={'reliability_tier': 'Low', 'overall_source_score': 40},
            sensational_data={'is_sensational': False},
            final_verdict=verdict_res['verdict']
        )
        self.assertIn("was assigned because the retrieved evidence directly contradicts the claim's country-level ranking", sr['verdict_explanation'])
        self.assertIn('fourth country', sr['verdict_explanation'].lower())

    def test_7_long_evidence_explanation_not_truncated(self):
        vp = extract_verification_points(
            claim_text=self.claim,
            evidence_data={
                'claim': self.claim,
                'contradicting_evidence': [{
                    'conflict_type': 'Ranking contradiction',
                    'evidence_attribute': '4th country'
                }]
            },
            final_verdict='LIKELY FAKE'
        )
        self.assertEqual(vp['claim_subject'], 'India')
        self.assertEqual(vp['object'], 'Moon')
        self.assertEqual(vp['time'], '2023')
        self.assertEqual(vp['ranking_claimed'], 'First country')
        self.assertEqual(vp['ranking_found_in_evidence'], 'Fourth country')
        self.assertEqual(vp['classification'], 'Contradicting')
        self.assertEqual(vp["reason"], 'The evidence directly conflicts with the claimed country-level ranking.')
        self.assertFalse(vp['reason'].endswith('...'))
        self.assertTrue(vp['reason'].endswith('.'))


    def test_8_pdf_report_generation(self):
        payload = {
            'input_type': 'Text',
            'title': self.claim,
            'text_summary': self.claim,
            'ml_prediction': 'Fake News',
            'confidence': 63.78,
            'verdict': 'LIKELY FAKE',
            'verdict_desc': "LIKELY FAKE was assigned because the retrieved evidence directly contradicts the claim's country-level ranking. The claim states that India was the first country, while the evidence identifies India as the fourth country to successfully land a spacecraft on the Moon.",
            'evidence_status': 'Contradicting',
            'supporting_count': 0,
            'contradicting_count': 1,
            'contextual_count': 6,
            'domain': 'Direct Input',
            'author': 'Not Available',
            'publish_date': '2023',
            'supporting_evidence': [],
            'contradicting_evidence': [{
                'source': 'Reuters',
                'domain': 'reuters.com',
                'title': 'India becomes 4th country to land on Moon',
                'url': 'https://reuters.com/article1',
                'conflict_type': 'Ranking contradiction',
                'claim_attribute': '1st country',
                'evidence_attribute': '4th country',
                'reason': 'Directly refutes country-level ranking.'
            }],
            'contextual_evidence': [{
                'source': 'SpaceNews',
                'domain': 'spacenews.com',
                'title': 'Chandrayaan-3 landing',
                'url': 'https://spacenews.com/art1',
                'reason': 'Probe level landing'
            }],
            'verification_points': {
                'claim_subject': 'India',
                'action': 'land a spacecraft',
                'object': 'Moon',
                'time': '2023',
                'ranking_claimed': 'First country',
                'ranking_found_in_evidence': 'Fourth country',
                'classification': 'Contradicting',
                'reason': 'The evidence directly conflicts with the claimed country-level ranking.'
            },
            'reasoning': [
                'The ML model classified the content as Fake News with 63.78% confidence.',
                'Live evidence analysis identified 1 direct contradiction refuting the central proposition.',
                'Retrieved evidence confirms India was the 4th country to land a spacecraft on the Moon, refuting first country.'
            ]
        }
        pdf_bytes = generate_pdf_report(payload)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)

if __name__ == '__main__':
    unittest.main()
