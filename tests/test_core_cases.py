import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.verdict_engine import fuse_verdict

def test_cases_a_through_g():
    # Case A: ML Fake + Evidence 0/0 -> UNCERTAIN
    res_a = fuse_verdict(
        ml_prediction=0,
        confidence=82.0,
        evidence_data={'supporting_evidence': [], 'contradicting_evidence': []},
        source_data={'reliability_level': 'Low'},
        sensational_data={'is_sensational': False}
    )
    assert res_a['verdict'] == 'UNCERTAIN', f"Case A failed: got {res_a['verdict']}"

    # Case B: ML Real + Evidence 0/0 -> UNCERTAIN
    res_b = fuse_verdict(
        ml_prediction=1,
        confidence=91.0,
        evidence_data={'supporting_evidence': [], 'contradicting_evidence': []},
        source_data={'reliability_level': 'High'},
        sensational_data={'is_sensational': False}
    )
    assert res_b['verdict'] == 'UNCERTAIN', f"Case B failed: got {res_b['verdict']}"

    # Case C: ML Fake + Strong contradiction -> LIKELY FAKE
    res_c = fuse_verdict(
        ml_prediction=0,
        confidence=85.0,
        evidence_data={'supporting_evidence': [], 'contradicting_evidence': [{'title': 'Debunked 1'}, {'title': 'Debunked 2'}]},
        source_data={'reliability_level': 'Low'},
        sensational_data={'is_sensational': True}
    )
    assert res_c['verdict'] == 'LIKELY FAKE', f"Case C failed: got {res_c['verdict']}"

    # Case D: ML Fake + Strong supporting evidence -> VERIFIED
    res_d = fuse_verdict(
        ml_prediction=0,
        confidence=78.0,
        evidence_data={'supporting_evidence': [{'title': 'Corroboration 1'}, {'title': 'Corroboration 2'}], 'contradicting_evidence': []},
        source_data={'reliability_level': 'High'},
        sensational_data={'is_sensational': False}
    )
    assert res_d['verdict'] == 'VERIFIED', f"Case D failed: got {res_d['verdict']}"

    # Case E: ML Real + Strong supporting evidence -> VERIFIED
    res_e = fuse_verdict(
        ml_prediction=1,
        confidence=95.0,
        evidence_data={'supporting_evidence': [{'title': 'Corroboration 1'}, {'title': 'Corroboration 2'}], 'contradicting_evidence': []},
        source_data={'reliability_level': 'High'},
        sensational_data={'is_sensational': False}
    )
    assert res_e['verdict'] == 'VERIFIED', f"Case E failed: got {res_e['verdict']}"

    # Case F: Conflicting evidence -> UNCERTAIN or MISLEADING
    res_f = fuse_verdict(
        ml_prediction=0,
        confidence=60.0,
        evidence_data={'supporting_evidence': [{'title': 'Report 1'}], 'contradicting_evidence': [{'title': 'Denial 1'}]},
        source_data={'reliability_level': 'Medium'},
        sensational_data={'is_sensational': False}
    )
    assert res_f['verdict'] in ('MISLEADING', 'UNCERTAIN'), f"Case F failed: got {res_f['verdict']}"

    # Case G: Partial / contextual truth -> MISLEADING
    res_g = fuse_verdict(
        ml_prediction=0,
        confidence=75.0,
        evidence_data={'supporting_evidence': [{'title': 'Report 1'}], 'contradicting_evidence': []},
        source_data={'reliability_level': 'Medium'},
        sensational_data={'is_sensational': True}
    )
    assert res_g['verdict'] == 'MISLEADING', f"Case G failed: got {res_g['verdict']}"

    print("ALL 7 REQUIRED CORE TEST CASES (A-G) PASSED WITH ZERO ERRORS!")

if __name__ == "__main__":
    test_cases_a_through_g()
