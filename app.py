import streamlit as st
import pickle
import os
import io
from datetime import datetime
import pandas as pd
from PIL import Image

# ─── Modular Services & Utilities ─────────────────────────────────────────────
from utils.text_cleaner import clean_text, detect_category, get_text_stats, analyze_sensationalism, detect_language
from utils.source_analyzer import analyze_source, extract_domain, classify_source_tier
from utils.pdf_generator import generate_pdf_report
from services.content_extractor import extract_from_url, extract_from_text, extract_from_claim
from services.ocr_service import extract_text_from_image
from services.evidence_service import verify_claim_evidence
from services.ai_reasoning import generate_ai_reasoning, generate_structured_reasoning, query_gemini_synthesis, get_gemini_api_key
from services.verdict_engine import fuse_verdict

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens AI — Global News Verification System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .verdict-card {
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
    .verdict-verified { background: linear-gradient(135deg, #15803d, #16a34a); }
    .verdict-misleading { background: linear-gradient(135deg, #c2410c, #ea580c); }
    .verdict-fake { background: linear-gradient(135deg, #b91c1c, #dc2626); }
    .verdict-uncertain { background: linear-gradient(135deg, #334155, #475569); }
    
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-high { background-color: #dcfce7; color: #166534; }
    .badge-medium { background-color: #fef9c3; color: #854d0e; }
    .badge-low { background-color: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# ─── Load Saved Model & Vectorizer ────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load the trained TF-IDF vectorizer and Logistic Regression classifier."""
    model_path = "fake_news_model.pkl"
    vect_path = "vectorizer.pkl"
    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        return None, None, "Model or vectorizer file missing from project directory."
    try:
        model = pickle.load(open(model_path, "rb"))
        vectorizer = pickle.load(open(vect_path, "rb"))
        return model, vectorizer, None
    except Exception as e:
        return None, None, f"Error loading ML artifacts: {str(e)}"

model, vectorizer, model_err = load_models()

# ─── Session State Initialization ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "last_verification" not in st.session_state:
    st.session_state.last_verification = None

if "extracted_content" not in st.session_state:
    st.session_state.extracted_content = {
        "text": "",
        "title": "",
        "author": "",
        "publish_date": "",
        "domain": "",
        "input_type": "Text"
    }

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🌐 TruthLens AI")
st.sidebar.caption("AI-Powered Global News Verification System")
st.sidebar.markdown("*\"Don't trust the headline. Verify the claim.\"*")

st.sidebar.info("""
**Architecture:** 6-Stage Hybrid AI & ML
- **Classifier:** Logistic Regression
- **Vectorization:** TF-IDF (44,898 articles)
- **OCR Engine:** Tesseract OCR (with confidence scoring)
- **Entailment:** Claim-Level Proposition Verification
- **Corroboration:** Live Public Evidence (Tier 1-4)
- **Verdict Fusion:** Transparent Multi-Factor Rules
- **Trained Accuracy:** 98.63%
""")

st.sidebar.metric(
    "📊 Total Verifications",
    len(st.session_state.history)
)

st.sidebar.subheader("📜 Recent History")

if st.session_state.history:
    for item in st.session_state.history[-5:]:
        verdict_badge = {
            "VERIFIED": "🟢",
            "MISLEADING": "🟠",
            "LIKELY FAKE": "🔴",
            "UNCERTAIN": "⚪"
        }.get(item.get("Verdict"), "⚪")
        
        st.sidebar.markdown(
            f"**{verdict_badge} {item.get('Verdict', 'N/A')}**  \n"
            f"*{item.get('InputType', 'Text')}* | {item.get('Category', 'News')}  \n"
            f"<small>{item.get('Title', 'Article')[:40]}...</small>",
            unsafe_allow_html=True
        )
        st.sidebar.markdown("---")

    if st.sidebar.button("🗑 Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_verification = None
        st.rerun()
else:
    st.sidebar.caption("No verifications yet.")

st.sidebar.markdown("---")
st.sidebar.markdown("🔒 **Privacy & Ephemeral Notice**")
st.sidebar.caption("TruthLens AI does not permanently store submitted article texts or claims on external servers. Verification queries are processed ephemerally.")

# ─── Main Title ───────────────────────────────────────────────────────────────
st.markdown("""
# 🌐 TruthLens AI
### AI-Powered Global News Verification System
*"Don't trust the headline. Verify the claim."*

Verify news articles, URLs, claims, and image screenshots using **Hybrid AI & ML** combined with **deterministic proposition entailment** and **live evidence cross-referencing**.
""")

if model_err:
    st.error(f"⚠️ {model_err}")
    st.stop()

# ─── Multi-Input Types ────────────────────────────────────────────────────────
st.subheader("📥 Choose Input Method")
tab_text, tab_url, tab_claim, tab_image = st.tabs([
    "📝 News Text",
    "🔗 News URL",
    "💡 Plain Claim",
    "🖼️ Image Upload (OCR)"
])

# 1. Plain News Text Tab
with tab_text:
    news_input = st.text_area(
        "Paste Full News Article Text",
        value=st.session_state.extracted_content["text"] if st.session_state.extracted_content["input_type"] == "Text" else "",
        height=180,
        placeholder="Paste complete news article here (minimum 20 words for best ML accuracy)...",
        key="text_input_area"
    )
    if st.button("📌 Set as Active Content", key="btn_set_text"):
        if news_input.strip():
            extracted = extract_from_text(news_input)
            st.session_state.extracted_content = {
                "text": extracted["text"],
                "title": extracted["title"],
                "author": extracted["author"],
                "publish_date": extracted["publish_date"],
                "domain": extracted["domain"],
                "input_type": "News Text"
            }
            st.success("✅ News text set as active content for verification.")
        else:
            st.warning("Please enter text first.")

# 2. URL Tab
with tab_url:
    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        url_input = st.text_input("News Webpage URL", placeholder="https://example.com/article-url", key="url_input_box")
    with col_u2:
        st.write("")
        st.write("")
        fetch_btn = st.button("🌐 Extract from URL", key="btn_fetch_url", use_container_width=True)

    if fetch_btn:
        if not url_input.strip():
            st.warning("⚠️ Please enter a URL first.")
        else:
            with st.spinner("Extracting article text, author, and metadata..."):
                ext_res = extract_from_url(url_input)
                if ext_res["success"]:
                    st.session_state.extracted_content = {
                        "text": ext_res["text"],
                        "title": ext_res["title"],
                        "author": ext_res["author"],
                        "publish_date": ext_res["publish_date"],
                        "domain": ext_res["domain"],
                        "url": ext_res.get("url", url_input.strip()),
                        "input_type": "News URL"
                    }
                    st.success(f"✅ Extracted article from **{ext_res['domain']}**")
                    with st.expander("📄 View Extracted Content Preview", expanded=True):
                        st.write(f"**Title:** {ext_res['title']}")
                        st.write(f"**Author:** {ext_res['author']} | **Date:** {ext_res['publish_date']}")
                        st.write(f"**Text Preview:** {ext_res['text'][:350]}...")
                else:
                    st.error(f"❌ {ext_res['error']}")

# 3. Plain Claim Tab
with tab_claim:
    claim_input = st.text_input(
        "Enter Statement or Claim to Fact-Check",
        placeholder="e.g. Scientists discover liquid water on Jupiter's moon Europa",
        key="claim_input_box"
    )
    if st.button("📌 Set Claim as Active", key="btn_set_claim"):
        if claim_input.strip():
            extracted = extract_from_claim(claim_input)
            st.session_state.extracted_content = {
                "text": extracted["text"],
                "title": extracted["title"],
                "author": extracted["author"],
                "publish_date": extracted["publish_date"],
                "domain": extracted["domain"],
                "input_type": "Claim"
            }
            st.success("✅ Claim set as active content for verification.")
        else:
            st.warning("Please enter a claim statement.")

# 4. Image Upload (OCR) Tab
with tab_image:
    uploaded_image = st.file_uploader(
        "Upload News Screenshot or Newspaper Image",
        type=["png", "jpg", "jpeg", "webp"],
        key="ocr_uploader"
    )
    if uploaded_image:
        col_img1, col_img2 = st.columns([1, 2])
        with col_img1:
            try:
                pil_img = Image.open(uploaded_image)
                st.image(pil_img, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Could not read uploaded image: {e}")
                pil_img = None
        
        with col_img2:
            if st.button("🔍 Extract Text via OCR", key="btn_run_ocr"):
                if pil_img:
                    with st.spinner("Running Optical Character Recognition (Tesseract)..."):
                        ocr_res = extract_text_from_image(pil_img)
                        if ocr_res["success"]:
                            st.session_state.extracted_content = {
                                "text": ocr_res["text"],
                                "title": (ocr_res["text"].split("\n")[0][:100] if ocr_res["text"] else "Image Text"),
                                "author": "Image OCR",
                                "publish_date": "Not Available",
                                "domain": "Direct Image Upload",
                                "input_type": "Image OCR",
                                "ocr_confidence": ocr_res.get("confidence", 0.0),
                                "is_low_confidence": ocr_res.get("is_low_confidence", False)
                            }
                            st.success(f"✅ Extracted {ocr_res['word_count']} words from image! (OCR Confidence: {ocr_res.get('confidence', 0):.1f}%)")
                        else:
                            st.error(f"❌ {ocr_res['error']}")

            if st.session_state.extracted_content["input_type"] == "Image OCR":
                if st.session_state.extracted_content.get("is_low_confidence"):
                    st.warning("⚠️ **Low OCR confidence (< 60%). Results may be unreliable.** Please review and correct the text below before running verification.")
                edited_ocr = st.text_area(
                    "Extracted OCR Text (Review & Edit if needed):",
                    value=st.session_state.extracted_content["text"],
                    height=120,
                    key="ocr_text_preview"
                )
                if edited_ocr != st.session_state.extracted_content["text"]:
                    st.session_state.extracted_content["text"] = edited_ocr

# ─── Active Content Status ────────────────────────────────────────────────────
active = st.session_state.extracted_content
st.markdown("---")

has_active_text = bool(active["text"].strip())

col_act1, col_act2 = st.columns([3, 1])
with col_act1:
    if has_active_text:
        lang_res = detect_language(active["text"])
        lang_badge = f" &nbsp;•&nbsp; 🌐 **Language:** `{lang_res['language']} ({lang_res['lang_code']})`"
        st.info(f"🎯 **Active Input:** [{active['input_type']}] {active['title'][:70]}{lang_badge}")
    else:
        st.warning("👉 Select an input method above and provide news content or a claim to proceed.")

with col_act2:
    verify_button = st.button("🚀 Run Verification", use_container_width=True, type="primary", disabled=not has_active_text)

# ─── Execute Verification Pipeline ────────────────────────────────────────────
if verify_button and has_active_text:
    raw_content = active["text"].strip()
    
    with st.spinner("Analyzing content: Preprocessing, ML Classification, Evidence Gathering & Fusion..."):
        # Step 1: Preprocessing & Text Statistics
        cleaned = clean_text(raw_content)
        stats = get_text_stats(raw_content)
        category = detect_category(raw_content)
        sensational_signals = analyze_sensationalism(raw_content)

        # Step 2: ML Model Inference
        # Note: Model trained on news text. If cleaned text is empty, handle safely
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
            url=active.get("url"),
            domain=active.get("domain", ""),
            author=active.get("author", ""),
            publish_date=active.get("publish_date", "")
        )

        # Step 4: Real Evidence Search (No Fabrication)
        evidence_data = verify_claim_evidence(raw_content, active.get("title", ""))
        evidence_data["claim"] = raw_content

        # Step 5: Verdict Fusion
        verdict_res = fuse_verdict(
            ml_prediction=ml_pred,
            confidence=confidence,
            evidence_data=evidence_data,
            source_data=source_analysis,
            sensational_data=sensational_signals
        )
        final_verdict = verdict_res["verdict"]

        # Step 6: AI Reasoning Generation
        structured_reasoning = generate_structured_reasoning(
            ml_prediction=ml_pred,
            confidence=confidence,
            evidence_data=evidence_data,
            source_data=source_analysis,
            sensational_data=sensational_signals,
            final_verdict=final_verdict
        )
        reasoning_points = structured_reasoning.get("bullet_points", [])

        # Optional Gemini synthesis if API key is provided
        gemini_key = get_gemini_api_key()
        llm_summary = query_gemini_synthesis(raw_content, reasoning_points, final_verdict, gemini_key)

        # Record Verification State
        verification_payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_type": active["input_type"],
            "title": active.get("title") or "Article Content",
            "raw_text": raw_content,
            "text_summary": raw_content[:400],
            "category": category,
            "stats": stats,
            "ml_prediction": "Real News" if ml_pred == 1 else "Fake News",
            "ml_raw_pred": ml_pred,
            "confidence": confidence,
            "verdict": final_verdict,
            "verdict_res": verdict_res,
            "evidence_status": evidence_data.get("status") or evidence_data.get("evidence_strength", "Available"),
            "evidence_data": evidence_data,
            "source_data": source_analysis,
            "sensational_signals": sensational_signals,
            "reasoning": reasoning_points,
            "structured_reasoning": structured_reasoning,
            "llm_summary": llm_summary
        }
        
        st.session_state.last_verification = verification_payload

        # Append to History
        st.session_state.history.append({
            "Timestamp": verification_payload["timestamp"],
            "InputType": active["input_type"],
            "Title": verification_payload["title"],
            "Category": category,
            "Result": verification_payload["ml_prediction"],
            "Confidence": f"{confidence:.1f}%",
            "Verdict": final_verdict,
            "Domain": source_analysis.get("domain", "Direct Input")
        })

def render_evidence_card(item: dict, classification: str, index: int):
    """
    Renders an evidence item in a clean, professional, fully-wrapped structured card.
    Guarantees no truncated text, clear proposition matching, and expandable full details.
    """
    tier_label = item.get("tier_badge") or item.get("source_tier") or "Public Source"
    source_name = item.get("source") or "Independent Publisher"
    domain = item.get("domain") or "news.google.com"
    title = item.get("title") or "Untitled Report"
    url = item.get("url") or "#"
    pub_date = item.get("pub_date") or "Recent"
    matched_prop = item.get("matched_proposition") or "P1 (Central Claim)"
    excerpt = item.get("snippet") or "No excerpt available."
    reason = item.get("reason") or "Evidence relevance verified."
    conflict_type = item.get("conflict_type")
    claim_attr = item.get("claim_attribute")
    ev_attr = item.get("evidence_attribute")

    if classification == "SUPPORTING":
        badge_cls = "badge-high"
        card_border = "#16a34a"
        bg_tint = "#f0fdf4"
        icon = "✅"
    elif classification == "CONTRADICTING":
        badge_cls = "badge-low"
        card_border = "#dc2626"
        bg_tint = "#fef2f2"
        icon = "🚨"
    else:
        badge_cls = "badge-medium"
        card_border = "#64748b"
        bg_tint = "#f8fafc"
        icon = "ℹ️"

    st.markdown(f"""
    <div style="border: 1.5px solid {card_border}; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; background-color: {bg_tint}; word-wrap: break-word; white-space: normal;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 8px;">
            <span class="metric-badge {badge_cls}" style="font-size: 0.9rem; padding: 4px 12px;">
                {icon} {classification}
            </span>
            <span style="font-size: 0.85rem; color: #475569; font-weight: 500;">
                🏛️ {tier_label} &nbsp;|&nbsp; 📅 {pub_date}
            </span>
        </div>
        <h4 style="margin: 6px 0 8px 0; color: #0f172a; font-size: 1.15rem; line-height: 1.4;">
            <a href="{url}" target="_blank" style="text-decoration: none; color: #1e40af;">
                {title}
            </a>
        </h4>
        <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 10px;">
            <b>Source:</b> {source_name} &nbsp;•&nbsp; <b>Domain:</b> <code>{domain}</code> &nbsp;•&nbsp; 
            <a href="{url}" target="_blank" style="color: #2563eb; text-decoration: underline; word-break: break-all;">
                Direct Article Link ↗
            </a>
        </div>
        <div style="font-size: 0.9rem; color: #334155; margin-bottom: 4px;">
            🎯 <b>Matched Proposition:</b> <code>{matched_prop}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if conflict_type:
        st.error(
            f"🚨 **{conflict_type}:**  \n"
            f"• **Claim Asserted:** `{claim_attr}`  \n"
            f"• **Evidence Confirms:** `{ev_attr}`"
        )

    st.markdown("**Evidence Excerpt:**")
    st.info(f"\"{excerpt}\"")
    if len(excerpt) > 180 or len(title) > 100:
        with st.expander("📖 Read full evidence & details", expanded=False):
            st.write(f"**Full Headline:** {title}")
            st.write(f"**Source URL:** {url}")
            st.write(f"**Publisher / Domain:** {source_name} (`{domain}`)")
            st.write(f"**Full Snippet / Excerpt:** {excerpt}")

    st.markdown(f"💡 **Explanation:** {reason}")
    st.markdown("---")

# ─── Render Verification Results ──────────────────────────────────────────────
if st.session_state.last_verification:
    res = st.session_state.last_verification
    verdict = res["verdict"]
    
    st.markdown("---")
    st.markdown("## 📢 Verification Analysis & Results")

    # ── Verdict Banner ──
    banner_classes = {
        "VERIFIED": "verdict-verified",
        "MISLEADING": "verdict-misleading",
        "LIKELY FAKE": "verdict-fake",
        "UNCERTAIN": "verdict-uncertain"
    }
    banner_cls = banner_classes.get(verdict, "verdict-uncertain")

    supp_cnt = res["evidence_data"].get("supporting_count", len(res["evidence_data"].get("supporting_evidence", [])))
    cont_cnt = res["evidence_data"].get("contradicting_count", len(res["evidence_data"].get("contradicting_evidence", [])))
    ctx_cnt = res["evidence_data"].get("contextual_count", len(res["evidence_data"].get("contextual_evidence", [])))

    verdict_desc = res["verdict_res"].get("decision_rationale")
    if not verdict_desc:
        if verdict == "VERIFIED":
            verdict_desc = f"Available public evidence supports the central claim. {supp_cnt} supporting public source(s) found and no contradictory source identified."
        elif verdict == "MISLEADING":
            verdict_desc = "Elements of the claim conflict with facts, exaggerate details, or use sensational framing."
        elif verdict == "LIKELY FAKE":
            verdict_desc = "Available public evidence directly contradicts or refutes the central claim."
        else:
            verdict_desc = "Evidence is insufficient or conflicting; unable to independently certify truth or falsehood."

    st.markdown(f"""
    <div class="verdict-card {banner_cls}">
        <h1 style="margin:0; font-size: 2.3rem;">FINAL VERDICT: {verdict}</h1>
        <p style="margin-top: 8px; font-size: 1.05rem; opacity: 0.95;">{verdict_desc}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics Overview ──
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            label="Model Confidence",
            value=f"{res['confidence']:.2f}%",
            help="Statistical confidence of the TF-IDF + Logistic Regression model. This reflects pattern similarity to the training dataset, NOT absolute truth probability."
        )
        st.caption("Model confidence indicates classifier certainty, not factual truth.")
    with col_m2:
        st.metric(
            label="ML Prediction",
            value=res["ml_prediction"],
            help="Direct binary classification from Logistic Regression model."
        )
    with col_m3:
        st.metric(
            label="Evidence Status",
            value=res["evidence_status"],
            help="Status of real-time public corroboration (Google News RSS & Wikipedia)."
        )
    with col_m4:
        st.metric(
            label="Source Domain",
            value=res["source_data"]["domain"][:18],
            help="Extracted publisher domain or direct input type."
        )

    # ── FEATURE 9: Visual Score Dashboard ──
    st.markdown("### 📊 Visual Evaluation Dashboard")
    col_v1, col_v2, col_v3 = st.columns(3)

    with col_v1:
        c_lvl = res["verdict_res"]["model_confidence_level"]
        badge_cls = "badge-high" if c_lvl == "High" else ("badge-medium" if c_lvl == "Medium" else "badge-low")
        st.markdown(f"**ML Model Confidence:** <span class='metric-badge {badge_cls}'>{c_lvl}</span>", unsafe_allow_html=True)
        st.progress(min(1.0, res["confidence"] / 100.0))

    with col_v2:
        e_lvl = res["verdict_res"]["evidence_strength"]
        if e_lvl in ("Stronger", "Available", "Strong"):
            badge_cls = "badge-high"
        elif e_lvl in ("Contradicting", "Strong Contradiction", "Conflicting"):
            badge_cls = "badge-low"
        else:
            badge_cls = "badge-medium"
        st.markdown(f"**Evidence Strength:** <span class='metric-badge {badge_cls}'>{e_lvl}</span>", unsafe_allow_html=True)
        st.caption(f"Supporting: {supp_cnt} | Contradicting: {cont_cnt}")

    with col_v3:
        s_lvl = res["verdict_res"]["source_information_level"]
        badge_cls = "badge-high" if s_lvl == "High" else ("badge-medium" if s_lvl == "Medium" else "badge-low")
        st.markdown(f"**Source Reliability:** <span class='metric-badge {badge_cls}'>{s_lvl}</span>", unsafe_allow_html=True)
        st.caption(f"Metadata: {res['source_data']['metadata_status']}")

    # ── FEATURE 6 & 8: AI Reasoning & "Why this result?" ──
    st.markdown("### 🤖 Why this result?")
    if res.get("llm_summary"):
        st.info(f"**AI Synthesis:** {res['llm_summary']}")

    col_r1, col_r2, col_r3 = st.columns(3)
    structured_r = res.get("structured_reasoning", {})
    with col_r1:
        st.markdown("##### 🧠 ML Linguistic Analysis")
        st.markdown(structured_r.get("ml_explanation", f"Model classified content as {res['ml_prediction']} with {res['confidence']:.1f}% confidence."))
    with col_r2:
        st.markdown("##### 🔍 Live Evidence Corroboration")
        st.markdown(structured_r.get("evidence_explanation", "Evidence cross-referencing completed."))
    with col_r3:
        st.markdown("##### ⚖️ Verdict Fusion Rationale")
        st.markdown(structured_r.get("verdict_explanation", f"Assigned verdict: {verdict}."))

    with st.expander("📋 Detailed Verification Points", expanded=True if cont_cnt > 0 else False):
        vp = structured_r.get("verification_points")
        if vp:
            col_vp1, col_vp2 = st.columns(2)
            with col_vp1:
                st.markdown(f"**Claim Subject:** {vp.get('claim_subject', 'N/A')}")
                st.markdown(f"**Action:** {vp.get('action', 'N/A')}")
                st.markdown(f"**Object / Target:** {vp.get('object', 'N/A')}")
                st.markdown(f"**Time / Date:** {vp.get('time', 'N/A')}")
            with col_vp2:
                st.markdown(f"**Ranking Claimed:** {vp.get('ranking_claimed', 'N/A')}")
                st.markdown(f"**Ranking in Evidence:** {vp.get('ranking_found_in_evidence', 'N/A')}")
                st.markdown(f"**Classification:** `{vp.get('classification', 'N/A')}`")
                st.markdown(f"**Reason:** {vp.get('reason', 'N/A')}")
            st.markdown("---")
        if res.get("reasoning"):
            st.markdown("**Decision Engine Findings:**")
            for reason in res["reasoning"]:
                st.markdown(f"- {reason}")

    # ── FEATURE 4 & 8: Evidence Found ──
    st.markdown("### 🔍 Real Evidence Cross-Referencing")
    if cont_cnt > 0:
        st.error(f"🚨 **Contradictory Public Evidence Identified ({cont_cnt} source(s)):** Direct refutation found in independent public records contradicting the claim's core assertions.")

    if not res["evidence_data"]["is_available"]:
        st.warning("⚠️ **External evidence verification unavailable.** No matching live public reports were identified. The verdict reflects stylistic ML patterns and source metadata without hallucinating external sources.")
    else:
        # If contradiction exists and supporting is 0, activate Contradicting tab first so user immediately sees contradiction
        if cont_cnt > 0 and supp_cnt == 0:
            tab_cont, tab_ctx, tab_supp = st.tabs([
                f"🚨 Contradicting Evidence ({cont_cnt})",
                f"ℹ️ Contextual References ({ctx_cnt})",
                f"✅ Supporting Evidence ({supp_cnt})"
            ])
            with tab_cont:
                for idx, item in enumerate(res["evidence_data"].get("contradicting_evidence", []), 1):
                    render_evidence_card(item, "CONTRADICTING", idx)
            with tab_ctx:
                if res["evidence_data"].get("contextual_evidence"):
                    for idx, item in enumerate(res["evidence_data"]["contextual_evidence"], 1):
                        render_evidence_card(item, "CONTEXTUAL", idx)
                else:
                    st.write("No additional background articles returned.")
            with tab_supp:
                st.warning(f"⚠️ No corroborating reports found. Note: {cont_cnt} contradictory source(s) identified refuting this claim (see Contradicting tab).")
        else:
            tab_supp, tab_cont, tab_ctx = st.tabs([
                f"✅ Supporting Evidence ({supp_cnt})",
                f"🚨 Contradicting Evidence ({cont_cnt})",
                f"ℹ️ Contextual References ({ctx_cnt})"
            ])
            with tab_supp:
                if res["evidence_data"].get("supporting_evidence"):
                    for idx, item in enumerate(res["evidence_data"]["supporting_evidence"], 1):
                        render_evidence_card(item, "SUPPORTING", idx)
                else:
                    if cont_cnt > 0:
                        st.warning(f"⚠️ No direct corroborating reports found. Note: {cont_cnt} contradictory source(s) were identified (see Contradicting tab).")
                    else:
                        st.write("No direct corroborating reports found in public news feeds.")
            with tab_cont:
                if res["evidence_data"].get("contradicting_evidence"):
                    for idx, item in enumerate(res["evidence_data"]["contradicting_evidence"], 1):
                        render_evidence_card(item, "CONTRADICTING", idx)
                else:
                    st.write("No direct contradictory or debunking reports found.")
            with tab_ctx:
                if res["evidence_data"].get("contextual_evidence"):
                    for idx, item in enumerate(res["evidence_data"]["contextual_evidence"], 1):
                        render_evidence_card(item, "CONTEXTUAL", idx)
                else:
                    st.write("No additional background articles returned.")

        # ── Developer Debug Mode: Proposition Entailment Trace ──
        debug_trace = res["evidence_data"].get("debug_trace")
        if debug_trace:
            with st.expander("🛠️ Developer Debug Mode: Proposition Entailment Trace", expanded=False):
                st.markdown(f"**Input Claim:** `{debug_trace.get('input_claim', '')}`")
                st.markdown(f"**Extracted Search Query:** `{debug_trace.get('search_query', '')}`")
                st.markdown(f"**Evidence Strength Assigned:** `{res['evidence_data'].get('evidence_strength', 'INSUFFICIENT')}`")
                st.markdown(f"**Independent Publishers Count:** `{res['evidence_data'].get('independent_publishers_count', 0)}`")
                
                # Extracted Claim Attributes
                claim_attrs = debug_trace.get("claim_attributes", [])
                if claim_attrs:
                    st.markdown("#### 📌 Extracted Claim Attributes:")
                    for ca in claim_attrs:
                        st.markdown(f"**Proposition {ca.get('proposition_id')}:** *\"{ca.get('proposition_text')}\"*")
                        col_ca1, col_ca2 = st.columns(2)
                        with col_ca1:
                            st.markdown(f"- **Subject:** `{ca.get('subject')}` ({ca.get('subject_type')})")
                            st.markdown(f"- **Actions:** `{', '.join(ca.get('actions', []))}`")
                            st.markdown(f"- **Objects:** `{', '.join(ca.get('objects', []))}`")
                        with col_ca2:
                            st.markdown(f"- **Locations:** `{', '.join(ca.get('locations', []))}` (Scope: `{ca.get('geographic_scope')}`)")
                            st.markdown(f"- **Occurrence Years:** `{', '.join(str(y) for y in ca.get('years', []))}` (Phase: `{ca.get('event_phase')}`)")
                            st.markdown(f"- **Rankings / Superlatives:** `{', '.join(ca.get('rankings', []))}`")
                    st.markdown("---")
                else:
                    st.markdown("**Claim Propositions:**")
                    for prop in debug_trace.get("claim_propositions", []):
                        st.markdown(f"- `{prop}`")
                    st.markdown("---")
                
                st.markdown("#### 🔬 Retrieved Sources & Proposition Entailment:")
                evals = debug_trace.get("evaluations", [])
                if evals:
                    for idx, ev in enumerate(evals, 1):
                        badge = "✅ SUPPORTING" if ev["classification"] == "SUPPORTING" else ("🚨 CONTRADICTING" if ev["classification"] == "CONTRADICTING" else "ℹ️ CONTEXTUAL")
                        tier_lbl = ev.get("tier_badge") or ev.get("source_tier") or "General"
                        dom_str = f" (`{ev.get('domain')}`)" if ev.get('domain') else ""
                        url_str = f" — [Article Link]({ev.get('url')})" if ev.get('url') else ""
                        st.markdown(f"**{idx}. [{ev.get('source')}]**{dom_str}{url_str} — `{tier_lbl}` — **{badge}**")
                        st.markdown(f"- **Source Title/Claim:** {ev.get('source_claim')}")
                        st.markdown(f"- **Semantic Relevance:** `{int(ev.get('semantic_relevance', 0)*100)}%`")
                        st.markdown(f"- **Proposition Match:** `{ev.get('proposition_match')}`")
                        if ev.get("ranking_comparison"):
                            st.markdown(f"- **Ranking Comparison:** `{ev.get('ranking_comparison')}`")
                        if ev.get("scope_comparison"):
                            st.markdown(f"- **Scope Comparison:** `{ev.get('scope_comparison')}`")
                        if ev.get("conflict_type"):
                            st.markdown(f"- 🚨 **Conflict Type:** `{ev.get('conflict_type')}` (Claim: `{ev.get('claim_attribute')}` vs Evidence: `{ev.get('evidence_attribute')}`)")
                        st.markdown(f"- **Final Classification Reason:** {ev.get('final_classification_reason') or ev.get('reason')}")
                        if ev.get("source_summary"):
                            st.caption(f"Summary: {ev.get('source_summary')}")
                        st.markdown("---")
                else:
                    st.write("No external source evaluations recorded.")

    # ── FEATURE 5: Source Reliability Section ──
    st.markdown("### 🏛️ Source & Publisher Analysis")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.write(f"**Domain:** `{res['source_data']['domain']}`")
        st.write(f"**Metadata Status:** {res['source_data']['metadata_status']}")
    with col_s2:
        st.write(f"**Author:** {res['source_data']['author']}")
        st.write(f"**Publication Date:** {res['source_data']['publish_date']}")
    with col_s3:
        st.write(f"**Assessment:** {res['source_data']['reliability_notes']}")

    # ── FEATURE 10: News Statistics (Always Preserved) ──
    st.markdown("### 📑 Content Linguistic Statistics")
    col_st1, col_st2, col_st3, col_st4 = st.columns(4)
    with col_st1:
        st.metric("📝 Words", res["stats"]["word_count"])
    with col_st2:
        st.metric("📄 Sentences", res["stats"]["sentence_count"])
    with col_st3:
        st.metric("⏱ Reading Time", f"{res['stats']['reading_time']} min")
    with col_st4:
        st.metric("📰 Detected Category", res["category"])

    # ── FEATURE 11: PDF Report Generation ──
    st.markdown("### 📄 Export Verification Dossier")
    pdf_payload = {
        "input_type": res["input_type"],
        "title": res["title"],
        "text_summary": res["text_summary"],
        "ml_prediction": res["ml_prediction"],
        "confidence": res["confidence"],
        "verdict": res["verdict"],
        "verdict_desc": verdict_desc,
        "evidence_status": res["evidence_status"],
        "supporting_count": supp_cnt,
        "contradicting_count": cont_cnt,
        "contextual_count": ctx_cnt,
        "domain": res["source_data"]["domain"],
        "author": res["source_data"]["author"],
        "publish_date": res["source_data"]["publish_date"],
        "supporting_evidence": res["evidence_data"].get("supporting_evidence", []),
        "contradicting_evidence": res["evidence_data"].get("contradicting_evidence", []),
        "contextual_evidence": res["evidence_data"].get("contextual_evidence", []),
        "verification_points": structured_r.get("verification_points", {}),
        "reasoning": res["reasoning"]
    }

    try:
        pdf_bytes = generate_pdf_report(pdf_payload)
        st.download_button(
            label="📥 Download Official PDF Verification Report",
            data=pdf_bytes,
            file_name=f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"Could not generate PDF report: {e}")

# ─── Analytics Dashboard (Preserved & Enhanced) ───────────────────────────────
st.markdown("---")
st.markdown("## 📈 Platform Analytics Dashboard")

if len(st.session_state.history) > 0:
    df_history = pd.DataFrame(st.session_state.history)

    col_an1, col_an2, col_an3 = st.columns(3)

    with col_an1:
        st.subheader("ML Predictions")
        pred_counts = df_history["Result"].value_counts().to_dict()
        st.markdown(f"**Fake News:** {pred_counts.get('Fake News', 0)}  \n**Real News:** {pred_counts.get('Real News', 0)}")
        st.bar_chart(df_history["Result"].value_counts())

    with col_an2:
        st.subheader("Final Verdicts")
        verdict_counts = df_history["Verdict"].value_counts().to_dict()
        verdict_lines = []
        for v_name in ["VERIFIED", "MISLEADING", "LIKELY FAKE", "UNCERTAIN"]:
            if v_name in verdict_counts or len(verdict_counts) > 0:
                verdict_lines.append(f"**{v_name}:** {verdict_counts.get(v_name, 0)}")
        st.markdown("  \n".join(verdict_lines))
        st.bar_chart(df_history["Verdict"].value_counts())

    with col_an3:
        st.subheader("News Categories")
        cat_counts = df_history["Category"].value_counts().to_dict()
        cat_lines = [f"**{cat}:** {cnt}" for cat, cnt in cat_counts.items()]
        st.markdown("  \n".join(cat_lines))
        st.bar_chart(df_history["Category"].value_counts())
else:
    st.info("Run verifications on news articles or claims above to generate interactive platform analytics.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🌐 TruthLens AI — AI-Powered Global News Verification System | "
    "TF-IDF + Logistic Regression, Deterministic Proposition Entailment & Live Evidence Corroboration"
)