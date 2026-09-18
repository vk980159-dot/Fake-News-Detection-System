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

    supp_cnt = len(res["evidence_data"].get("supporting_evidence", []))
    cont_cnt = len(res["evidence_data"].get("contradicting_evidence", []))

    if verdict == "VERIFIED":
        if supp_cnt >= 2:
            verdict_desc = f"Available public evidence supports the central claim. {supp_cnt} supporting public sources were found and no contradictory source was identified."
        else:
            verdict_desc = "Available public evidence supports the central claim. 1 supporting source was found and no contradictory source was identified."
    elif verdict == "MISLEADING":
        verdict_desc = res["verdict_res"].get("decision_rationale") or "Elements of the claim conflict with facts, exaggerate details, or use sensational framing."
    elif verdict == "LIKELY FAKE":
        verdict_desc = res["verdict_res"].get("decision_rationale") or "Linguistic patterns strongly align with fabricated news and/or debunking public reports refute the claim."
    else: # UNCERTAIN
        verdict_desc = res["verdict_res"].get("decision_rationale") or "Evidence is insufficient or conflicting; unable to independently certify truth or falsehood."

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

    with st.expander("📋 Detailed Verification Points", expanded=False):
        for reason in res["reasoning"]:
            st.markdown(f"- {reason}")

    # ── FEATURE 4 & 8: Evidence Found ──
    st.markdown("### 🔍 Real Evidence Cross-Referencing")
    if not res["evidence_data"]["is_available"]:
        st.warning("⚠️ **External evidence verification unavailable.** No matching live public reports were identified. The verdict reflects stylistic ML patterns and source metadata without hallucinating external sources.")
    else:
        tab_supp, tab_cont, tab_ctx = st.tabs([
            f"✅ Supporting Evidence ({len(res['evidence_data']['supporting_evidence'])})",
            f"🚨 Contradicting Evidence ({len(res['evidence_data']['contradicting_evidence'])})",
            f"ℹ️ Contextual References ({len(res['evidence_data']['contextual_evidence'])})"
        ])

        with tab_supp:
            if res["evidence_data"]["supporting_evidence"]:
                for item in res["evidence_data"]["supporting_evidence"]:
                    dom_str = f" (`{item['domain']}`)" if item.get("domain") else ""
                    tier_badge = item.get("tier_badge") or (f"🏛️ {item['source_tier']}" if item.get("source_tier") else "")
                    tier_str = f" &nbsp;•&nbsp; `{tier_badge}`" if tier_badge else ""
                    st.markdown(f"**[{item['source']}]**{dom_str} [{item['title']}]({item['url']}){tier_str}")
                    prop_str = f" | **Proposition:** `{item.get('matched_proposition')}`" if item.get('matched_proposition') and item.get('matched_proposition') != "None" else ""
                    rel_str = f" | **Relevance:** `{int(item.get('semantic_relevance', 0)*100)}%`" if item.get('semantic_relevance') is not None else ""
                    st.markdown(f"**Classification:** `SUPPORTING`{rel_str}{prop_str} | **Date:** {item.get('pub_date', 'Recent')}")
                    st.caption(f"**Snippet:** {item.get('snippet', '')}")
                    if item.get("reason"):
                        st.caption(f"ℹ️ *Entailment logic:* {item.get('reason')}")
                    st.markdown("---")
            else:
                st.write("No direct corroborating reports found in public news feeds.")

        with tab_cont:
            if res["evidence_data"]["contradicting_evidence"]:
                for item in res["evidence_data"]["contradicting_evidence"]:
                    dom_str = f" (`{item['domain']}`)" if item.get("domain") else ""
                    tier_badge = item.get("tier_badge") or (f"🏛️ {item['source_tier']}" if item.get("source_tier") else "")
                    tier_str = f" &nbsp;•&nbsp; `{tier_badge}`" if tier_badge else ""
                    st.markdown(f"**[{item['source']}]**{dom_str} [{item['title']}]({item['url']}){tier_str}")
                    prop_str = f" | **Proposition:** `{item.get('matched_proposition')}`" if item.get('matched_proposition') and item.get('matched_proposition') != "None" else ""
                    rel_str = f" | **Relevance:** `{int(item.get('semantic_relevance', 0)*100)}%`" if item.get('semantic_relevance') is not None else ""
                    st.markdown(f"**Classification:** `CONTRADICTING`{rel_str}{prop_str} | **Date:** {item.get('pub_date', 'Recent')}")
                    if item.get("conflict_type"):
                        st.error(
                            f"🚨 **Conflict Type:** {item['conflict_type']}  \n"
                            f"• **Claim Attribute:** `{item.get('claim_attribute')}`  \n"
                            f"• **Evidence Attribute:** `{item.get('evidence_attribute')}`"
                        )
                    st.caption(f"**Snippet:** {item.get('snippet', '')}")
                    if item.get("reason"):
                        st.caption(f"ℹ️ *Refutation logic:* {item.get('reason')}")
                    st.markdown("---")
            else:
                st.write("No direct contradictory or debunking reports found.")

        with tab_ctx:
            if res["evidence_data"]["contextual_evidence"]:
                for item in res["evidence_data"]["contextual_evidence"]:
                    dom_str = f" (`{item['domain']}`)" if item.get("domain") else ""
                    tier_badge = item.get("tier_badge") or (f"🏛️ {item['source_tier']}" if item.get("source_tier") else "")
                    tier_str = f" &nbsp;•&nbsp; `{tier_badge}`" if tier_badge else ""
                    st.markdown(f"**[{item['source']}]**{dom_str} [{item['title']}]({item['url']}){tier_str}")
                    prop_str = f" | **Proposition:** `{item.get('matched_proposition')}`" if item.get('matched_proposition') and item.get('matched_proposition') != "None" else ""
                    rel_str = f" | **Relevance:** `{int(item.get('semantic_relevance', 0)*100)}%`" if item.get('semantic_relevance') is not None else ""
                    st.markdown(f"**Classification:** `CONTEXTUAL`{rel_str}{prop_str} | **Date:** {item.get('pub_date', 'Recent')}")
                    st.caption(f"**Snippet:** {item.get('snippet', '')}")
                    if item.get("reason"):
                        st.caption(f"ℹ️ *Classification reason:* {item.get('reason')}")
                    st.markdown("---")
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
                st.markdown("**Claim Propositions:**")
                for prop in debug_trace.get("claim_propositions", []):
                    st.markdown(f"- `{prop}`")
                
                st.markdown("---")
                st.markdown("**Retrieved Sources & Proposition Entailment:**")
                evals = debug_trace.get("evaluations", [])
                if evals:
                    for idx, ev in enumerate(evals, 1):
                        badge = "✅ SUPPORTING" if ev["classification"] == "SUPPORTING" else ("🚨 CONTRADICTING" if ev["classification"] == "CONTRADICTING" else "ℹ️ CONTEXTUAL")
                        tier_lbl = ev.get("tier_badge") or ev.get("source_tier") or "General"
                        st.markdown(f"**{idx}. [{ev.get('source')}]** (`{ev.get('domain')}`) — `{tier_lbl}` — **{badge}**")
                        st.markdown(f"- **Source Title/Claim:** {ev.get('source_claim')}")
                        st.markdown(f"- **Semantic Relevance:** `{int(ev.get('semantic_relevance', 0)*100)}%`")
                        st.markdown(f"- **Proposition Match:** `{ev.get('proposition_match')}`")
                        if ev.get("conflict_type"):
                            st.markdown(f"- 🚨 **Conflict Type:** `{ev.get('conflict_type')}` (Claim: `{ev.get('claim_attribute')}` vs Evidence: `{ev.get('evidence_attribute')}`)")
                        st.markdown(f"- **Entailment Reason:** {ev.get('reason')}")
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
        "evidence_status": res["evidence_status"],
        "domain": res["source_data"]["domain"],
        "author": res["source_data"]["author"],
        "publish_date": res["source_data"]["publish_date"],
        "supporting_evidence": res["evidence_data"].get("supporting_evidence", []),
        "contradicting_evidence": res["evidence_data"].get("contradicting_evidence", []),
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