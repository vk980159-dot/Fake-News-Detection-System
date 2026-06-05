import streamlit as st
import pickle
import re
import pandas as pd
from reportlab.pdfgen import canvas
import io
from newspaper import Article


# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# ─── Load Saved Model ─────────────────────────────────────────────────────────
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ─── Prediction History ───────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "fetched_news" not in st.session_state:
    st.session_state.fetched_news = ""

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Project Information")

st.sidebar.info("""
Algorithm: Logistic Regression

Vectorization: TF-IDF

Dataset Size: 44,898 Articles

Accuracy: 98.63%
""")

st.sidebar.metric(
    "📊 Total Predictions",
    len(st.session_state.history)
)

st.sidebar.subheader("📜 Prediction History")

for item in st.session_state.history[-5:]:
    st.sidebar.write(f"{item['Category']} → {item['Result']}")

if st.sidebar.button("🗑 Clear History"):
    st.session_state.history = []
    st.rerun()


# ─── Helper Functions ─────────────────────────────────────────────────────────

def clean_text(text):
    """Clean and preprocess input text."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def detect_category(text):
    """Detect news category based on keywords."""
    text = text.lower()

    categories = {
        "Politics": ["government", "minister", "election", "parliament", "president", "policy"],
        "Sports": ["cricket", "football", "match", "player", "tournament", "olympics"],
        "Technology": ["ai", "artificial intelligence", "software", "technology", "computer", "google"],
        "Business": ["market", "stock", "economy", "bank", "investment", "finance"],
        "Entertainment": ["movie", "actor", "actress", "film", "music", "celebrity"]
    }

    for category, keywords in categories.items():
        for word in keywords:
            if word in text:
                return category

    return "General News"


# ─── Main Title ───────────────────────────────────────────────────────────────
st.markdown("""
# 📰 Fake News Detection System

### AI Powered News Verification Platform

Check whether a news article is **Fake** or **Real** using Machine Learning.
""")

st.info(
    "⚠️ This model is trained on news articles only. "
    "For best results, enter a complete news article."
)

# ─── URL Analyzer ─────────────────────────────────────────────────────────────
st.subheader("🔗 Analyze News from URL")

news_url = st.text_input(
    "Paste News URL",
    placeholder="https://example.com/news"
)

if st.button("🌐 Fetch News from URL"):
    if news_url.strip():
        try:
            article = Article(news_url)
            article.download()
            article.parse()
            st.session_state.fetched_news = article.text
            st.success("✅ News article fetched successfully")
        except Exception:
            st.error("❌ Unable to fetch article from URL")

# ─── Input Area ───────────────────────────────────────────────────────────────
news = st.text_area(
    "📝 Paste News Article Here",
    value=st.session_state.fetched_news,
    height=250,
    placeholder="Paste complete news article here..."
)

# ─── Prediction ───────────────────────────────────────────────────────────────
if st.button("🔍 Analyze News", use_container_width=True):

    # Validation: empty input
    if news.strip() == "":
        st.warning("⚠️ Please enter news text first.")
        st.stop()

    # Validation: minimum words
    if len(news.split()) < 20:
        st.warning("⚠️ Please enter a complete news article (minimum 20 words).")
        st.stop()

    # Preprocessing & Prediction
    cleaned_news = clean_text(news)
    category = detect_category(news)
    news_vector = vectorizer.transform([cleaned_news])
    prediction = model.predict(news_vector)[0]
    probability = model.predict_proba(news_vector)
    confidence = max(probability[0]) * 100

    # Save to history
    st.session_state.history.append({
        "Category": category,
        "Result": "Fake" if prediction == 0 else "Real"
    })

    # ── Result ────────────────────────────────────────────────────────────────
    st.markdown("## 📢 Prediction Result")
    st.info(f"📰 Category: {category}")

    if prediction == 0:
        st.error("🚨 Fake News Detected")
        st.subheader("🤖 Why this prediction?")
        st.write("""
        • Sensational or misleading wording detected

        • Similar patterns found in fake news dataset

        • Unverified reporting style

        • Language differs from trusted news articles
        """)
    else:
        st.success("✅ Real News Detected")
        st.subheader("🤖 Why this prediction?")
        st.write("""
        • Formal news reporting style

        • Similar to trusted news articles

        • Consistent language patterns

        • Matches characteristics of verified news content
        """)

    # ── Confidence Score ──────────────────────────────────────────────────────
    st.subheader("📊 Confidence Score")
    st.progress(confidence / 100)
    st.success(f"{confidence:.2f}% Confidence")
 # Confidence Status
if confidence >= 90:
    st.success("🟢 High Confidence Prediction")

elif confidence >= 70:
    st.info("🟡 Medium Confidence Prediction")

else:
    st.warning(
        "⚠️ Low Confidence Prediction. Please verify from trusted news sources."
    )
    # ── PDF Report Download ───────────────────────────────────────────────────
    pdf_buffer = io.BytesIO()
    p = canvas.Canvas(pdf_buffer)

    p.drawString(100, 800, "Fake News Detection Report")
    p.drawString(100, 770, f"Category: {category}")
    p.drawString(100, 740, f"Prediction: {'Fake News' if prediction == 0 else 'Real News'}")
    p.drawString(100, 710, f"Confidence: {confidence:.2f}%")
    p.save()

    pdf_buffer.seek(0)

    st.download_button(
        label="📄 Download Report",
        data=pdf_buffer.getvalue(),
        file_name="fake_news_report.pdf",
        mime="application/pdf"
    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Developed by Vivek Kumar | Python • Machine Learning • Streamlit")

# ─── Analytics Dashboard ──────────────────────────────────────────────────────
st.markdown("## 📊 Analytics Dashboard")

if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)

    st.subheader("📈 Prediction Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Fake vs Real News")
        result_counts = df["Result"].value_counts()
        st.bar_chart(result_counts)

    with col2:
        st.write("### News Categories")
        category_counts = df["Category"].value_counts()
        st.bar_chart(category_counts)

else:
    st.info("Analyze some news articles to view dashboard analytics.")