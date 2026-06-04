import streamlit as st
import pickle
import re

# Page Configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# Load Saved Model
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Sidebar
st.sidebar.title("📊 Project Information")

st.sidebar.info("""
Algorithm: Logistic Regression

Vectorization: TF-IDF

Dataset Size: 44,898 Articles

Accuracy: 98.63%
""")

# Text Cleaning Function
def clean_text(text):
    text = str(text)
    text = text.lower()

    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text

# Main Title
st.markdown("""
# 📰 Fake News Detection System

### AI Powered News Verification Platform

Check whether a news article is Fake or Real using Machine Learning.
""")

# Information Box
st.info(
    "⚠️ This model is trained on news articles only. "
    "For best results, enter a complete news article."
)

# Input Area
news = st.text_area(
    "📝 Paste News Article Here",
    height=250,
    placeholder="Paste complete news article here..."
)

# Prediction Button
if st.button("🔍 Analyze News", use_container_width=True):

    # Empty Input Validation
    if news.strip() == "":
        st.warning("⚠️ Please enter news text first.")
        st.stop()

    # Minimum Word Validation
    if len(news.split()) < 20:
        st.warning(
            "⚠️ Please enter a complete news article (minimum 20 words)."
        )
        st.stop()

    cleaned_news = clean_text(news)

    news_vector = vectorizer.transform([cleaned_news])

    prediction = model.predict(news_vector)[0]

    probability = model.predict_proba(news_vector)

    confidence = max(probability[0]) * 100

    # Prediction Result
    st.markdown("## 📢 Prediction Result")

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

    # Confidence Score
    st.subheader("📊 Confidence Score")

    st.progress(confidence / 100)

    st.success(f"{confidence:.2f}% Confidence")

# Footer
st.markdown("---")

st.caption(
    "Developed by Vivek Kumar | Python • Machine Learning • Streamlit"
)