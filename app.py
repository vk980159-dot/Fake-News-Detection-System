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

# Input Area
news = st.text_area(
    "📝 Paste News Article Here",
    height=250,
    placeholder="Paste complete news article here..."
)

# Prediction Button
if st.button("🔍 Analyze News", use_container_width=True):

    if news.strip() == "":
        st.warning("⚠️ Please enter news text first.")

    else:
        cleaned_news = clean_text(news)

        news_vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(news_vector)

        probability = model.predict_proba(news_vector)

        if prediction[0] == 0:

            confidence = probability[0][0] * 100

            st.error("🚨 Fake News Detected")

            st.progress(int(confidence))

            st.write(f"Confidence: {confidence:.2f}%")

        else:

            confidence = probability[0][1] * 100

            st.success("✅ Real News Detected")

            st.progress(int(confidence))

            st.write(f"Confidence: {confidence:.2f}%")

# Footer
st.markdown("---")
st.caption("Developed using Python, Scikit-Learn and Streamlit")