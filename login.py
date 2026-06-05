import streamlit as st

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection Login",
    page_icon="📰",
    layout="centered",
)

# ─── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>📰 Fake News Detection</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h3 style='text-align:center;'>Secure Login Portal</h3>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ─── Login Form ──────────────────────────────────────────
st.subheader("🔐 Sign In")

username = st.text_input(
    "Email Address",
    placeholder="Enter your email"
)

password = st.text_input(
    "Password",
    type="password",
    placeholder="Enter your password"
)

st.checkbox("Remember Me")

# Login Button
if st.button("🚀 Login"):

    if username == "vivek@gmail.com" and password == "1234":

        st.success("✅ Login Successful")
        st.balloons()

        st.info("🚀 Fake News Detector App Ready")

        st.markdown(
            """
            ### Open Main Application

            👉 Run app.py in another terminal:

            ```bash
            python -m streamlit run app.py
            ```

            Then open:

            http://localhost:8501
            """
        )

    else:
        st.error("❌ Invalid Email or Password")

st.markdown("")

# Google Login Button
if st.button("🔵 Continue with Google"):
    st.info("Google Login will be connected later.")

st.markdown("---")

# Register & Forgot Password
col1, col2 = st.columns(2)

with col1:
    st.button("📝 Register")

with col2:
    st.button("🔑 Forgot Password")

# Footer
st.caption("© 2026 Fake News Detection System")