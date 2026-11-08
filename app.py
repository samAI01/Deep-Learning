
import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st

# =====================================================
# 1. STREAMLIT PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)

# =====================================================
# 2. PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "phishing_url_cnn.keras"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "metadata.json"
)

# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #003B46;
    color: white;
}

[data-testid="stHeader"] {
    background-color: transparent;
}

.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    color: #FFFFFF;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #D0E8EB;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

.safe {
    background-color: #145A32;
    color: white;
}

.danger {
    background-color: #922B21;
    color: white;
}

.warning {
    background-color: #7D6608;
    color: white;
}

.stButton > button {
    background-color: #00A6A6;
    color: white;
    border-radius: 8px;
    width: 100%;
    font-weight: bold;
    border: none;
    padding: 10px;
}

.stButton > button:hover {
    background-color: #008080;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 4. APPLICATION HEADER
# =====================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ AI-Powered Phishing URL Detector'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Based Malicious URL Detection '
    'Using Character-Level CNN'
    '</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

# =====================================================
# 5. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_phishing_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


@st.cache_data
def load_metadata():

    if os.path.exists(METADATA_PATH):

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    return {}


try:

    model = load_phishing_model()
    metadata = load_metadata()

    st.success("CNN Model loaded successfully!")

except Exception as e:

    st.error("Model loading failed.")

    st.exception(e)

    st.stop()


# =====================================================
# 6. URL INPUT
# =====================================================

st.subheader("🔍 Enter Website URL")

st.write(
    "Enter a URL below to check whether it resembles "
    "a phishing or legitimate URL."
)

url = st.text_input(
    "Website URL",
    placeholder="https://www.example.com"
)

# =====================================================
# 7. EXAMPLE URLS
# =====================================================

st.write("**Try example URLs:**")

col1, col2 = st.columns(2)

with col1:

    if st.button("Legitimate Example"):

        st.session_state["selected_url"] = (
            "https://www.wikipedia.org/"
        )

with col2:

    if st.button("Phishing Example"):

        st.session_state["selected_url"] = (
            "http://secure-login-account.example.com/verify"
        )

if "selected_url" in st.session_state:

    url = st.session_state["selected_url"]

    st.session_state.pop("selected_url")


# =====================================================
# 8. PREDICTION FUNCTION
# =====================================================

def predict_url(candidate):

    input_data = np.array(
        [candidate],
        dtype=str
    )

    prediction = model.predict(
        input_data,
        verbose=0
    )

    phishing_probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    return phishing_probability


# =====================================================
# 9. DETECT PHISHING URL
# =====================================================

if st.button("🚀 Analyze URL"):

    candidate = url.strip()

    if not candidate:

        st.warning("Please enter a website URL.")

    elif len(candidate) > 2000:

        st.warning(
            "URL is too long. Please enter a URL "
            "with fewer than 2001 characters."
        )

    elif len(candidate) == 0:

        st.warning("Invalid URL.")

    else:

        with st.spinner(
            "Analyzing URL using Deep Learning CNN..."
        ):

            try:

                phishing_probability = predict_url(
                    candidate
                )

                threshold = float(
                    metadata.get(
                        "classification_threshold",
                        0.50
                    )
                )

                # Model output:
                # 0 = Legitimate
                # 1 = Phishing

                if phishing_probability >= threshold:

                    st.markdown(
                        """
                        <div class="result-box danger">
                            <h2>🚨 PHISHING URL DETECTED</h2>
                            <h3>This URL resembles a phishing URL.</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="result-box safe">
                            <h2>✅ LEGITIMATE URL PREDICTED</h2>
                            <h3>This URL resembles a legitimate URL.</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ---------------------------------
                # DISPLAY PREDICTION PROBABILITIES
                # ---------------------------------

                st.write("")

                st.subheader("📊 Prediction Results")

                col1, col2 = st.columns(2)

                legitimate_probability = (
                    1 - phishing_probability
                )

                with col1:

                    st.metric(
                        "Legitimate Probability",
                        f"{legitimate_probability * 100:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Phishing Probability",
                        f"{phishing_probability * 100:.2f}%"
                    )

                st.write("### Phishing Probability")

                st.progress(
                    min(
                        max(phishing_probability, 0.0),
                        1.0
                    )
                )

                st.write(
                    f"Classification Threshold: "
                    f"{threshold * 100:.0f}%"
                )

                st.caption(
                    "Prediction is based on the URL's "
                    "character patterns learned by the CNN."
                )

            except Exception as e:

                st.error(
                    "An error occurred during prediction."
                )

                st.exception(e)


# =====================================================
# 10. MODEL INFORMATION
# =====================================================

st.divider()

st.subheader("🧠 Model Information")

st.write("**Deep Learning Architecture:** Character-Level CNN")

st.write("**Dataset:** UCI PhiUSIIL Phishing URL Dataset")

st.write("**Dataset ID:** 967")

st.write("**Input:** Raw URL strings")

st.write("**Output:** Probability of phishing")

st.write("**Classification Threshold:** 0.50 (default)")

# =====================================================
# 11. SAFETY DISCLAIMER
# =====================================================

st.divider()

st.subheader("⚠️ Security Disclaimer")

st.warning(
    """
    This application is an educational deep learning demonstration.

    - It analyzes URL text patterns only.
    - It does not visit or verify the website.
    - A legitimate prediction does not guarantee that a
      website is safe.
    - Do not open suspicious URLs just to test them.

    Always use additional security checks before trusting
    an unfamiliar website.
    """
)

st.caption(
    "AI-Powered Malicious URL and Phishing Detection | "
    "Character-Level CNN"
)
