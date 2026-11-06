
import os
import json
import tensorflow as tf
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)

# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

BASE_DIR = os.getcwd()

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

THRESHOLD = 0.50

# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_data
def load_metadata():

    if not os.path.exists(METADATA_PATH):
        return {}

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


try:

    model = load_model()
    metadata = load_metadata()

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# ============================================================
# 4. APPLICATION HEADER
# ============================================================

st.title("🛡️ AI-Powered Phishing URL Detector")

st.write(
    "Enter a URL to estimate whether its character patterns "
    "resemble phishing URLs."
)

st.success("Trained CNN model loaded successfully!")

# ============================================================
# 5. URL INPUT
# ============================================================

candidate = st.text_input(
    "Enter URL",
    placeholder="https://www.example.com"
)

# ============================================================
# 6. PREDICTION
# ============================================================

if st.button("Detect URL", type="primary"):

    candidate = candidate.strip()

    if not candidate:

        st.warning("Please enter a URL.")

    elif len(candidate) > 2000:

        st.warning("URL is too long.")

    else:

        try:

            with st.spinner("Analyzing URL..."):

                probability = model.predict(
                    [candidate],
                    verbose=0
                ).ravel()[0]

            phishing_probability = float(probability)

            if phishing_probability >= THRESHOLD:

                st.error("🚨 Phishing URL detected!")

            else:

                st.success("✅ Likely legitimate URL")

            st.metric(
                "Phishing Probability",
                f"{phishing_probability:.2%}"
            )

            st.progress(
                min(
                    max(phishing_probability, 0.0),
                    1.0
                )
            )

            st.caption(
                f"Classification threshold: {THRESHOLD:.0%}"
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# 7. MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write("Model: Character-Level CNN")

    st.write("Dataset: UCI PhiUSIIL")

    st.write("Input: Raw URL strings")

    st.write("Output: Probability of phishing")

    st.write(
        "Training epochs:",
        metadata.get("epochs_completed", "Not available")
    )

# ============================================================
# 8. SECURITY DISCLAIMER
# ============================================================

st.warning(
    "Educational demonstration only. This application "
    "analyzes URL text patterns and does not visit or verify "
    "websites. A legitimate prediction does not guarantee "
    "that a website is safe. Do not open suspicious links "
    "to test them."
)
