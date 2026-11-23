
import os
import json
import numpy as np
import pandas as pd
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
    BASE_DIR, "artifacts", "phishing_url_cnn.keras"
)

METADATA_PATH = os.path.join(
    BASE_DIR, "artifacts", "metadata.json"
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
# 6. INITIALIZE SESSION STATE
# =====================================================

if "url_input" not in st.session_state:
    st.session_state["url_input"] = ""

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []


# =====================================================
# 7. EXAMPLE URL CALLBACK FUNCTION
# =====================================================

def set_example_url(example):

    st.session_state["url_input"] = example


# =====================================================
# 8. URL INPUT AND EXAMPLE BUTTONS
# =====================================================

st.subheader("🔍 Enter Website URL")

st.write(
    "Enter a URL below to check whether it resembles "
    "a phishing or legitimate URL."
)

st.write("**Try example URLs:**")

col1, col2 = st.columns(2)

with col1:

    st.button(
        "✅ Legitimate Example",
        on_click=set_example_url,
        args=("https://www.wikipedia.org/",),
        use_container_width=True
    )

with col2:

    st.button(
        "🚨 Phishing Example",
        on_click=set_example_url,
        args=("http://secure-login-account.example.com/verify",),
        use_container_width=True
    )


# =====================================================
# 9. WEBSITE URL INPUT FIELD
# =====================================================

url = st.text_input(
    "Website URL",
    placeholder="https://www.example.com",
    key="url_input"
)


# =====================================================
# 10. PREDICTION FUNCTION
# =====================================================

def predict_url(url, model):

    """
    Predict whether a URL is legitimate or phishing.

    Training labels:
    1 = Legitimate
    0 = Phishing

    Model output:
    Probability of legitimate class.
    """

    url = str(url).strip()

    if not url:
        raise ValueError("Please enter a valid URL.")

    # Convert URL into TensorFlow string tensor
    url_tensor = tf.constant(
        [url],
        dtype=tf.string
    )

    # CNN prediction
    prediction = model(
        url_tensor,
        training=False
    )

    prediction = prediction.numpy()

    if prediction.size != 1:
        raise ValueError(
            "Unexpected model output shape. "
            "Expected one probability per URL."
        )

    # Model output represents legitimate probability
    legitimate_probability = float(
        prediction.reshape(-1)[0]
    )

    if not np.isfinite(legitimate_probability):
        raise ValueError(
            "Model returned an invalid probability."
        )

    legitimate_probability = min(
        max(legitimate_probability, 0.0),
        1.0
    )

    # Calculate phishing probability
    phishing_probability = (
        1.0 - legitimate_probability
    )

    return phishing_probability


# =====================================================
# 11. ANALYZE URL BUTTON
# =====================================================

if st.button(
    "🚀 Analyze URL",
    use_container_width=True
):

    candidate = url.strip()

    if not candidate:

        st.warning("Please enter a website URL.")

    elif len(candidate) > 2000:

        st.warning(
            "URL is too long. Please enter a URL "
            "with fewer than 2001 characters."
        )

    else:

        with st.spinner(
            "Analyzing URL using Deep Learning CNN..."
        ):

            try:

                # ---------------------------------
                # GET MODEL PREDICTION
                # ---------------------------------

                phishing_probability = predict_url(
                    candidate,
                    model
                )

                # ---------------------------------
                # CLASSIFICATION THRESHOLD
                # ---------------------------------

                threshold = float(
                    metadata.get(
                        "classification_threshold",
                        0.50
                    )
                )

                if (
                    not np.isfinite(threshold)
                    or not 0.0 <= threshold <= 1.0
                ):
                    raise ValueError(
                        "Invalid classification threshold."
                    )

                # ---------------------------------
                # CALCULATE PROBABILITIES
                # ---------------------------------

                legitimate_probability = (
                    1.0 - phishing_probability
                )

                # ---------------------------------
                # CLASSIFICATION
                # ---------------------------------

                if phishing_probability >= threshold:

                    predicted_output = "PHISHING"

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

                    predicted_output = "LEGITIMATE"

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
                # SAVE PREDICTION RESULT
                # ---------------------------------

                result_record = {
                    "Input URL": candidate,

                    "Predicted Output": predicted_output,

                    "Legitimate Probability": round(
                        legitimate_probability * 100, 2
                    ),

                    "Phishing Probability": round(
                        phishing_probability * 100, 2
                    )
                }

                # Add result to session history
                st.session_state[
                    "prediction_history"
                ].append(result_record)

                # ---------------------------------
                # DISPLAY PREDICTION PROBABILITIES
                # ---------------------------------

                st.write("")

                st.subheader("📊 Prediction Results")

                col1, col2 = st.columns(2)

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

                # ---------------------------------
                # PROBABILITY PROGRESS BAR
                # ---------------------------------

                st.write("### Phishing Probability")

                st.progress(
                    phishing_probability
                )

                st.write(
                    f"Classification Threshold: "
                    f"{threshold * 100:.0f}%"
                )

                st.caption(
                    "Prediction is based on the URL's "
                    "character patterns learned by the CNN."
                )

                st.write("**Analyzed URL:**")

                st.code(candidate)

            except Exception as e:

                st.error(
                    "An error occurred during prediction."
                )

                st.exception(e)


# =====================================================
# 12. OUTPUT / DETECTION HISTORY
# =====================================================

st.divider()

st.subheader("📋 URL Detection Output")

st.write(
    "View the prediction results of all URLs "
    "analyzed during this session."
)

if st.session_state["prediction_history"]:

    results_df = pd.DataFrame(
        st.session_state["prediction_history"]
    )

    # ---------------------------------
    # DISPLAY RESULT TABLE
    # ---------------------------------

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------
    # DISPLAY TOTAL PREDICTIONS
    # ---------------------------------

    total_predictions = len(results_df)

    phishing_count = (
        results_df["Predicted Output"] == "PHISHING"
    ).sum()

    legitimate_count = (
        results_df["Predicted Output"] == "LEGITIMATE"
    ).sum()

    st.write("### 📈 Detection Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total URLs",
            total_predictions
        )

    with col2:

        st.metric(
            "Legitimate URLs",
            int(legitimate_count)
        )

    with col3:

        st.metric(
            "Phishing URLs",
            int(phishing_count)
        )

    # ---------------------------------
    # DOWNLOAD RESULTS AS CSV
    # ---------------------------------

    csv_data = results_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Detection Results",
        data=csv_data,
        file_name="phishing_detection_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ---------------------------------
    # CLEAR DETECTION HISTORY
    # ---------------------------------

    if st.button(
        "🗑️ Clear Detection History",
        use_container_width=True
    ):

        st.session_state["prediction_history"] = []

        st.rerun()

else:

    st.info(
        "No URLs analyzed yet. Enter a URL and click "
        "'Analyze URL' to display prediction results."
    )


# =====================================================
# 13. MODEL INFORMATION
# =====================================================

st.divider()

st.subheader("🧠 Model Information")

st.write(
    "**Deep Learning Architecture:** Character-Level CNN"
)

st.write(
    "**Dataset:** UCI PhiUSIIL Phishing URL Dataset"
)

st.write("**Dataset ID:** 967")

st.write("**Input:** Raw URL strings")

st.write("**Output:** Probability of phishing")

st.write(
    "**Classification Threshold:** "
    f"{metadata.get('classification_threshold', 0.50):.2f}"
)


# =====================================================
# 14. SECURITY DISCLAIMER
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
