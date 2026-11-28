```python
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
    layout="wide"
)


# =====================================================
# 2. PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "phishing_url_cnn.keras"
)

METADATA_PATH = os.path.join(
    ARTIFACT_DIR,
    "metadata.json"
)


# =====================================================
# 3. PAGE STYLING
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #003B46;
        color: white;
    }

    h1, h2, h3 {
        color: #FFFFFF;
    }

    .main-title {
        text-align: center;
        color: #FFFFFF;
        font-size: 36px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #D0E8E8;
        font-size: 17px;
    }

    .result-card {
        background-color: #07515B;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #16818A;
        margin-top: 15px;
    }

    .result-text {
        color: white;
        font-size: 23px;
        font-weight: bold;
    }

    div.stButton > button {
        background: linear-gradient(
            90deg,
            #00C9A7,
            #00A896
        );
        color: white;
        font-weight: bold;
        border-radius: 9px;
        border: none;
        padding: 10px 20px;
    }

    div.stButton > button:hover {
        background: #02B89D;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 4. APPLICATION TITLE
# =====================================================

st.markdown(
    '<p class="main-title">🛡️ AI-Powered Phishing URL Detector</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Character-Level CNN for Malicious URL and Phishing Detection'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# 5. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource(show_spinner="Loading trained CNN model...")
def load_phishing_model(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    trained_model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return trained_model


# =====================================================
# 6. LOAD MODEL METADATA
# =====================================================

@st.cache_data(show_spinner=False)
def load_metadata(metadata_path):

    if not os.path.exists(metadata_path):
        return {}

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =====================================================
# 7. MODEL INITIALIZATION
# =====================================================

try:

    model = load_phishing_model(MODEL_PATH)

    metadata = load_metadata(METADATA_PATH)

    THRESHOLD = float(
        metadata.get(
            "classification_threshold",
            0.50
        )
    )

    st.success("CNN model loaded successfully!")

except Exception as error:

    st.error("Unable to load the trained model.")

    st.exception(error)

    st.info(
        "Check that phishing_url_cnn.keras exists "
        "inside the artifacts folder."
    )

    st.stop()


# =====================================================
# 8. URL PREDICTION FUNCTION
# =====================================================

def predict_url(input_url, trained_model):

    input_url = str(input_url).strip()

    if not input_url:
        raise ValueError("Please enter a URL.")

    prediction = trained_model.predict(
        tf.constant(
            [input_url],
            dtype=tf.string
        ),
        verbose=0
    )

    phishing_probability = float(
        prediction[0][0]
    )

    if not np.isfinite(phishing_probability):
        raise ValueError(
            "The model returned an invalid prediction."
        )

    phishing_probability = float(
        np.clip(phishing_probability, 0.0, 1.0)
    )

    legitimate_probability = (
        1.0 - phishing_probability
    )

    if phishing_probability >= THRESHOLD:

        predicted_output = "Potentially Phishing"

    else:

        predicted_output = "Likely Legitimate"

    return (
        predicted_output,
        phishing_probability,
        legitimate_probability
    )


# =====================================================
# 9. SESSION STATE
# =====================================================

if "results" not in st.session_state:

    st.session_state.results = []


# =====================================================
# 10. URL INPUT SECTION
# =====================================================

st.subheader("🔍 Check a Website URL")

st.write(
    "Enter a URL below to estimate whether it contains "
    "patterns associated with phishing."
)

example_urls = {
    "Select an example": "",
    "Google": "https://www.google.com",
    "Wikipedia": "https://www.wikipedia.org",
    "Example phishing URL 1":
        "http://secure-login-account.example.com/verify",
    "Example phishing URL 2":
        "http://paypal-login.example.com"
}

selected_example = st.selectbox(
    "Try an example URL",
    list(example_urls.keys())
)

default_url = example_urls[selected_example]

url_input = st.text_input(
    "Enter URL",
    value=default_url,
    placeholder="https://example.com"
)


# =====================================================
# 11. PREDICTION BUTTON
# =====================================================

if st.button(
    "🔍 Detect Phishing",
    use_container_width=True
):

    if not url_input.strip():

        st.warning("Please enter a URL.")

    else:

        try:

            with st.spinner("Analyzing URL..."):

                (
                    predicted_output,
                    phishing_probability,
                    legitimate_probability
                ) = predict_url(
                    url_input,
                    model
                )

            st.session_state.results.append({

                "Input URL": url_input,

                "Predicted Output": predicted_output,

                "Phishing Probability":
                    round(phishing_probability * 100, 2),

                "Legitimate Probability":
                    round(legitimate_probability * 100, 2)

            })

            st.session_state.latest_result = {

                "url": url_input,

                "output": predicted_output,

                "phishing": phishing_probability,

                "legitimate": legitimate_probability

            }

        except Exception as error:

            st.error("Prediction failed.")

            st.exception(error)


# =====================================================
# 12. DISPLAY PREDICTION RESULT
# =====================================================

if "latest_result" in st.session_state:

    result = st.session_state.latest_result

    st.divider()

    st.subheader("📊 Detection Results")

    st.write("**Analyzed URL:**", result["url"])

    if result["phishing"] >= THRESHOLD:

        st.error(
            "⚠️ Potentially Phishing URL"
        )

    else:

        st.success(
            "✅ Likely Legitimate URL"
        )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Phishing Probability",
            f"{result['phishing'] * 100:.2f}%"
        )

        st.progress(
            float(result["phishing"])
        )

    with col2:

        st.metric(
            "Legitimate Probability",
            f"{result['legitimate'] * 100:.2f}%"
        )

        st.progress(
            float(result["legitimate"])
        )

    st.markdown(
        f"""
        <div class="result-card">
            <p class="result-text">
                Prediction: {result['output']}
            </p>
            <p>
                Classification threshold:
                {THRESHOLD:.2f}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# 13. PREDICTION HISTORY
# =====================================================

st.divider()

st.subheader("📜 Prediction History")

if st.session_state.results:

    history_df = pd.DataFrame(
        st.session_state.results
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="phishing_detection_results.csv",
        mime="text/csv"
    )

    if st.button("Clear Prediction History"):

        st.session_state.results = []

        st.session_state.pop(
            "latest_result",
            None
        )

        st.rerun()

else:

    st.info("No URLs have been analyzed yet.")


# =====================================================
# 14. MODEL INFORMATION
# =====================================================

st.divider()

st.subheader("🧠 Model Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        metadata.get(
            "model_type",
            "Character-Level CNN"
        )
    )

with col2:

    st.metric(
        "Maximum URL Length",
        metadata.get(
            "max_length",
            200
        )
    )

with col3:

    st.metric(
        "Classification Threshold",
        f"{THRESHOLD:.2f}"
    )

with st.expander("View Model Details"):

    st.write(
        "**Model Name:**",
        metadata.get(
            "model_name",
            "AI-Powered Malicious URL Detection"
        )
    )

    st.write(
        "**Dataset:**",
        metadata.get(
            "dataset",
            "UCI PhiUSIIL Phishing URL Dataset"
        )
    )

    st.write(
        "**Dataset ID:**",
        metadata.get(
            "dataset_id",
            967
        )
    )

    st.write(
        "**Input:** Raw URL strings"
    )

    st.write(
        "**Output:** Probability of phishing"
    )

    st.write(
        "**Embedding Dimension:**",
        metadata.get("embedding_dimension", 64)
    )

    st.write(
        "**Convolution Filters:**",
        metadata.get("convolution_filters", 128)
    )

    st.write(
        "**Kernel Size:**",
        metadata.get("kernel_size", 5)
    )

    evaluation = metadata.get("evaluation", {})

    if evaluation:

        st.write("### Training Evaluation")

        metrics_df = pd.DataFrame(
            [
                {
                    "Metric": key.replace("_", " ").title(),
                    "Value": round(value, 4)
                }
                for key, value in evaluation.items()
            ]
        )

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            metadata.get(
                "evaluation_note",
                "Evaluation results are from the training script."
            )
        )


# =====================================================
# 15. DISCLAIMER
# =====================================================

st.divider()

st.subheader("⚠️ Disclaimer")

st.warning(
    """
    This application is an educational machine-learning demonstration.

    The model analyzes URL character patterns and estimates phishing
    probability. It does not visit websites, inspect website content,
    verify website ownership, or guarantee that a URL is safe.

    A URL classified as likely legitimate may still be malicious.
    Always use additional security checks before opening unfamiliar links.
    """
)


# =====================================================
# 16. FOOTER
# =====================================================

st.markdown(
    """
    <p style="text-align:center; color:#D0E8E8;">
        AI-Powered Malicious URL and Phishing Detection
        <br>
        Character-Level Convolutional Neural Network
    </p>
    """,
    unsafe_allow_html=True
)
```