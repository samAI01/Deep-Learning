
import pandas as pd
import os
import tensorflow as tf
import streamlit as st

# Streamlit page configuration
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)

# CUSTOM CSS DESIGN

st.markdown("""
<style>

.stApp {
    background-color: #003B46;
    color: #ffffff;
}

[data-testid="stHeader"] {
    background-color: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1000px;
}

.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 16px;
    color: #cbd5e1;
    margin-bottom: 25px;
}

.section-title {
    color: #38bdf8;
    font-size: 23px;
    font-weight: 700;
}

div.stButton > button {
    background: linear-gradient(90deg, #0284c7, #2563eb);
    color: WHITE;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: bold;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #0369a1, #1d4ed8);
    color: WHITE;
    border: none;
}

div[data-testid="stMetric"] {
    background-color: #172338;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263750;
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1;
}

div[data-testid="stMetricValue"] {
    color: #38bdf8;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263750;
    border-radius: 10px;
}

input {
    border-radius: 8px !important;
}

.result-card {
    background-color: #172338;
    border: 1px solid #263750;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin: 15px 0;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    padding-top: 25px;
}

/* URL Input Box */
div[data-testid="stTextInput"] input {
    background-color: #172338 !important;
    color: #38bdf8 !important;
    border: 2px solid #0ea5e9 !important;
    border-radius: 10px !important;
    font-size: 16px !important;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #94a3b8 !important;
}

div[data-testid="stTextInput"] label {
    color: #38bdf8 !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

#model path
MODEL_PATH = os.path.join(
    os.getcwd(),
    "artifacts",
    "phishing_url_cnn.keras"
)

# Load trained CNN model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("CNN model loaded successfully!")

        return model
    except Exception:
    
        import traceback
    
        print("MODEL LOADING ERROR:")
        traceback.print_exc()
    
        return None


# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = []


# Application title
st.title("🛡️ AI-Powered Phishing URL Detector")

st.write(
    "Enter a URL to estimate whether its text pattern "
    "resembles phishing URLs learned by the model."
)

st.warning(
    "Educational demonstration only. This model does not visit "
    "or verify websites and cannot guarantee that a URL is safe. "
    "Do not open suspicious links just to test them."
)


# Load model
model = load_model()

if model is None:
    st.error(
        "Model file not found. Train the model first and place "
        "phishing_url_cnn.keras in the artifacts/ folder."
    )
    st.stop()


# Example URLs
examples = [
    "https://www.wikipedia.org/",
    "http://secure-login-account.example.com/verify",
]

example = st.selectbox(
    "Choose an example or enter your own below",
    ["(custom URL)"] + examples
)

default_url = "" if example == "(custom URL)" else example

url = st.text_input(
    "🌐 URL",
    value=default_url,
    placeholder="https://example.com/login"
)


# Scan button
if st.button(
    "🛡️ Scan URL",
    type="primary",
    use_container_width=True
):

    candidate = url.strip()

    if not candidate:
        st.error("Please enter a URL.")

    elif len(candidate) > 2000:
        st.error(
            "URL is too long for this demo. "
            "Please enter a URL under 2,000 characters."
        )

    else:
        try:

            # Convert URL into TensorFlow string tensor
            url_tensor = tf.constant(
                [candidate],
                dtype=tf.string
            )

            # Get prediction
            prediction = model(
                url_tensor,
                training=False
            )

            # Extract legitimate probability
            legitimate_probability = float(
                tf.reshape(prediction, [-1])[0].numpy()
            )

            # Calculate phishing probability
            phishing_probability = (
                1.0 - legitimate_probability
            )

            # Display prediction
            if phishing_probability >= 0.5:

                st.error("Prediction: Potentially phishing")

                predicted_output = "Potentially phishing"

            else:

                st.success("Prediction: Likely legitimate")

                predicted_output = "Likely legitimate"

            # Display probability
            st.metric(
                "Model-estimated phishing probability",
                f"{phishing_probability:.1%}"
            )

            st.progress(phishing_probability)

            st.caption(
                "Threshold = 50%. This score is not a verified "
                "real-world probability or a safety guarantee."
            )

            # Store results
            st.session_state.results.append({

                "Input URL": candidate,

                "Predicted Output": predicted_output,

                "Phishing Probability": (
                    f"{phishing_probability:.2%}"
                )

            })

        except Exception as e:

            st.error(f"Prediction error: {e}")


# DISPLAY RESULTS HISTORY
# --------------------------------------------------

if st.session_state.results:

    st.divider()

    st.subheader("📊 Phishing URL Detection Results")

    # Convert stored results into DataFrame
    df = pd.DataFrame(st.session_state.results)
    
    def color_URL(value):
            return "color: #3b82f6; font-weight;"
    # Function to color Predicted Output
    def color_prediction(value):
        if value == "Likely legitimate":
            return "color: #22c55e; font-weight: bold;"
        elif value == "Potentially phishing":
            return "color: #ef4444; font-weight: bold;"
        return "color: white;"

    # Function to color Phishing Probability
    def color_probability(value):
        probability = float(value.replace("%", ""))
    
        if probability >= 50:
            return "color: #ef4444; font-weight: bold;"
        else:
            return "color: #22c55e; font-weight: bold;"
    # Style DataFrame
    styled_df = (df.style.set_properties(
        **{
            "background-color": "#172338",
            "color": "#22c55e",
            "border-color": "#334155",
            "text-align": "center"
        }
    ).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#0369a1"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("text-align", "center")
            ]
        },
        {
            "selector": "td",
            "props": [
                ("padding", "12px")
            ]
        }
    ])
    .map(color_URL, subset=["Input URL"])             
    .map(color_prediction, subset=["Predicted Output"])
    .map(color_probability, subset=["Phishing Probability"])
)
    # Display styled DataFrame
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    # Count phishing and legitimate URLs
    phishing_count = sum(
        item["Predicted Output"] == "Potentially phishing"
        for item in st.session_state.results
    )

    legitimate_count = sum(
        item["Predicted Output"] == "Likely legitimate"
        for item in st.session_state.results
    )

    # Display counts
    col1, col2 = st.columns(2)

    col1.metric(
        "Phishing URLs",
        phishing_count
    )

    col2.metric(
        "Legitimate URLs",
        legitimate_count
    )

    # How it works
    st.divider()

    st.subheader("How it works")

    st.markdown(
        "1. The URL is treated as a character sequence.  \n"
        "2. A character-level CNN estimates a score.  \n"
        "3. The application displays a class using a 0.5 threshold.  \n"
        "4. The application never opens the submitted URL."
    )

    st.caption(
        "Dataset: UCI PhiUSIIL. "
        "See README for training instructions and limitations."
    )

    # Clear results button
    if st.button("Clear Results"):

        st.session_state.results = []

        st.rerun()
