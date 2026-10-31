import os
import tensorflow as tf
import streamlit as st

st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️", layout="centered")

MODEL_PATH = os.path.join("artifacts", "phishing_url_cnn.keras")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

st.title("🛡️ AI-Powered Phishing URL Detector")
st.write("Enter a URL to estimate whether its text pattern resembles phishing URLs learned by the model.")
st.warning(
    "Educational demonstration only. This model does not visit or verify websites and cannot guarantee that a URL is safe. "
    "Do not open suspicious links just to test them."
)

model = load_model()
if model is None:
    st.error("Model file not found. Train the model first and place phishing_url_cnn.keras in the artifacts/ folder.")
    st.stop()

examples = [
    "https://www.wikipedia.org/",
    "http://secure-login-account.example.com/verify",
]
example = st.selectbox("Choose an example or enter your own below", ["(custom URL)"] + examples)
default_url = "" if example == "(custom URL)" else example
url = st.text_input("URL", value=default_url, placeholder="https://example.com/login")

if st.button("Scan URL", type="primary", use_container_width=True):
    candidate = url.strip()
    if not candidate:
        st.error("Please enter a URL.")
    elif len(candidate) > 2000:
        st.error("URL is too long for this demo. Please enter a URL under 2,000 characters.")
    else:
        # Do not fetch/open the URL. Model prediction uses the text string only.
        probability = float(model.predict([candidate], verbose=0).ravel()[0])
        if probability >= 0.5:
            st.error("Prediction: Potentially phishing")
        else:
            st.success("Prediction: Likely legitimate")
        st.metric("Model-estimated phishing probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("Threshold = 50%. This score is not a verified real-world probability or a safety guarantee.")

st.divider()
st.subheader("How it works")
st.markdown(
    "1. The URL is treated as a character sequence.  \n"
    "2. A character-level convolutional neural network estimates a phishing score.  \n"
    "3. The app displays a class based on a 0.5 threshold.  \n"
    "4. The app never opens the submitted URL."
)
st.caption("Dataset: UCI PhiUSIIL. See README for training instructions and limitations.")
