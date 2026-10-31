# AI-Powered Malicious URL and Phishing Detection — YouTube Project

This is a beginner-friendly deep-learning demo for a YouTube screen recording. It includes:
- `train_model.py`: trains a character-level CNN using raw URL strings from UCI PhiUSIIL.
- `app.py`: Streamlit app for interactive predictions.
- `requirements.txt`: Python dependencies.
- `artifacts/`: put the trained `.keras` model and metadata here after training.

## Dataset

UCI PhiUSIIL Phishing URL Dataset:
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+website+dataset

The official dataset contains 235,795 rows, including 134,850 legitimate and 100,945 phishing examples. In the source label, `1 = legitimate` and `0 = phishing`. The training script maps `0` to model class 1 (phishing) so the model output is a phishing score.

This project intentionally uses only the raw `URL` column for the CNN. The published dataset also contains engineered features, including some derived from webpage source code; this demo does not use those columns.

## Part A — Train in Google Colab

1. Open https://colab.research.google.com/
2. Create a new notebook.
3. In Runtime → Change runtime type, select a GPU if one is available. Colab GPU availability and quotas vary.
4. Upload `train_model.py` into the Colab session, or copy its code into a notebook cell.
5. Run this install cell:

```python
!pip -q install ucimlrepo scikit-learn pandas
```

6. Run training:

```python
!python train_model.py
```

The script downloads the dataset via `ucimlrepo`, trains the model, prints the evaluation metrics from its held-out random split, and saves:
- `artifacts/phishing_url_cnn.keras`
- `artifacts/metadata.json`

If you ran it as a notebook cell instead, download the model and metadata from the Files pane. Do not report metrics until your run has completed; copy the actual output into your presentation.

### If the UCI downloader exposes columns differently

UCI packages can change how columns are exposed. The script checks both `data.features` and `data.targets`, but if it stops with a column error, run:

```python
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=967)
print(ds.data.features.columns.tolist())
print(ds.data.targets.columns.tolist() if ds.data.targets is not None else None)
```

Then adjust the URL and label column selection to match the printed columns. Do not silently guess the label mapping.

## Part B — Run the Streamlit app

Recommended: use Python 3.11 or 3.12 in a virtual environment.

1. Copy `app.py`, `requirements.txt`, and the `artifacts` folder (containing `phishing_url_cnn.keras`) into one project folder.
2. Open a terminal in that folder.
3. Install packages:

```bash
python -m pip install -r requirements.txt
```

4. Run:

```bash
streamlit run app.py
```

The browser should open the app. The model file must be at `artifacts/phishing_url_cnn.keras`.

## Part C — Optional public deployment

Push the project to a GitHub repository, including the trained model artifact if its size and your repository policy allow it. In Streamlit Community Cloud, select the repository, branch, and `app.py` as the entrypoint:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy

If the model file is too large for GitHub, use an appropriate model artifact hosting method and modify the loading code to retrieve it safely. Never commit secrets or tokens.

## YouTube demo plan (about 4–6 minutes)

1. **0:00–0:25 — Introduction:** “This project uses deep learning to estimate whether a URL resembles phishing URLs.”
2. **0:25–1:00 — Dataset:** Show the UCI PhiUSIIL page and explain the original label convention.
3. **1:00–2:00 — Model:** Show the character-level CNN and explain that it learns patterns from URL text, not page content.
4. **2:00–3:30 — Live demo:** Enter the two provided example strings and one additional URL you control or have safely copied as text. Do not open suspicious links.
5. **3:30–4:30 — Evaluation:** Show the actual classification report and confusion matrix printed by your training run. Do not invent results.
6. **4:30–5:00 — Limitations:** Explain that a random split does not prove performance on new sources or newly emerging attacks.

### Suggested narration

“Hello everyone. In this video, I’m demonstrating an AI-powered malicious URL and phishing detection system. I use a public UCI dataset and train a character-level convolutional neural network. The application analyzes the URL text only and does not open the website. The displayed score is a model estimate, not a guarantee of safety. I’ll show the dataset, training process, evaluation results from my actual run, and the live Streamlit demo.”

## Research and safety limitations

- The default evaluation is a stratified random split after exact URL deduplication. Similar domains can still occur across splits; therefore, results do not establish cross-domain or cross-source generalization.
- For a research paper, add domain-disjoint splits and a truly independent, separately collected test set. Ensure timestamps and labels are trustworthy.
- The score is not necessarily calibrated; do not call it a real-world probability without calibration testing.
- A URL may be dangerous even if the model says “likely legitimate.” Do not use this demo as the sole security control.
- The example `.example` domain is reserved for documentation and is not a live phishing URL.
- Cite the original PhiUSIIL dataset and paper in academic work. Verify the dataset license and citation instructions on UCI before redistribution.

## Project structure

```text
phishing_url_youtube_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── artifacts/
    └── (place trained model and metadata here)
```
