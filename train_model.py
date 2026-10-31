"""
Train a character-level CNN using the raw URL string from UCI PhiUSIIL.
Label convention in the source: 1 = legitimate, 0 = phishing.
The trained model accepts raw strings directly.
"""
import json
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from ucimlrepo import fetch_ucirepo

SEED = 42
MAX_TOKENS = 160
MAX_LEN = 200
BATCH_SIZE = 256
EPOCHS = 8
ARTIFACT_DIR = "artifacts"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("Downloading PhiUSIIL from UCI...")
dataset = fetch_ucirepo(id=967)
features = dataset.data.features.copy()
targets = dataset.data.targets

# The UCI package may expose the target separately or alongside the features.
label_col = next((c for c in features.columns if str(c).strip().lower() == "label"), None)
if label_col is not None:
    y_raw = features[label_col]
    X_df = features.drop(columns=[label_col])
elif targets is not None and len(targets.columns):
    y_raw = targets.iloc[:, 0]
    X_df = features
else:
    raise ValueError("Could not find the Label target in the UCI dataset. Inspect dataset.data.features/targets.")

url_col = next((c for c in X_df.columns if str(c).strip().lower() == "url"), None)
if url_col is None:
    raise ValueError(f"Could not find a URL column. Available columns: {list(X_df.columns)}")

df = pd.DataFrame({"url": X_df[url_col].astype(str), "label_source": pd.to_numeric(y_raw, errors="coerce")})
df = df.dropna()
df["url"] = df["url"].str.strip()
df = df[df["url"].str.len() > 0]
df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

# Source convention: 0 = phishing, 1 = legitimate. Convert to phishing=1 for model training.
# Model output is probability of phishing.
df["phishing"] = (df["label_source"] == 0).astype("int32")
df = df[df["label_source"].isin([0, 1])].copy()

print(f"Rows after cleaning/deduplication: {len(df):,}")
print("Class counts (0=legitimate, 1=phishing for model):")
print(df["phishing"].value_counts().sort_index().rename(index={0: "legitimate", 1: "phishing"}))

X_train, X_test, y_train, y_test = train_test_split(
    df["url"].values, df["phishing"].values,
    test_size=0.20, random_state=SEED, stratify=df["phishing"].values
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
)

# TextVectorization learns its vocabulary only from training URLs.
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    standardize=None,
    split="character",
    name="char_vectorizer",
)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(1024))

url_input = tf.keras.Input(shape=(), dtype=tf.string, name="url")
x = vectorizer(url_input)
x = tf.keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=False)(x)
x = tf.keras.layers.Conv1D(128, 5, activation="relu", padding="same")(x)
x = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Dropout(0.30)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.20)(x)
output = tf.keras.layers.Dense(1, activation="sigmoid", name="phishing_probability")(x)
model = tf.keras.Model(url_input, output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
             tf.keras.metrics.AUC(name="auc")]
)
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.string)
X_val_tf = tf.convert_to_tensor(X_val, dtype=tf.string)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
]
history = model.fit(
    X_train_tf,
    y_train,
    validation_data=(X_val_tf, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
X_test_tf = tf.convert_to_tensor(
    X_test,
    dtype=tf.string
)
prob = model.predict(X_test_tf, batch_size=BATCH_SIZE).ravel()
pred = (prob >= 0.5).astype(int)
print("\nHeld-out random-split evaluation (not an external-source test):")
print(classification_report(y_test, pred, target_names=["legitimate", "phishing"], digits=4))
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, pred))
try:
    print("ROC-AUC:", round(float(roc_auc_score(y_test, prob)), 4))
except ValueError:
    print("ROC-AUC unavailable for this split.")

model_path = os.path.join(ARTIFACT_DIR, "phishing_url_cnn.keras")
model.save(model_path)
with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump({
        "model_type": "character-level CNN",
        "dataset": "UCI PhiUSIIL Phishing URL Dataset (id=967)",
        "label_mapping": {"0": "legitimate model class", "1": "phishing model class"},
        "source_label_mapping": {"0": "phishing", "1": "legitimate"},
        "threshold": 0.5,
        "max_tokens": MAX_TOKENS,
        "max_length": MAX_LEN,
        "seed": SEED,
        "warning": "Random split metrics are not proof of real-world or cross-source generalization."
    }, f, indent=2)

print(f"\nSaved model to {model_path}")
print("Download artifacts/phishing_url_cnn.keras and artifacts/metadata.json from Colab for the Streamlit app.")
