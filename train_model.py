
"""
AI-Powered Malicious URL and Phishing Detection
Character-Level CNN using UCI PhiUSIIL Dataset

Source labels:
    0 = Phishing
    1 = Legitimate

Model labels:
    0 = Legitimate
    1 = Phishing

Input:
    Raw URL strings

Output:
    Probability of phishing
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

MAX_TOKENS = 160
MAX_LEN = 200

BATCH_SIZE = 256
EPOCHS = 8

THRESHOLD = 0.50

ARTIFACT_DIR = "artifacts"

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "phishing_url_cnn.keras"
)

METADATA_PATH = os.path.join(
    ARTIFACT_DIR,
    "metadata.json"
)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# 4. DOWNLOAD UCI DATASET
# ============================================================

print("\nDownloading UCI PhiUSIIL dataset...")

dataset = fetch_ucirepo(id=967)

features = dataset.data.features.copy()
targets = dataset.data.targets

print("Dataset downloaded successfully!")

print("\nFeature columns:")
print(list(features.columns))


# ============================================================
# 5. EXTRACT URL AND LABEL
# ============================================================

label_col = next(
    (
        col for col in features.columns
        if str(col).strip().lower() == "label"
    ),
    None
)

if label_col is not None:

    y_raw = features[label_col]
    X_df = features.drop(columns=[label_col])

elif targets is not None and len(targets.columns) > 0:

    y_raw = targets.iloc[:, 0]
    X_df = features

else:

    raise ValueError(
        "Could not find the Label column."
    )


# Find URL column.

url_col = next(
    (
        col for col in X_df.columns
        if str(col).strip().lower() == "url"
    ),
    None
)

if url_col is None:

    raise ValueError(
        f"URL column not found. Available columns: {list(X_df.columns)}"
    )


# ============================================================
# 6. DATA PREPROCESSING
# ============================================================

print("\nPreprocessing dataset...")

df = pd.DataFrame({
    "url": X_df[url_col].to_numpy(dtype=object),

    "label_source": pd.to_numeric(
        y_raw,
        errors="coerce"
    ).to_numpy()
})


# Remove missing records.

df = df.dropna(
    subset=["url", "label_source"]
).copy()


# Keep valid source labels only.

df = df[
    df["label_source"].isin([0, 1])
].copy()


# Convert URLs to strings.

df["url"] = (
    df["url"]
    .astype(str)
    .str.strip()
)


# Remove empty URLs.

df = df[
    df["url"].str.len() > 0
].copy()


# Remove URLs with conflicting labels.

label_counts = (
    df.groupby("url")["label_source"]
    .nunique()
)

conflicting_urls = label_counts[
    label_counts > 1
].index

if len(conflicting_urls) > 0:

    print(
        "Conflicting URLs removed:",
        len(conflicting_urls)
    )

    df = df[
        ~df["url"].isin(conflicting_urls)
    ].copy()


# Remove duplicate URLs.

df = (
    df.drop_duplicates(
        subset=["url"],
        keep="first"
    )
    .reset_index(drop=True)
)


# ============================================================
# 7. LABEL ENCODING
# ============================================================

# Source:
# 0 = Phishing
# 1 = Legitimate

# Model:
# 0 = Legitimate
# 1 = Phishing

df["phishing"] = (
    df["label_source"] == 0
).astype("int32")


# ============================================================
# 8. DATASET VERIFICATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")

print("Total URLs:", len(df))

print(
    "Missing URLs:",
    df["url"].isna().sum()
)

print(
    "Duplicate URLs:",
    df["url"].duplicated().sum()
)

print(
    "Source labels:",
    sorted(df["label_source"].unique())
)

print("\nClass distribution:")

print(
    df["phishing"]
    .value_counts()
    .sort_index()
    .rename({
        0: "Legitimate",
        1: "Phishing"
    })
)


if len(df) == 0:

    raise ValueError(
        "No valid records remain after preprocessing."
    )


if df["phishing"].nunique() != 2:

    raise ValueError(
        "Both legitimate and phishing classes are required."
    )


# ============================================================
# 9. PREPARE MODEL INPUT ARRAYS
# ============================================================

# Convert pandas data into ordinary NumPy arrays.
# This avoids Arrow-backed indexing problems.

X = df["url"].to_numpy(dtype=object)

y = df["phishing"].to_numpy(dtype=np.int32)


# ============================================================
# 10. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split:
# 80% training pool
# 20% testing

X_train_val, X_test, y_train_val, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=SEED,

    stratify=y

)


# Second split:
# 85% of training pool for training
# 15% of training pool for validation

# Final proportions:
# 68% training
# 12% validation
# 20% testing

X_train, X_val, y_train, y_val = train_test_split(

    X_train_val,

    y_train_val,

    test_size=0.15,

    random_state=SEED,

    stratify=y_train_val

)


print("\n========== DATASET SPLIT ==========")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# ============================================================
# 11. CHARACTER-LEVEL TEXT VECTORIZATION
# ============================================================

print("\nCreating character-level vectorizer...")

vectorizer = tf.keras.layers.TextVectorization(

    max_tokens=MAX_TOKENS,

    output_mode="int",

    output_sequence_length=MAX_LEN,

    standardize=None,

    split="character",

    name="char_vectorizer"

)


# Learn vocabulary using training data only.

vectorizer.adapt(

    tf.data.Dataset
    .from_tensor_slices(X_train)
    .batch(1024)

)

print("Character-level vectorization completed.")


# ============================================================
# 12. BUILD CHARACTER-LEVEL CNN MODEL
# ============================================================

print("\nBuilding CNN model...")

url_input = tf.keras.Input(

    shape=(),

    dtype=tf.string,

    name="url"

)


# Character encoding.

x = vectorizer(url_input)


# Embedding layer.

x = tf.keras.layers.Embedding(

    input_dim=MAX_TOKENS,

    output_dim=64,

    mask_zero=False

)(x)


# Convolutional layer.

x = tf.keras.layers.Conv1D(

    filters=128,

    kernel_size=5,

    activation="relu",

    padding="same"

)(x)


# Global max pooling.

x = tf.keras.layers.GlobalMaxPooling1D()(x)


# Dropout.

x = tf.keras.layers.Dropout(0.30)(x)


# Dense layer.

x = tf.keras.layers.Dense(

    64,

    activation="relu"

)(x)


x = tf.keras.layers.Dropout(0.20)(x)


# Output layer.

output = tf.keras.layers.Dense(

    1,

    activation="sigmoid",

    name="phishing_probability"

)(x)


# Create model.

model = tf.keras.Model(

    inputs=url_input,

    outputs=output

)


# ============================================================
# 13. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-3

    ),

    loss="binary_crossentropy",

    metrics=[

        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )

    ]

)


print("\nModel architecture:")

model.summary()


# ============================================================
# 14. PREPARE TENSORFLOW INPUTS
# ============================================================

X_train_tf = tf.convert_to_tensor(

    X_train.tolist(),

    dtype=tf.string

)


X_val_tf = tf.convert_to_tensor(

    X_val.tolist(),

    dtype=tf.string

)


X_test_tf = tf.convert_to_tensor(

    X_test.tolist(),

    dtype=tf.string

)


# ============================================================
# 15. MODEL TRAINING
# ============================================================

print("\n========== MODEL TRAINING ==========")

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=2,

        restore_best_weights=True

    )

]


history = model.fit(

    X_train_tf,

    y_train,

    validation_data=(

        X_val_tf,

        y_val

    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1

)


# ============================================================
# 16. MODEL PREDICTION
# ============================================================

print("\nGenerating test predictions...")

prob = model.predict(

    X_test_tf,

    batch_size=BATCH_SIZE,

    verbose=1

).ravel()


# Apply classification threshold.

pred = (

    prob >= THRESHOLD

).astype("int32")


# ============================================================
# 17. MODEL EVALUATION
# ============================================================

print("\n========== MODEL EVALUATION ==========")


print("\nClassification Report:\n")

report = classification_report(

    y_test,

    pred,

    labels=[0, 1],

    target_names=[

        "Legitimate",

        "Phishing"

    ],

    digits=4,

    zero_division=0

)

print(report)


# Confusion matrix.

cm = confusion_matrix(

    y_test,

    pred,

    labels=[0, 1]

)


print("\nConfusion Matrix:")

print("[[TN, FP], [FN, TP]]")

print(cm)


# Calculate metrics.

accuracy = accuracy_score(

    y_test,

    pred

)


precision = precision_score(

    y_test,

    pred,

    zero_division=0

)


recall = recall_score(

    y_test,

    pred,

    zero_division=0

)


f1 = f1_score(

    y_test,

    pred,

    zero_division=0

)


roc_auc = roc_auc_score(

    y_test,

    prob

)


pr_auc = average_precision_score(

    y_test,

    prob

)


print("\n========== PERFORMANCE METRICS ==========")

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1-Score  : {f1:.4f}")

print(f"ROC-AUC   : {roc_auc:.4f}")

print(f"PR-AUC    : {pr_auc:.4f}")


print(

    "\nNote: These are held-out random-split results, "

    "not external-source or real-world performance estimates."

)


# ============================================================
# 18. SAVE TRAINED MODEL
# ============================================================

print("\nSaving trained model...")

model.save(MODEL_PATH)

print("Model saved successfully!")

print("Model location:", os.path.abspath(MODEL_PATH))


# ============================================================
# 19. SAVE METADATA
# ============================================================

metadata = {

    "model_type": "Character-Level CNN",

    "dataset": "UCI PhiUSIIL Phishing URL Dataset",

    "dataset_id": 967,

    "label_mapping": {

        "0": "Legitimate",

        "1": "Phishing"

    },

    "source_label_mapping": {

        "0": "Phishing",

        "1": "Legitimate"

    },

    "prediction_output": "Probability of phishing",

    "classification_threshold": THRESHOLD,

    "max_tokens": MAX_TOKENS,

    "max_length": MAX_LEN,

    "batch_size": BATCH_SIZE,

    "epochs_configured": EPOCHS,

    "epochs_completed": len(history.history["loss"]),

    "random_seed": SEED,

    "dataset_size_after_cleaning": int(len(df)),

    "training_samples": int(len(X_train)),

    "validation_samples": int(len(X_val)),

    "testing_samples": int(len(X_test)),

    "evaluation": {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "roc_auc": float(roc_auc),

        "pr_auc": float(pr_auc)

    },

    "confusion_matrix": cm.tolist(),

    "evaluation_note": (

        "Held-out random-split evaluation. "

        "Not proof of external-source or real-world generalization."

    )

}


with open(

    METADATA_PATH,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        metadata,

        f,

        indent=4

    )


# ============================================================
# 20. VERIFY SAVED FILES
# ============================================================

print("\n========== SAVED FILE VERIFICATION ==========")

print("Model exists:", os.path.exists(MODEL_PATH))

print("Metadata exists:", os.path.exists(METADATA_PATH))

if os.path.exists(MODEL_PATH):

    print(

        "Model size (MB):",

        round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2)

    )


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n========== PROJECT COMPLETED ==========")

print("Training completed successfully.")

print("\nGenerated files:")

print("1.", MODEL_PATH)

print("2.", METADATA_PATH)

print("\nUse the .keras model in your Streamlit application.")
