"""
Train emotion classification model using TF-IDF + numerical features.
Saves model and vectorizer as pickle files.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, csr_matrix

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get(
    "DATA_PATH",
    os.path.join(os.path.dirname(__file__), "Employee_Emotion_Dashboard 1.xlsx"),
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

STRESS_LEVEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
WORKLOAD_LEVEL_MAP = {"Low": 0, "Medium": 1, "High": 2}

# ── Load data ────────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    return df

# ── Feature engineering ──────────────────────────────────────────────────────
def build_features(df, vectorizer=None, scaler=None, fit=True):
    # TF-IDF on text
    texts = df["Text Statement"].fillna("").astype(str)
    if fit:
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)

    # Numerical features
    stress_num = df["Stress Level"].map(STRESS_LEVEL_MAP).fillna(1).values
    workload_num = df["Workload Level"].map(WORKLOAD_LEVEL_MAP).fillna(1).values
    productivity = df["Productivity Score"].fillna(df["Productivity Score"].median()).values

    num_features = np.column_stack([stress_num, workload_num, productivity])

    if fit:
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(num_features)
    else:
        num_scaled = scaler.transform(num_features)

    X = hstack([tfidf_matrix, csr_matrix(num_scaled)])
    return X, vectorizer, scaler

# ── Train ────────────────────────────────────────────────────────────────────
def train(data_path=DATA_PATH):
    print(f"Loading data from: {data_path}")
    df = load_data(data_path)

    # Encode target (Mood: Happy / Neutral / Stressed)
    le = LabelEncoder()
    y = le.fit_transform(df["Mood"])

    X, vectorizer, scaler = build_features(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Persist artifacts
    artifacts = {
        "model": model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "label_encoder": le,
        "stress_map": STRESS_LEVEL_MAP,
        "workload_map": WORKLOAD_LEVEL_MAP,
    }
    for name, obj in artifacts.items():
        with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)
        print(f"Saved {name}.pkl")

    print("Training complete.")

if __name__ == "__main__":
    train()
