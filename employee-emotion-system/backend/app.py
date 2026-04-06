"""
Flask API — Employee Emotion & Task Optimization System
Uses SQLite (no MySQL install needed).

Endpoints:
  GET  /              → dashboard HTML
  POST /predict       → emotion prediction
  POST /recommend     → task recommendation
  GET  /history       → stored predictions
  GET  /stats         → mood/productivity stats for charts
"""

import os, pickle, sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from scipy.sparse import hstack, csr_matrix

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
STATIC_DIR  = os.path.join(BASE_DIR, "static")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DB_PATH     = os.path.join(BASE_DIR, "emotion.db")

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# ── Load ML artifacts ─────────────────────────────────────────────────────────
def load(name):
    with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)

model       = load("model")
vectorizer  = load("vectorizer")
scaler      = load("scaler")
le          = load("label_encoder")
stress_map  = load("stress_map")
workload_map= load("workload_map")

# ── SQLite setup ──────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id        TEXT    NOT NULL,
                text_statement     TEXT    NOT NULL,
                stress_level       TEXT    NOT NULL,
                workload_level     TEXT    NOT NULL,
                productivity_score REAL    NOT NULL,
                predicted_mood     TEXT    NOT NULL,
                recommendation     TEXT    NOT NULL,
                created_at         TEXT    NOT NULL
            )
        """)
        conn.commit()

init_db()

# ── Recommendation logic ──────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Stressed": {
        "task": "Light Work",
        "description": "Assign low-complexity tasks — documentation, routine updates, easy reviews.",
        "priority": "Low",
    },
    "Happy": {
        "task": "Complex Task",
        "description": "Leverage peak performance — assign high-priority, challenging work.",
        "priority": "High",
    },
    "Neutral": {
        "task": "Normal Task",
        "description": "Standard workload with medium priority is ideal.",
        "priority": "Medium",
    },
}

# ── Feature builder ───────────────────────────────────────────────────────────
def build_features(text, stress_level, workload_level, productivity_score):
    tfidf       = vectorizer.transform([text])
    stress_num  = stress_map.get(stress_level, 1)
    work_num    = workload_map.get(workload_level, 1)
    num         = scaler.transform([[stress_num, work_num, float(productivity_score)]])
    return hstack([tfidf, csr_matrix(num)])

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    required = ["text_statement", "stress_level", "workload_level", "productivity_score"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    text        = str(data["text_statement"]).strip()
    stress      = str(data["stress_level"])
    workload    = str(data["workload_level"])
    productivity= float(data["productivity_score"])
    emp_id      = data.get("employee_id", "unknown")

    X           = build_features(text, stress, workload, productivity)
    pred_idx    = model.predict(X)[0]
    proba       = model.predict_proba(X)[0]
    mood        = le.inverse_transform([pred_idx])[0]
    confidence  = float(np.max(proba))
    rec         = RECOMMENDATIONS.get(mood, RECOMMENDATIONS["Neutral"])

    # Store in SQLite
    with get_db() as conn:
        conn.execute(
            """INSERT INTO predictions
               (employee_id, text_statement, stress_level, workload_level,
                productivity_score, predicted_mood, recommendation, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (emp_id, text, stress, workload, productivity,
             mood, rec["task"], datetime.utcnow().isoformat()),
        )
        conn.commit()

    return jsonify({
        "employee_id":   emp_id,
        "predicted_mood": mood,
        "confidence":    round(confidence, 4),
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)},
        "recommendation": rec,
    })

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True)
    mood = data.get("mood", "Neutral")
    return jsonify({"mood": mood, "recommendation": RECOMMENDATIONS.get(mood, RECOMMENDATIONS["Neutral"])})

@app.route("/history")
def history():
    limit = int(request.args.get("limit", 100))
    emp   = request.args.get("employee_id")
    with get_db() as conn:
        if emp:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE employee_id=? ORDER BY created_at DESC LIMIT ?",
                (emp, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/stats")
def stats():
    with get_db() as conn:
        # Mood counts
        mood_rows = conn.execute(
            "SELECT predicted_mood, COUNT(*) as count FROM predictions GROUP BY predicted_mood"
        ).fetchall()
        # Avg productivity per mood
        prod_rows = conn.execute(
            "SELECT predicted_mood, ROUND(AVG(productivity_score),1) as avg_prod FROM predictions GROUP BY predicted_mood"
        ).fetchall()
        # Daily mood trend (last 14 days)
        trend_rows = conn.execute(
            """SELECT substr(created_at,1,10) as date, predicted_mood, COUNT(*) as count
               FROM predictions
               GROUP BY date, predicted_mood
               ORDER BY date DESC LIMIT 42"""
        ).fetchall()
        # Recent stress alerts
        alerts = conn.execute(
            """SELECT employee_id, text_statement, stress_level, workload_level, created_at
               FROM predictions WHERE predicted_mood='Stressed'
               ORDER BY created_at DESC LIMIT 10"""
        ).fetchall()

    return jsonify({
        "mood_counts":  [dict(r) for r in mood_rows],
        "productivity": [dict(r) for r in prod_rows],
        "trend":        [dict(r) for r in trend_rows],
        "alerts":       [dict(r) for r in alerts],
    })

if __name__ == "__main__":
    print("Starting Flask API on http://localhost:5000")
    app.run(debug=True, port=5000)
