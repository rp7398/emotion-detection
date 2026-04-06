# Employee Emotion & Task Optimization System

An ML-powered system that detects employee emotions from text and recommends tasks accordingly.

---

## Project Structure

```
employee-emotion-system/
├── backend/
│   ├── app.py                        # Flask API + serves dashboard
│   ├── train_model.py                # ML training script
│   ├── requirements.txt              # Python dependencies
│   ├── Employee_Emotion_Dashboard 1.xlsx  # Dataset
│   ├── emotion.db                    # SQLite DB (auto-created on first run)
│   ├── models/                       # Saved ML artifacts (auto-created after training)
│   │   ├── model.pkl
│   │   ├── vectorizer.pkl
│   │   ├── scaler.pkl
│   │   ├── label_encoder.pkl
│   │   ├── stress_map.pkl
│   │   └── workload_map.pkl
│   └── static/
│       └── index.html                # Dashboard UI (plain HTML/JS)
└── notebook/
    └── emotion_model.ipynb           # Jupyter notebook (EDA + training)
```

---

## Tech Stack

| Layer     | Technology                        |
|-----------|-----------------------------------|
| ML        | scikit-learn (Random Forest + TF-IDF) |
| API       | Flask (Python)                    |
| Database  | SQLite (zero setup)               |
| Frontend  | Plain HTML + CSS + JavaScript     |
| Charts    | Chart.js (CDN)                    |

---

## How It Works

```
Excel Dataset
     │
     ▼
train_model.py
  ├── Text Statement  ──► TF-IDF (500 features)  ─┐
  ├── Stress Level    ──► encoded + scaled         ├──► Random Forest ──► models/*.pkl
  ├── Workload Level  ──► encoded + scaled         │
  └── Productivity    ──► scaled                  ─┘

app.py (Flask)
  ├── Loads models/*.pkl on startup
  ├── POST /predict  ──► runs model ──► saves to SQLite ──► returns mood + recommendation
  ├── GET  /stats    ──► queries SQLite ──► returns chart data
  ├── GET  /history  ──► returns stored predictions
  └── GET  /         ──► serves static/index.html (dashboard)

Browser (static/index.html)
  ├── Predict tab  ──► form ──► POST /predict ──► shows mood + task recommendation
  └── Dashboard tab ──► GET /stats ──► renders mood pie, productivity bar, trend line, stress alerts
```

---

## Recommendation Logic

| Predicted Mood | Recommended Task | Priority |
|----------------|-----------------|----------|
| Stressed       | Light Work       | Low      |
| Neutral        | Normal Task      | Medium   |
| Happy          | Complex Task     | High     |

---

## Setup & Run

### 1. Install dependencies
```bash
cd employee-emotion-system/backend
pip install -r requirements.txt
```

### 2. Train the model

**Option A — Jupyter Notebook (recommended, includes EDA + charts)**
```bash
jupyter notebook ../notebook/emotion_model.ipynb
```
Run all cells top to bottom.

**Option B — Direct script**
```bash
cd employee-emotion-system/backend
python train_model.py
```

Expected output:
```
Accuracy: 1.0000
Saved model.pkl
Saved vectorizer.pkl
...
Training complete.
```

### 3. Start the Flask API
```bash
cd employee-emotion-system/backend
python app.py
```

Expected output:
```
Starting Flask API on http://localhost:5000
 * Running on http://127.0.0.1:5000
```

### 4. Open the dashboard
Go to **http://localhost:5000** in your browser.

> Keep the terminal running. Closing it stops the server.

---

## API Reference

### POST /predict
**Request:**
```json
{
  "employee_id": "E0001",
  "text_statement": "Deadline is stressing me",
  "stress_level": "High",
  "workload_level": "High",
  "productivity_score": 62
}
```
**Response:**
```json
{
  "predicted_mood": "Stressed",
  "confidence": 0.97,
  "probabilities": { "Happy": 0.01, "Neutral": 0.02, "Stressed": 0.97 },
  "recommendation": {
    "task": "Light Work",
    "description": "Assign low-complexity tasks...",
    "priority": "Low"
  }
}
```

### POST /recommend
```json
{ "mood": "Happy" }
```

### GET /history?limit=100&employee_id=E0001
Returns stored predictions from SQLite.

### GET /stats
Returns aggregated data for all dashboard charts.

---

## Dataset Columns Used

| Column             | Role                        |
|--------------------|-----------------------------|
| Text Statement     | NLP input (TF-IDF)          |
| Mood               | Target label (Happy/Neutral/Stressed) |
| Stress Level       | Numerical feature           |
| Workload Level     | Numerical feature           |
| Productivity Score | Numerical feature           |
