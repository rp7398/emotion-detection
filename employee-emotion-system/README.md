# Employee Emotion & Task Optimization System

An ML-powered system that detects employee emotions from text statements and recommends tasks based on the predicted mood.

---

## Project Structure

```
employee-emotion-system/
├── backend/
│   ├── app.py                             # Flask API + serves the dashboard
│   ├── train_model.py                     # ML training script
│   ├── requirements.txt                   # Python dependencies
│   ├── Employee_Emotion_Dashboard 1.xlsx  # Dataset (8000 rows)
│   ├── emotion.db                         # SQLite database (auto-created on first run)
│   ├── models/                            # Saved ML artifacts (auto-created after training)
│   │   ├── model.pkl                      # Trained Random Forest
│   │   ├── vectorizer.pkl                 # TF-IDF Vectorizer
│   │   ├── scaler.pkl                     # StandardScaler
│   │   ├── label_encoder.pkl              # Label Encoder (Happy / Neutral / Stressed)
│   │   ├── stress_map.pkl                 # Stress level mapping
│   │   └── workload_map.pkl               # Workload level mapping
│   └── static/
│       └── index.html                     # Dashboard UI (plain HTML + JS)
├── notebook/
│   └── emotion_model.ipynb                # Jupyter Notebook (EDA + training)
└── README.md
```

---

## Tech Stack

| Layer    | Technology                            |
|----------|---------------------------------------|
| ML       | scikit-learn — Random Forest + TF-IDF |
| API      | Flask (Python)                        |
| Database | SQLite (zero setup, no install)       |
| Frontend | Plain HTML + CSS + JavaScript         |
| Charts   | Chart.js (loaded via CDN)             |

---

## Setup & Run

### Step 1 — Install Python dependencies
```bash
cd employee-emotion-system/backend
pip install -r requirements.txt
```

### Step 2 — Train the model

Option A — Jupyter Notebook (includes EDA charts):
```bash
jupyter notebook ../notebook/emotion_model.ipynb
```
Run all cells from top to bottom.

Option B — Script only:
```bash
cd employee-emotion-system/backend
python train_model.py
```

Expected output:
```
Accuracy: 1.0000
Saved model.pkl
Saved vectorizer.pkl
Saved scaler.pkl
Saved label_encoder.pkl
Saved stress_map.pkl
Saved workload_map.pkl
Training complete.
```

### Step 3 — Start the Flask server
```bash
cd employee-emotion-system/backend
python app.py
```

Expected output:
```
Starting Flask API on http://localhost:5000
 * Running on http://127.0.0.1:5000
```

### Step 4 — Open the dashboard
Visit http://localhost:5000 in your browser.

> Keep the terminal open. Closing it stops the server.

---

## How It Works

### Training phase
```
Excel Dataset
    │
    ├── Text Statement   ──► TF-IDF Vectorizer (500 features)
    ├── Stress Level     ──► encoded (Low=0, Medium=1, High=2) + scaled
    ├── Workload Level   ──► encoded (Low=0, Medium=1, High=2) + scaled
    └── Productivity Score ──► scaled
                │
                ▼
        Random Forest Classifier
                │
                ▼
          models/*.pkl  (saved to disk)
```

### Prediction phase
```
User fills form on dashboard
    │
    ▼
POST /predict  (JSON sent to Flask)
    │
    ├── Text  ──► same TF-IDF vectorizer transforms it
    ├── Stress / Workload / Productivity  ──► same scaler transforms them
    │
    ▼
Random Forest predicts mood  (Happy / Neutral / Stressed)
    │
    ├── Result saved to SQLite (emotion.db)
    └── Response sent back to browser
            │
            ▼
    Dashboard shows:
      - Predicted mood + confidence
      - Probability bars
      - Task recommendation
```

---

## Recommendation Logic

| Predicted Mood | Recommended Task | Priority |
|----------------|-----------------|----------|
| Stressed       | Light Work       | Low      |
| Neutral        | Normal Task      | Medium   |
| Happy          | Complex Task     | High     |

---

## Dashboard Features

- Predict tab — enter employee details, get instant emotion prediction and task recommendation
- Dashboard tab — visualizations built from stored prediction history:
  - Mood distribution (donut chart)
  - Average productivity by mood (bar chart)
  - Daily mood trend (line chart)
  - Recent stress alerts list

---

## API Endpoints

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| GET    | /          | Serves the dashboard (index.html)    |
| POST   | /predict   | Predict emotion from input           |
| POST   | /recommend | Get task recommendation for a mood   |
| GET    | /history   | Fetch stored predictions from DB     |
| GET    | /stats     | Aggregated data for dashboard charts |
| GET    | /health    | Health check                         |

### POST /predict — example request
```json
{
  "employee_id": "E0002",
  "text_statement": "Deadline is stressing me",
  "stress_level": "High",
  "workload_level": "Low",
  "productivity_score": 57
}
```

### POST /predict — example response
```json
{
  "employee_id": "E0002",
  "predicted_mood": "Stressed",
  "confidence": 1.0,
  "probabilities": {
    "Happy": 0.0,
    "Neutral": 0.0,
    "Stressed": 1.0
  },
  "recommendation": {
    "task": "Light Work",
    "description": "Assign low-complexity tasks — documentation, routine updates, easy reviews.",
    "priority": "Low"
  }
}
```

---

## Dataset

File: `Employee_Emotion_Dashboard 1.xlsx`
Rows: 8000 | Columns: 21

Columns used for training:

| Column             | Purpose                              |
|--------------------|--------------------------------------|
| Text Statement     | NLP input — converted via TF-IDF     |
| Mood               | Target label (Happy / Neutral / Stressed) |
| Stress Level       | Numerical feature                    |
| Workload Level     | Numerical feature                    |
| Productivity Score | Numerical feature                    |
