#  AI/ML Project — Iris Classifier

A **production-style end-to-end machine learning project** that demonstrates how to move from dataset → trained model → REST API → Docker → CI.

## 🚀 Features

* 🧹 Modular ML pipeline
* ⚙️ YAML-based configuration
* 🤖 Random Forest classification
* 📊 Model evaluation & metrics
* 🌐 FastAPI prediction API
* ✅ Pytest testing
* 🐳 Docker & Docker Compose
* 🔄 GitHub Actions CI
* 🛠️ Makefile for one-command execution

---

## 📁 Project Structure

```text
ai-ml-project/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_eda.md
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── api/
│       └── main.py
├── tests/
│   └── test_model.py
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🏗️ Architecture

```text
Dataset
   ↓
Data Ingestion
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Evaluation
   ↓
model.pkl
   ↓
FastAPI
   ↓
Prediction
```

Each layer has a separate responsibility, making the project easier to **maintain, test, and extend**.

---

## ⚙️ Configuration

All important paths and hyperparameters are stored in:

```text
config/config.yaml
```

Example:

```yaml
model:
  n_estimators: 100
  max_depth: 5
  random_state: 42

training:
  test_size: 0.2
```

No need to modify Python code just to change model settings.

---

## 💻 Run Locally

### 1. Create environment

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the ML pipeline

```bash
make pipeline
```

### 4. Start the API

```bash
make serve
```

API:

```text
http://localhost:8000
```

Swagger Docs:

```text
http://localhost:8000/docs
```

### 5. Run tests

```bash
make test
```

---

## 🐳 Run with Docker

No local Python setup is required.

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000/docs
```

---

## 🔮 Example Prediction

### Request

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Response

```json
{
  "prediction": "setosa"
}
```

---

## 🧪 Testing & CI

Tests are written using **pytest** and automatically executed through **GitHub Actions**.

```text
Git Push
   ↓
GitHub Actions
   ↓
Install Dependencies
   ↓
Run Tests
   ↓
PASS / FAIL
```

---

## 🛠️ Tech Stack

| Category   | Tools             |
| ---------- | ----------------- |
| Language   | Python            |
| ML         | Scikit-learn      |
| Data       | Pandas, NumPy     |
| API        | FastAPI, Pydantic |
| Testing    | Pytest            |
| Container  | Docker            |
| CI/CD      | GitHub Actions    |
| Automation | Make              |
| Config     | YAML              |

---

## 🎯 Why This Project?

Instead of just creating a model in a notebook, this project demonstrates how to build a **complete ML application**:

```text
Data → ML → API → Testing → Docker → CI/CD
```

The structure can later be reused for **Churn Prediction, Fraud Detection, Recommendation Systems, NLP, Computer Vision, or other AI/ML projects**.

---

## 📌 Resume Description

> **Production-Style Iris Classification API** — Built an end-to-end ML pipeline with Scikit-learn, FastAPI, Pydantic, Docker, Pytest, YAML configuration, and GitHub Actions CI for reproducible model deployment.
