# Student Outcome MLOps

An end-to-end MLOps project for predicting student outcomes using Machine Learning, DVC, MLflow, Airflow, FastAPI, Docker, and GitHub Actions.

The project predicts one of three student outcomes:

* `Dropout`
* `Enrolled`
* `Graduate`

---

## 1. Project Overview

This project demonstrates a complete machine learning workflow from data validation and model training to model evaluation, quality control, API serving, UI demonstration, containerization, and CI.

### Main workflow

```text
                    ┌─────────────────┐
                    │   Raw Dataset   │
                    │   dataset.csv   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   DVC + MinIO   │
                    │ Data Versioning │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Airflow     │
                    │   Orchestrator │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        Validate Data     Train Model    Evaluate
              │              │              │
              │              ▼              ▼
              │           MLflow       metrics.json
              │              │              │
              └──────────────┴──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Quality Gate   │
                    │ Macro F1 >= 0.70│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Trained Model  │
                    │ best_model.joblib│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    FastAPI      │
                    │   /predict      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Gradio UI    │
                    └─────────────────┘

              GitHub Actions
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   API Tests              Docker Builds
```

---

## 2. Technologies

| Technology       | Purpose                              |
| ---------------- | ------------------------------------ |
| Python           | Main programming language            |
| Scikit-learn     | Machine Learning                     |
| Random Forest    | Classification model                 |
| imbalanced-learn | Random Oversampling                  |
| DVC              | Dataset/model pipeline versioning    |
| MinIO            | S3-compatible object storage for DVC |
| MLflow           | Experiment tracking                  |
| Apache Airflow   | Pipeline orchestration               |
| FastAPI          | Model serving API                    |
| Gradio           | User interface                       |
| Docker           | Containerization                     |
| GitHub Actions   | Continuous Integration               |
| Pytest           | API testing                          |

---

## 3. Project Structure

```text
student-success-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── raw/
│       ├── dataset.csv
│       └── dataset.csv.dvc
│
├── models/
│   └── best_model.joblib
│
├── reports/
│   └── metrics.json
│
├── project/
│   └── src/
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       ├── validate_data.py
│       ├── quality_gate.py
│       ├── data_preprocessing.py
│       ├── feature_engineering.py
│       ├── run_pipeline.py
│       │
│       ├── api/
│       │   ├── main.py
│       │   └── schemas.py
│       │
│       └── ui/
│           └── app.py
│
├── dags/
│   └── student_mlops_dag.py
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── dockerfile.ui
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 4. Machine Learning

The project uses a tuned `RandomForestClassifier`.

### Model configuration

```text
n_estimators = 700
criterion = entropy
max_depth = None
min_samples_split = 2
min_samples_leaf = 2
max_features = 0.5
class_weight = balanced
random_state = 42
n_jobs = -1
```

The training pipeline includes feature engineering and `RandomOverSampler`.

### Dataset split

```text
Training samples: 3539
Test samples:      885
```

Target classes:

```text
Dropout
Enrolled
Graduate
```

---

## 5. Model Evaluation

Current evaluation results:

| Class            |  Precision |     Recall |   F1-score |
| ---------------- | ---------: | ---------: | ---------: |
| Dropout          |     0.8487 |     0.7113 |     0.7739 |
| Enrolled         |     0.4859 |     0.5409 |     0.5119 |
| Graduate         |     0.8298 |     0.8824 |     0.8553 |
| **Macro Avg**    | **0.7215** | **0.7115** | **0.7137** |
| **Weighted Avg** |            |            | **0.7675** |

Accuracy:

```text
0.7661
```

The quality gate uses Macro F1:

```text
Minimum Macro F1 = 0.70
```

The model passes the current quality threshold with:

```text
Macro F1 = 0.7137
```

---

## 6. DVC and MinIO

DVC is used to version the dataset and manage data dependencies.

The dataset is tracked using:

```text
data/raw/dataset.csv.dvc
```

The DVC remote uses MinIO:

```text
s3://mlops-data/dvc
```

MinIO provides S3-compatible object storage for the local MLOps environment.

The important distinction is:

```text
Git
 └── tracks metadata/pointers

DVC
 └── tracks data/model artifacts

MinIO
 └── stores the actual DVC artifacts
```

The dataset itself is not committed directly to Git.

---

## 7. MLflow

MLflow is used for experiment tracking.

The project records:

* Model parameters
* Classification metrics
* Accuracy
* Macro F1
* Weighted F1
* Per-class precision
* Per-class recall
* Per-class F1
* Evaluation metrics artifact
* Trained model

Experiment:

```text
MLflow Quickstart
```

---

## 8. Airflow Pipeline

Apache Airflow orchestrates the machine learning workflow.

The DAG is:

```text
validate_data
      ↓
    train
      ↓
  evaluate
      ↓
 quality_gate
```

The DAG file is:

```text
dags/student_mlops_dag.py
```

The pipeline is manually triggered through Airflow.

Airflow is currently run locally in WSL rather than inside Docker.

---

## 9. Quality Gate

The quality gate prevents a model from being considered valid when its Macro F1 is below the required threshold.

```text
metrics.json
      │
      ▼
Read Macro F1
      │
      ▼
Macro F1 >= 0.70 ?
   │           │
  YES          NO
   │           │
 PASS         FAIL
```

Current threshold:

```python
MIN_F1 = 0.70
```

---

## 10. FastAPI

The trained model is served through FastAPI.

API:

```text
http://localhost:8005
```

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Prediction

```http
POST /predict
```

The endpoint accepts student information defined by the `StudentInput` schema and returns a prediction from the trained model.

FastAPI automatically provides interactive API documentation at:

```text
/docs
```

---

## 11. Gradio UI

A Gradio interface is provided for interacting with the prediction API.

The UI communicates with the API through the Docker Compose service name:

```text
http://api:8005
```

The UI is exposed on:

```text
http://localhost:7860
```

---

## 12. Docker

The project uses Docker Compose to run:

```text
┌───────────────┐
│     MinIO     │
│   :9000/:9001 │
└───────────────┘

┌───────────────┐
│    FastAPI    │
│     :8005     │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│    Gradio     │
│     :7860     │
└───────────────┘
```

Start the services:

```bash
docker compose up --build
```

Run in background:

```bash
docker compose up --build -d
```

Stop the services:

```bash
docker compose down
```

---

## 13. Running the Project Locally

### 13.1 Clone repository

```bash
git clone https://github.com/sondts-ai/student_outcome_mlops.git
cd student_outcome_mlops
```

### 13.2 Install dependencies

Create and activate a Python environment, then:

```bash
pip install -r requirements.txt
```

---

### 13.3 Run the ML pipeline

The main pipeline can be executed through:

```bash
python -m project.src.run_pipeline
```

The individual components are also available:

```bash
python -m project.src.validate_data
python -m project.src.train
python -m project.src.evaluate
python -m project.src.quality_gate
```

---

## 14. Running Airflow

The Airflow DAG is located at:

```text
dags/student_mlops_dag.py
```

DAG:

```text
student_mlops
```

Pipeline:

```text
Validate Data
      ↓
Train
      ↓
Evaluate
      ↓
Quality Gate
```

---

## 15. Running the API

Start FastAPI directly:

```bash
uvicorn project.src.api.main:app --host 0.0.0.0 --port 8005
```

Then open:

```text
http://localhost:8005/docs
```

---

## 16. Running Tests

API tests use Pytest.

Run:

```bash
PYTHONPATH=. pytest tests/test_api.py -v
```

The tests cover:

* Health endpoint
* Prediction endpoint
* Invalid input validation

The prediction API test mocks the model prediction function, so the API unit test does not require the trained model artifact to be available in the GitHub Actions runner.

---

## 17. Continuous Integration

GitHub Actions is used for CI.

The workflow is located at:

```text
.github/workflows/ci.yml
```

Current CI pipeline:

```text
Checkout repository
        ↓
Install dependencies
        ↓
Test imports
        ↓
Run API tests
        ↓
Build API Docker image
        ↓
Build UI Docker image
```

The CI pipeline does not train the model or pull data from the local MinIO server.

This is intentional because the current DVC remote is a local MinIO instance and is not accessible from the GitHub-hosted runner.

---

## 18. MLOps Architecture

The overall system can be summarized as:

```text
                    DATA
                     │
                     ▼
              ┌─────────────┐
              │ DVC + MinIO │
              └──────┬──────┘
                     │
                     ▼
                ┌─────────┐
                │ Airflow │
                └────┬────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Validate    Train     Evaluate
                     │          │
                     ▼          ▼
                  MLflow    metrics.json
                     │          │
                     └────┬─────┘
                          ▼
                    Quality Gate
                          │
                          ▼
                   Model Artifact
                          │
                          ▼
                       FastAPI
                          │
                          ▼
                      Gradio UI

                GitHub Actions
                       │
              ┌────────┴────────┐
              ▼                 ▼
          API Tests        Docker Build
```

---

## 19. Key MLOps Concepts Demonstrated

This project demonstrates the following concepts:

* Data versioning with DVC
* Object storage with MinIO
* Machine learning experiment tracking with MLflow
* Pipeline orchestration with Airflow
* Model evaluation
* Automated quality gates
* Model serving with FastAPI
* Interactive ML UI with Gradio
* Containerization with Docker
* API testing with Pytest
* Continuous Integration with GitHub Actions

---

## 20. Project Status

| Component         | Status |
| ----------------- | ------ |
| Machine Learning  | ✅      |
| DVC               | ✅      |
| MinIO             | ✅      |
| MLflow            | ✅      |
| Airflow           | ✅      |
| Quality Gate      | ✅      |
| FastAPI           | ✅      |
| Gradio UI         | ✅      |
| Docker            | ✅      |
| API Tests         | ✅      |
| GitHub Actions CI | ✅      |

The project currently focuses on the **MLOps workflow and CI**, while deployment/CD is outside the current scope.
