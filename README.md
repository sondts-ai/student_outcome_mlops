# Student Outcome Prediction - MLOps System

Dự án triển khai quy trình MLOps đầu cuối (End-to-End MLOps Pipeline) nhằm dự đoán kết quả học tập của sinh viên (Student Outcome / Dropout / Academic Success). Hệ thống tích hợp toàn diện các công cụ chuẩn MLOps: **DVC** quản lý phiên bản dữ liệu/mô hình, **MLflow** theo dõi thí nghiệm, **Apache Airflow** lập lịch và điều phối pipeline, **FastAPI** phục vụ API suy luận, **Streamlit** giao diện người dùng, và đóng gói triển khai bằng **Docker / Docker Compose**.

---

## 1. Kiến trúc Hệ thống (System Architecture)

```
                                  +---------------------------------------+
                                  |            Apache Airflow             |
                                  |     (dags/student_mlops_dag.py)       |
                                  +-------------------+-------------------+
                                                      |
                                                      v Kích hoạt Pipeline
+--------------------+            +---------------------------------------+
|  Dữ liệu thô (DVC) | ---------> |           DVC Pipeline Execution      |
|  data/raw/         |            |         (dvc.yaml / dvc repro)        |
+--------------------+            +-------------------+-------------------+
                                                      |
                    +---------------------------------+---------------------------------+
                    |                                 |                                 |
                    v                                 v                                 v
          [1. Validate Data]               [2. Preprocess & Feature]             [3. Train Model]
         (validate_data.py)               (preprocessing, feat_eng)                 (train.py)
                    |                                 |                                 |
                    +---------------------------------+---------------------------------+
                                                      |
                                                      v
                                           [4. Model Evaluation]
                                              (evaluate.py)
                                                      |
                                                      +------------------------> [MLflow Tracking Server]
                                                      |                          - Log Metrics, Params
                                                      v                          - Model Registry
                                           [5. Quality Gate]
                                           (quality_gate.py)
                                                      |
                                     (Đạt ngưỡng chất lượng?)
                                      /                      \
                                    [Có]                    [Không]
                                     |                         |
                                     v                         v
                           [models/best_model.joblib]     Dừng pipeline & Cảnh báo
                                     |
                                     +---------------------------------+
                                     |                                 |
                                     v                                 v
                          +---------------------+           +---------------------+
                          | FastAPI Backend API | <-------  | gradio UI App    |
                          | (Port 8000)         |           | (Port 8501)         |
                          +---------------------+           +---------------------+
```

### Các thành phần chính:
- **Data & Pipeline Versioning (DVC):** Quản lý phiên bản tập dữ liệu `dataset.csv` và định nghĩa pipeline tự động qua `dvc.yaml` (`validate` -> `preprocess` -> `feature_engineering` -> `train` -> `evaluate` -> `quality_gate`).
- **Experiment Tracking (MLflow):** Ghi lại siêu tham số (hyperparameters), độ đo đánh giá (Accuracy, F1-Score, ROC-AUC), và lưu trữ artifact mô hình qua từng lần chạy.
- **Workflow Orchestration (Airflow):** Tự động hóa việc kích hoạt, theo dõi pipeline huấn luyện định kỳ hoặc theo sự kiện dữ liệu mới.
- **Serving & UI Layer:**
  - **FastAPI (`project/src/api/`):** API chuẩn hóa bằng Pydantic schemas, cung cấp endpoint `/predict` và `/health`.
  - **Streamlit (`project/src/ui/app.py`):** Giao diện tương tác cho người dùng nhập thông tin và nhận kết quả dự đoán trực quan.
- **Containerization (Docker Compose):** Đóng gói toàn bộ dịch vụ (Airflow, MLflow, FastAPI, Streamlit) trong các container độc lập.

---

## 2. Cấu trúc Thư mục

```text
student_outcome_mlops/
├── .github/workflows/ci.yml       # CI/CD pipeline tự động test & lint
├── .dvc/                          # Cấu hình DVC tracking
├── dags/
│   └── student_mlops_dag.py       # DAG Airflow điều phối luồng MLOps
├── data/
│   └── raw/                       # Chứa dữ liệu gốc và file metadata .dvc
├── models/                        # Chứa model xuất bản và metrics
├── project/
│   ├── notebooks/                 # EDA & thực nghiệm baseline ban đầu
│   └── src/                       # Mã nguồn pipeline MLOps
│       ├── api/                   # FastAPI service (main.py, schemas.py)
│       ├── ui/                    # Streamlit frontend (app.py)
│       ├── validate_data.py       # Kiểm định dữ liệu
│       ├── data_preprocessing.py  # Xử lý dữ liệu
│       ├── feature_engineering.py # Trích xuất và biến đổi đặc trưng
│       ├── train.py               # Huấn luyện mô hình & log MLflow
│       ├── evaluate.py            # Đánh giá hiệu năng mô hình
│       ├── quality_gate.py        # Kiểm duyệt ngưỡng chất lượng trước khi release
│       └── run_pipeline.py        # Script chạy toàn bộ pipeline
├── tests/                         # Unit tests và API tests
├── Dockerfile                     # Image cho FastAPI API & Training
├── dockerfile.ui                  # Image cho Streamlit UI
├── docker-compose.yml             # Điều phối các services
├── dvc.yaml & dvc.lock            # Định nghĩa các stage của DVC pipeline
└── requirements.txt               # Danh sách thư viện Python phụ thuộc
```

---

## 3. Cài đặt Môi trường Cục bộ (Local Setup)

### Bước 1: Clone dự án và tạo môi trường ảo
```bash
git clone <repository_url>
cd student_outcome_mlops

# Tạo và kích hoạt môi trường ảo Python (khuyến nghị Python 3.10)
python -m venv venv
source venv/bin/activate  # Trên Linux/macOS
# venv\Scripts\activate   # Trên Windows
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Quản lý Dữ liệu và Pipeline với DVC

### Kéo dữ liệu từ remote storage (hoặc kiểm tra dữ liệu hiện tại):
```bash
# Kéo dữ liệu thô đã được track bởi DVC
dvc pull

# Hoặc nếu bạn thêm tập dữ liệu mới:
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc data/raw/.gitignore
git commit -m "chore: update raw dataset"
```

### Chạy và tái lập Pipeline với DVC:
Pipeline được định nghĩa trong `dvc.yaml`. Để chạy lại toàn bộ hoặc các bước có thay đổi:
```bash
# Thực thi toàn bộ pipeline theo phụ thuộc
dvc repro

# Xem biểu đồ phụ thuộc của pipeline
dvc dag

# Xem metrics sau khi chạy pipeline
dvc metrics show
```

---

## 5. Theo dõi Thí nghiệm với MLflow

### Chạy MLflow Tracking Server cục bộ:
Khởi động máy chủ MLflow để theo dõi metrics, hyperparameters và artifacts:
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```
Truy cập giao diện Web của MLflow tại: **`http://localhost:5000`**

### Cấu hình biến môi trường trước khi chạy Train:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python project/src/run_pipeline.py
```

---

## 6. Điều phối Luồng Tự động với Apache Airflow

DAG điều phối được đặt tại `dags/student_mlops_dag.py`.

### Khởi động Airflow cục bộ (Chế độ Standalone):
```bash
# Thiết lập thư mục làm việc cho Airflow
export AIRFLOW_HOME=$(pwd)/airflow

# Khởi tạo DB và chạy Airflow Standalone
airflow standalone
```
*Truy cập giao diện Airflow tại: **`http://localhost:8080`*** (Tài khoản và mật khẩu hiển thị tại terminal trong lần đầu khởi tạo).

Sau khi đăng nhập:
1. Tìm DAG có tên `student_mlops_pipeline` (hoặc tên DAG định nghĩa trong `dags/student_mlops_dag.py`).
2. Bật toggle **`Unpause`** và nhấn **`Trigger DAG`** để kích hoạt pipeline huấn luyện tự động.

---

## 7. Chạy Ứng dụng bằng Docker Compose (Khuyến nghị)

Để chạy toàn bộ hệ thống gồm API, Web UI, MLflow và Airflow một cách đồng bộ mà không cần cài đặt nhiều môi trường thủ công:

### Bước 1: Build và khởi động các container
```bash
# Khởi chạy toàn bộ hệ sinh thái
docker compose up --build -d
```

### Bước 2: Kiểm tra trạng thái các container
```bash
docker compose ps
```

### Bước 3: Danh sách các cổng dịch vụ

| Dịch vụ | Địa chỉ truy cập | Mô tả |
| :--- | :--- | :--- |
| **gradio UI** | [http://localhost:8501](http://localhost:8501) | Giao diện dự đoán kết quả học tập |
| **FastAPI Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger UI kiểm thử API dự đoán |
| **MLflow UI** | [http://localhost:5000](http://localhost:5000) | Bảng điều khiển quản lý mô hình & metric |
| **Apache Airflow** | [http://localhost:8080](http://localhost:8080) | Giao diện điều phối và lập lịch DAG |

### Bước 4: Tắt hệ thống
```bash
docker compose down
```

---

## 8. Kiểm thử (Testing) & CI/CD

Dự án đi kèm bộ unit test cho pipeline và API phục vụ cho quy trình tích hợp liên tục (CI) qua GitHub Actions (`.github/workflows/ci.yml`).

Chạy test thủ công:
```bash
# Chạy toàn bộ test suites
pytest tests/ -v
```