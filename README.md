# 🎓 Student Outcome Prediction - MLOps System

Dự án triển khai quy trình MLOps đầu-cuối (End-to-End MLOps Pipeline) cho bài toán dự đoán kết quả học tập của sinh viên (Student Outcome: Dropout / Enrolled / Graduate). Hệ thống tích hợp toàn diện từ quản lý phiên bản dữ liệu/mô hình (DVC), tự động hóa kiểm thử và triển khai liên tục (GitHub Actions CI/CD), quản lý thí nghiệm (MLflow), điều phối pipeline (Apache Airflow), cho đến đóng gói container (Docker) và phục vụ người dùng thông qua FastAPI kết hợp Gradio UI.

---

## 🏗️ 1. Kiến trúc Hệ thống (System Architecture)

```
                                +-----------------------------------+
                                |        Developer / Git Repo       |
                                +-----------------+-----------------+
                                                  |
                                                  v (git push / PR)
                                +-----------------------------------+
                                |     CI/CD: GitHub Actions         |
                                | - Linting & Code Style Checks     |
                                | - Pytest Unit / API Testing       |
                                | - Model Quality Gate Inspection   |
                                | - Docker Images Build & Push      |
                                +-----------------+-----------------+
                                                  |
                                                  v (Kích hoạt điều phối)
+-----------------------------------------------------------------------------------+
|                        MLOps Orchestration & Experiment Layer                     |
|                                                                                   |
|  [DVC: Data & Pipeline Versioning]                                                |
|   └── dataset.csv.dvc -> Validation -> Preprocessing -> Feature Eng -> Train      |
|                                                                                   |
|  [Apache Airflow: Workflow Orchestration]                                         |
|   └── dags/student_mlops_dag.py (Lập lịch kiểm định, huấn luyện & Quality Gate)   |
|                                                                                   |
|  [MLflow: Tracking & Model Registry]                                              |
|   └── Ghi log Parameters, Metrics (Accuracy, F1), Artifacts & Quản lý Best Model  |
+-------------------------------------------------+---------------------------------+
                                                  |
                                                  v (Artifacts: best_model.joblib)
+-----------------------------------------------------------------------------------+
|                     Triển khai Dịch vụ (Docker & Docker Compose)                  |
|                                                                                   |
|      +------------------------------+         +------------------------------+    |
|      |    FastAPI Backend           | <====== |    Gradio Web UI Frontend    |    |
|      |    - Endpoint: /predict      |  REST   |    - Interactive Inputs      |    |
|      |    - Port: 8000              |   API   |    - Port: 7860              |    |
|      +------------------------------+         +------------------------------+    |
+-----------------------------------------------------------------------------------+
```

### Các thành phần chính:
1. **GitHub Actions (CI/CD)**: Tự động chạy kiểm thử đơn vị (`pytest tests/`), kiểm tra cú pháp code, rà soát ngưỡng chất lượng mô hình (Quality Gate) và build image khi có thay đổi trong repository.
2. **DVC (Data Version Control)**: Quản lý phiên bản dữ liệu thô lớn (`data/raw/dataset.csv.dvc`) và duy trì đường ống xử lý nhiều giai đoạn thông qua file cấu hình `dvc.yaml` & `dvc.lock`.
3. **MLflow Tracking**: Theo dõi lịch sử huấn luyện, các siêu tham số, bảng chỉ số đánh giá (accuracy, precision, recall, f1-score) và lưu trữ artifacts mô hình.
4. **Apache Airflow**: Lập lịch trình và tự động kích hoạt workflow huấn luyện lại mô hình theo chu kỳ hoặc sự kiện dữ liệu mới (`student_mlops_dag.py`).
5. **FastAPI & Gradio**: 
   - **FastAPI**: REST API hiệu năng cao với Pydantic schema validation phục vụ suy luận (`/predict`).
   - **Gradio**: Giao diện web trực quan, thân thiện cho phép nhập thông tin sinh viên và hiển thị kết quả dự đoán ngay lập tức.
6. **Docker & Docker Compose**: Đóng gói cô lập các dịch vụ, đảm bảo tính nhất quán giữa môi trường phát triển và môi trường triển khai thực tế.

---

## 📁 2. Cấu trúc Thư mục

```text
student_outcome_mlops/
├── .dvc/                        # Cấu hình DVC nội bộ
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline tự động kiểm thử và build CI/CD
├── dags/
│   └── student_mlops_dag.py     # DAG Airflow điều phối luồng MLOps
├── data/
│   └── raw/
│       ├── dataset.csv.dvc      # Tracking tập dữ liệu gốc bằng DVC
│       └── .gitignore
├── models/                      # Chứa artifact mô hình xuất xưởng và metrics.json
├── project/
│   ├── notebooks/               # Notebooks EDA và thí nghiệm mô hình baseline
│   └── src/
│       ├── api/                 # Mã nguồn FastAPI service (main.py, schemas.py)
│       ├── ui/                  # Giao diện Gradio Web UI (app.py)
│       ├── validate_data.py     # Kiểm tra tính toàn vẹn và schema dữ liệu
│       ├── data_preprocessing.py# Tiền xử lý dữ liệu
│       ├── feature_engineering.py# Trích xuất và chuẩn hóa đặc trưng
│       ├── train.py             # Huấn luyện mô hình và tích hợp log MLflow
│       ├── evaluate.py          # Đánh giá hiệu năng mô hình
│       ├── quality_gate.py      # Kiểm tra điều kiện chất lượng trước khi release
│       ├── predict.py           # Logic load model và suy luận kết quả
│       └── run_pipeline.py      # Script chạy tuần tự toàn bộ pipeline cục bộ
├── tests/
│   └── test_api.py              # Test cases kiểm thử API
├── Dockerfile                   # Dockerfile cho FastAPI API service
├── dockerfile.ui                # Dockerfile cho Gradio UI service
├── docker-compose.yml           # Khởi chạy cụm dịch vụ FastAPI và Gradio
├── dvc.yaml                     # Định nghĩa các stage của DVC pipeline
├── dvc.lock                     # Khóa trạng thái các stage pipeline
└── requirements.txt             # Danh sách thư viện Python phụ thuộc
```

---

## ⚙️ 3. Cài đặt Môi trường Cục bộ (Local Setup)

### Yêu cầu tiên quyết:
- Python 3.10 trở lên
- Git & Git CLI
- Docker & Docker Compose

### Các bước cài đặt:
```bash
# 1. Clone repository
git clone https://github.com/sondts-ai/student_outcome_mlops.git
cd student_outcome_mlops

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv

# Trên Linux / macOS:
source venv/bin/activate
# Trên Windows:
venv\Scripts\activate

# 3. Cài đặt toàn bộ thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔄 4. Hướng dẫn Chạy Từng Thành phần

### 4.1. Quản lý Dữ liệu và Pipeline với DVC
DVC quản lý các bước tiền xử lý, trích xuất đặc trưng và huấn luyện mô hình thông qua file `dvc.yaml`:

```bash
# Kéo dữ liệu đã được track (nếu có cấu hình remote storage)
dvc pull

# Thực thi toàn bộ pipeline theo thứ tự phụ thuộc
dvc repro

# Kiểm tra sự thay đổi của metrics sau khi huấn luyện
dvc metrics show
dvc metrics diff
```

### 4.2. Theo dõi Thí nghiệm với MLflow
Mã nguồn trong `project/src/train.py` đã tích hợp theo dõi tham số và metric vào MLflow:

```bash
# Khởi động MLflow UI Server cục bộ
mlflow ui --port 5000
```
👉 Truy cập giao diện trực quan tại: **`http://localhost:5000`** để xem biểu đồ loss/accuracy, các tham số và artifact mô hình.

### 4.3. Tự động hóa CI/CD với GitHub Actions
Quy trình CI/CD được định nghĩa tại `.github/workflows/ci.yml`. Khi bạn tạo commit hoặc mở Pull Request:
1. GitHub Actions sẽ tự động khởi tạo môi trường Python và cài đặt `requirements.txt`.
2. Chạy linter kiểm tra chuẩn mã nguồn.
3. Chạy toàn bộ test suites tự động:
   ```bash
   # Lệnh chạy kiểm thử cục bộ:
   pytest tests/test_api.py -v
   ```
4. Kiểm tra điều kiện chất lượng mô hình qua bước `quality_gate.py`.

### 4.4. Điều phối Quy trình với Apache Airflow
File DAG `dags/student_mlops_dag.py` tự động hóa các tác vụ huấn luyện định kỳ:

```bash
# Cấu hình thư mục làm việc cho Airflow
export AIRFLOW_HOME=$(pwd)/airflow

# Khởi tạo cơ sở dữ liệu Airflow
airflow db init

# Tạo tài khoản quản trị Admin
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Khởi chạy Scheduler và Webserver
airflow scheduler &
airflow webserver --port 8080
```
👉 Truy cập: **`http://localhost:8080`** (Tài khoản: `admin` / `admin`), tìm kiếm `student_mlops_dag`, bật toggle **Unpause** và bấm **Trigger DAG**.

### 4.5. Chạy Trực tiếp FastAPI & Gradio UI (Không dùng Docker)
Nếu muốn phát triển và kiểm tra trực tiếp trên máy:
```bash
# Terminal 1: Chạy FastAPI backend
uvicorn project.src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Chạy Gradio frontend
python project/src/ui/app.py
```
👉 Truy cập Gradio UI tại: **`http://localhost:7860`**

---

## 🐳 5. Triển khai Hệ thống với Docker & Docker Compose

Cách nhanh nhất để chạy toàn bộ hệ thống (FastAPI Backend + Gradio UI) là sử dụng Docker Compose:

```bash
# 1. Build image và khởi động các container ở chế độ chạy ngầm
docker-compose up -d --build

# 2. Kiểm tra trạng thái hoạt động của các container
docker-compose ps

# 3. Xem nhật ký log thời gian thực
docker-compose logs -f
```

### Danh sách Cổng Dịch vụ:
| Dịch vụ | Đường dẫn truy cập | Mục đích |
| :--- | :--- | :--- |
| **Gradio Web UI** | `http://localhost:7860` | Giao diện nhập thông tin sinh viên và nhận kết quả dự đoán |
| **FastAPI Backend** | `http://localhost:8000` | REST API nhận payload và suy luận mô hình |
| **Swagger API Docs** | `http://localhost:8000/docs` | Tài liệu OpenAPI tương tác trực tiếp |
| **MLflow Server** (nếu bật) | `http://localhost:5000` | Quản lý thí nghiệm và Model Registry |
| **Apache Airflow** (nếu bật) | `http://localhost:8080` | Quản lý và lập lịch DAG điều phối |

Dừng và thu hồi tài nguyên các container:
```bash
docker-compose down
```

---

## 🧪 6. Kiểm thử Endpoint API (API Testing)

Gửi request kiểm tra kết quả dự đoán tới API qua lệnh `curl`:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "Marital_status": 1,
    "Application_mode": 1,
    "Application_order": 1,
    "Course": 9254,
    "Daytime_evening_attendance": 1,
    "Previous_qualification": 1,
    "Nacionality": 1,
    "Mother_qualification": 1,
    "Father_qualification": 3,
    "Mother_occupation": 5,
    "Father_occupation": 3,
    "Displaced": 1,
    "Educational_special_needs": 0,
    "Debtor": 0,
    "Tuition_fees_up_to_date": 1,
    "Gender": 1,
    "Scholarship_holder": 0,
    "Age_at_enrollment": 20,
    "International": 0,
    "Curricular_units_1st_sem_credited": 0,
    "Curricular_units_1st_sem_enrolled": 0,
    "Curricular_units_1st_sem_evaluations": 0,
    "Curricular_units_1st_sem_approved": 0,
    "Curricular_units_1st_sem_grade": 0.0,
    "Curricular_units_1st_sem_without_evaluations": 0,
    "Curricular_units_2nd_sem_credited": 0,
    "Curricular_units_2nd_sem_enrolled": 0,
    "Curricular_units_2nd_sem_evaluations": 0,
    "Curricular_units_2nd_sem_approved": 0,
    "Curricular_units_2nd_sem_grade": 0.0,
    "Curricular_units_2nd_sem_without_evaluations": 0,
    "Unemployment_rate": 10.8,
    "Inflation_rate": 1.4,
    "GDP": 1.74
}'
```