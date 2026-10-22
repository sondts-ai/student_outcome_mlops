# Student Outcome Prediction — MLOps System

Hệ thống dự đoán kết quả học tập của sinh viên theo ba nhóm: **Dropout (bỏ học), Enrolled (đang học) và Graduate (tốt nghiệp)**. Dự án xây dựng quy trình Machine Learning kết hợp các công cụ MLOps nhằm quản lý dữ liệu, huấn luyện mô hình, đánh giá kết quả, theo dõi thí nghiệm và triển khai mô hình dưới dạng API.

## 1. Mục tiêu dự án

- Xây dựng pipeline Machine Learning có thể tái chạy.
- Quản lý dữ liệu và các artifact bằng DVC.
- Lưu trữ dữ liệu trên Cloudflare R2.
- Theo dõi quá trình huấn luyện và các chỉ số mô hình bằng MLflow.
- Điều phối pipeline bằng Apache Airflow.
- Xây dựng API dự đoán bằng FastAPI.
- Xây dựng giao diện dự đoán bằng Gradio.
- Tích hợp kiểm thử và CI/CD bằng GitHub Actions.
- Đóng gói ứng dụng bằng Docker.

## 2. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ chính |
| Pandas, NumPy | Xử lý dữ liệu |
| Scikit-learn | Huấn luyện và đánh giá mô hình |
| Imbalanced-learn | Xử lý mất cân bằng dữ liệu khi cần |
| DVC | Quản lý phiên bản dữ liệu và pipeline |
| Cloudflare R2 | Lưu trữ dữ liệu và artifact trên cloud |
| MLflow | Theo dõi thí nghiệm Machine Learning |
| Apache Airflow | Điều phối pipeline |
| FastAPI | Xây dựng API dự đoán |
| Gradio | Giao diện người dùng |
| Pytest | Kiểm thử |
| GitHub Actions | Tự động hóa kiểm thử và CI/CD |
| Docker | Đóng gói và triển khai ứng dụng |

## 3. Kiến trúc tổng quan

```text
                 Dataset
                    |
                    v
           Cloudflare R2 / DVC
                    |
                    v
             Data Processing
                    |
                    v
            Feature Engineering
                    |
                    v
              Model Training
                    |
                    v
             Model Evaluation
                    |
                    v
              Saved Model
                    |
          +---------+---------+
          |                   |
          v                   v
      FastAPI API         Gradio UI

Apache Airflow: điều phối các bước của pipeline
MLflow: theo dõi thí nghiệm, tham số và metrics
GitHub Actions: tự động kiểm thử và kiểm tra chất lượng
Docker: đóng gói các thành phần triển khai
```

Sơ đồ trên mô tả vai trò dự kiến của các thành phần. Các bước thực tế được điều phối và triển khai theo mã nguồn trong repository.

## 4. Cấu trúc dự án

```text
student_outcome_mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── dags/
│   └── student_mlops_dag.py
├── data/
│   └── raw/
│       └── dataset.csv.dvc
├── models/
├── reports/
├── scripts/
│   └── download_data.py
├── project/
│   ├── notebooks/
│   └── src/
│       ├── api/
│       │   ├── main.py
│       │   └── schemas.py
│       ├── ui/
│       │   └── app.py
│       ├── validate_data.py
│       ├── data_preprocessing.py
│       ├── feature_engineering.py
│       ├── train.py
│       ├── evaluate.py
│       ├── quality_gate.py
│       ├── predict.py
│       └── run_pipeline.py
├── tests/
│   └── test_api.py
├── .dvc/
│   └── config
├── Dockerfile
├── dockerfile.ui
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── requirements-airflow.txt
└── README.md
```

Lưu ý: cấu trúc trên là cấu trúc tham chiếu. Một số file hoặc thư mục có thể khác với phiên bản hiện tại của repository; hãy đối chiếu trước khi sử dụng các lệnh bên dưới.

## 5. Yêu cầu môi trường

- Git
- Python 3.12.11 cho môi trường hiện tại của dự án
- pip và virtual environment
- DVC
- Tài khoản Cloudflare R2 chỉ cần thiết nếu muốn cấu hình remote riêng tư hoặc tải dữ liệu theo cơ chế xác thực
- Apache Airflow 3.3.2 cho môi trường Airflow hiện tại
- Docker và Docker Compose nếu muốn chạy các dịch vụ bằng Docker

Phiên bản Python 3.12.11 và Airflow 3.3.2 là phiên bản môi trường đang được sử dụng để phát triển; các môi trường khác cần được kiểm thử tương thích.

## 6. Cài đặt dự án

### Bước 1: Clone repository

```bash
git clone https://github.com/sondts-ai/student_outcome_mlops.git
cd student_outcome_mlops
```

### Bước 2: Tạo môi trường Python

Linux, macOS hoặc WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Bước 3: Cài đặt thư viện

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

File `requirements.txt` chứa các thư viện chính phục vụ xử lý dữ liệu, huấn luyện mô hình, DVC, MLflow, API và kiểm thử.

### Bước 4: Chuẩn bị dữ liệu

Dữ liệu được quản lý bằng DVC và lưu trên Cloudflare R2.

Nếu sử dụng script tải dữ liệu công khai:

```bash
python scripts/download_data.py
```

Script sẽ tải dataset vào `data/raw/dataset.csv` và kiểm tra MD5 để xác nhận tính toàn vẹn của file.

Nếu dataset đã có sẵn ở đúng đường dẫn, có thể bỏ qua bước này.

**Lưu ý về quyền truy cập:** URL công khai cho phép người dùng tải dữ liệu mà không cần API key. Không đưa Access Key hoặc Secret Access Key của Cloudflare R2 vào repository. Nếu cần dùng DVC trực tiếp với remote riêng tư, hãy cấu hình thông tin xác thực cục bộ theo hướng dẫn của DVC.

## 7. Quản lý dữ liệu và pipeline bằng DVC

### Kiểm tra trạng thái

```bash
dvc status
```

### Xem sơ đồ pipeline

```bash
dvc dag
```

### Chạy lại pipeline

```bash
dvc repro
```

Pipeline hiện được khai báo trong `dvc.yaml` với hai stage:

**Train**

```yaml
train:
  cmd: python -m project.src.train
  deps:
    - project/src/train.py
    - project/src/data_preprocessing.py
    - project/src/feature_engineering.py
    - data/raw/dataset.csv
  outs:
    - models/best_model.joblib
```

**Evaluate**

```yaml
evaluate:
  cmd: python -m project.src.evaluate
  deps:
    - project/src/evaluate.py
    - project/src/data_preprocessing.py
    - project/src/feature_engineering.py
    - data/raw/dataset.csv
    - models/best_model.joblib
  metrics:
    - reports/metrics.json
```

Stage `train` tạo model tại `models/best_model.joblib`. Stage `evaluate` sử dụng model đó để đánh giá và ghi metrics vào `reports/metrics.json`.

Các đường dẫn và đầu ra cần khớp với code thực tế của hai module.

### Xem metrics

```bash
dvc metrics show
dvc metrics diff
```

## 8. Theo dõi thí nghiệm bằng MLflow

MLflow được sử dụng để theo dõi các thí nghiệm Machine Learning, chẳng hạn như tham số huấn luyện, metrics đánh giá và thông tin model nếu mã nguồn có ghi log các thành phần này.

Khởi chạy giao diện MLflow:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Mở trình duyệt tại:

```text
http://localhost:5000
```

Để xem thông tin thí nghiệm, code huấn luyện cần tích hợp các lệnh logging tương ứng của MLflow. Chỉ khởi chạy giao diện không tự động tạo ra dữ liệu thí nghiệm.

Mặc định, MLflow UI có thể sử dụng backend lưu trữ cục bộ. Nếu muốn chia sẻ dữ liệu tracking giữa nhiều máy hoặc lưu trữ lâu dài, cần cấu hình backend và artifact store phù hợp.

## 9. Điều phối pipeline bằng Apache Airflow

Dự án sử dụng Apache Airflow để điều phối các bước trong pipeline thông qua DAG.

Môi trường hiện tại:

- Apache Airflow: `3.3.2`
- Python: `3.12.11`

### Cài đặt Airflow

Nên cài Airflow trong môi trường Python riêng để tránh xung đột với các thư viện của dự án ML.

Ví dụ trên môi trường Python 3.12:

```bash
python -m pip install "apache-airflow==3.3.2" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.12.txt"
```

Nếu Airflow đã được cài và hoạt động trong môi trường hiện tại, không cần cài lại.

### Khởi chạy Airflow

```bash
airflow standalone
```

Sau khi khởi động, truy cập:

```text
http://localhost:8080
```

Thông tin đăng nhập được tạo hoặc hiển thị trong quá trình khởi tạo Airflow. Không lưu mật khẩu thật trong README hoặc repository công khai.

### Cấu hình DAG

DAG dự kiến nằm trong:

```text
dags/student_mlops_dag.py
```

Cần bảo đảm DAG gọi đúng các bước trong pipeline, sử dụng đúng đường dẫn dự án và chạy trong môi trường có đủ thư viện, dataset và model cần thiết.

Airflow `standalone` phù hợp cho phát triển và thử nghiệm cục bộ, không phải cấu hình triển khai production.

## 10. API dự đoán bằng FastAPI

Nếu API đã được triển khai trong `project/src/api/main.py`, có thể khởi chạy bằng:

```bash
uvicorn project.src.api.main:app --host 127.0.0.1 --port 8000
```

Mở tài liệu API tại:

```text
http://localhost:8000/docs
```

API cần nạp model đã huấn luyện và xác thực dữ liệu đầu vào theo schema được định nghĩa trong code.

Để kiểm thử API:

```bash
pytest
```

Cấu trúc JSON đầu vào và endpoint dự đoán cần tuân theo implementation thực tế của `main.py` và `schemas.py`.

## 11. Giao diện dự đoán bằng Gradio

Nếu giao diện được triển khai tại `project/src/ui/app.py`, chạy:

```bash
python project/src/ui/app.py
```

Địa chỉ truy cập phụ thuộc vào cấu hình trong ứng dụng Gradio; thông thường giao diện chạy trên cổng `7860`.

## 12. Kiểm thử và CI/CD

Dự án có thể sử dụng GitHub Actions để tự động hóa các bước:

- Kiểm tra lỗi cú pháp và chất lượng code.
- Chạy unit test bằng Pytest.
- Kiểm tra API.
- Kiểm tra điều kiện chất lượng mô hình nếu workflow đã tích hợp quality gate.

Workflow tham chiếu:

```text
.github/workflows/ci.yml
```

Sau khi cập nhật code, có thể chạy kiểm thử cục bộ:

```bash
pytest
```

Các bước thực tế của CI phụ thuộc vào file workflow hiện có trong repository.

## 13. Docker

Nếu `Dockerfile`, `dockerfile.ui` và `docker-compose.yml` đã được cấu hình đầy đủ, có thể thử khởi chạy các dịch vụ bằng:

```bash
docker compose up -d --build
```

Kiểm tra container:

```bash
docker compose ps
docker compose logs -f
```

Dừng các dịch vụ:

```bash
docker compose down
```

Các cổng, biến môi trường và dịch vụ được khởi chạy phụ thuộc vào nội dung `docker-compose.yml`. Cần kiểm tra file này trước khi sử dụng các lệnh trên.

## 14. Bảo mật và quản lý thông tin xác thực

- Không commit Access Key hoặc Secret Access Key của Cloudflare R2.
- Không commit mật khẩu Airflow, token API hoặc các file `.env` chứa thông tin nhạy cảm.
- Không đưa thông tin xác thực vào README.
- Nếu sử dụng R2 công khai, chỉ lưu những dữ liệu được phép công khai.
- Nếu cần giữ dataset riêng tư, sử dụng cơ chế xác thực thay vì URL công khai.
- Không sử dụng cấu hình phát triển như Airflow `standalone` hoặc API bind toàn bộ network interface cho môi trường production mà chưa có biện pháp bảo vệ phù hợp.

## 15. Hướng phát triển

Các hướng phát triển có thể cân nhắc:

- Khóa phiên bản thư viện để tái lập môi trường.
- Bổ sung kiểm tra chất lượng dữ liệu trước khi train.
- Hoàn thiện logging và tracking cho MLflow.
- Hoàn thiện DAG và cơ chế retry trong Airflow.
- Bổ sung kiểm thử tích hợp cho pipeline.
- Triển khai API bằng Docker.
- Cấu hình lưu trữ artifact và tracking tập trung.
- Bổ sung giám sát chất lượng dự đoán khi triển khai thực tế.

## 16. License

Bổ sung thông tin license phù hợp trước khi công bố hoặc cho phép người khác sử dụng lại mã nguồn.

---

**Ghi chú:** README mô tả mục tiêu và cách chạy dự án. Trước khi phát hành, hãy xác nhận cấu trúc file, lệnh chạy, DAG Airflow, các endpoint API và cấu hình Docker thực tế khớp với nội dung tài liệu.
