FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY project ./project
COPY models ./models

CMD ["uvicorn", "project.src.api.main:app", "--host", "0.0.0.0", "--port", "8005"]