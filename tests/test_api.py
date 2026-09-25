from fastapi.testclient import TestClient

from project.src.api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    payload = {
        "age_at_enrollment": 20,
        "application_order": 1,

        "curricular_units_1st_sem_credited": 0,
        "curricular_units_1st_sem_enrolled": 6,
        "curricular_units_1st_sem_evaluations": 6,
        "curricular_units_1st_sem_approved": 5,
        "curricular_units_1st_sem_grade": 14.0,
        "curricular_units_1st_sem_without_evaluations": 0,

        "curricular_units_2nd_sem_credited": 0,
        "curricular_units_2nd_sem_enrolled": 6,
        "curricular_units_2nd_sem_evaluations": 6,
        "curricular_units_2nd_sem_approved": 5,
        "curricular_units_2nd_sem_grade": 14.0,
        "curricular_units_2nd_sem_without_evaluations": 0,

        "unemployment_rate": 10.0,
        "inflation_rate": 2.0,
        "gdp": 1.5,

        "marital_status": 1,
        "application_mode": 1,
        "course": 1,
        "daytime_evening_attendance": 1,
        "previous_qualification": 1,
        "nacionality": 1,
        "mothers_qualification": 1,
        "fathers_qualification": 1,
        "mothers_occupation": 1,
        "fathers_occupation": 1,

        "displaced": 0,
        "educational_special_needs": 0,
        "debtor": 0,
        "tuition_fees_up_to_date": 1,
        "gender": 1,
        "scholarship_holder": 0,
        "international": 0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() is not None
    
def test_predict_invalid_input():
    payload = {
        "age_at_enrollment": "twenty",
        "application_order": 1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422