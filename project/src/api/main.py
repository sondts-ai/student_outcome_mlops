from fastapi import FastAPI
from project.src.api.schemas import StudentInput
from project.src.predict import predict_sample



app=FastAPI()

FIELD_MAPPING = {
    "age_at_enrollment": "Age at enrollment",
    "application_order": "Application order",

    "curricular_units_1st_sem_credited": "Curricular units 1st sem (credited)",
    "curricular_units_1st_sem_enrolled": "Curricular units 1st sem (enrolled)",
    "curricular_units_1st_sem_evaluations": "Curricular units 1st sem (evaluations)",
    "curricular_units_1st_sem_approved": "Curricular units 1st sem (approved)",
    "curricular_units_1st_sem_grade": "Curricular units 1st sem (grade)",
    "curricular_units_1st_sem_without_evaluations": "Curricular units 1st sem (without evaluations)",

    "curricular_units_2nd_sem_credited": "Curricular units 2nd sem (credited)",
    "curricular_units_2nd_sem_enrolled": "Curricular units 2nd sem (enrolled)",
    "curricular_units_2nd_sem_evaluations": "Curricular units 2nd sem (evaluations)",
    "curricular_units_2nd_sem_approved": "Curricular units 2nd sem (approved)",
    "curricular_units_2nd_sem_grade": "Curricular units 2nd sem (grade)",
    "curricular_units_2nd_sem_without_evaluations": "Curricular units 2nd sem (without evaluations)",

    "unemployment_rate": "Unemployment rate",
    "inflation_rate": "Inflation rate",
    "gdp": "GDP",

    "marital_status": "Marital status",
    "application_mode": "Application mode",
    "course": "Course",
    "daytime_evening_attendance": "Daytime/evening attendance",
    "previous_qualification": "Previous qualification",
    "nacionality": "Nacionality",
    "mothers_qualification": "Mother's qualification",
    "fathers_qualification": "Father's qualification",
    "mothers_occupation": "Mother's occupation",
    "fathers_occupation": "Father's occupation",

    "displaced": "Displaced",
    "educational_special_needs": "Educational special needs",
    "debtor": "Debtor",
    "tuition_fees_up_to_date": "Tuition fees up to date",
    "gender": "Gender",
    "scholarship_holder": "Scholarship holder",
    "international": "International",
}


@app.get("/heath")
def heath():
    return {"status":"ok"}


@app.post("/predict")
def predict(student:StudentInput):
    data=student.model_dump()
    sample={
        FIELD_MAPPING[key]:value
        for key,value in data.items()
    }
    return predict_sample(sample)


