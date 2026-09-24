
import requests
import pandas as pd
import gradio as gr


API_URL = "http://127.0.0.1:8005"


# =========================================================
# Check API
# =========================================================

def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=10,
        )
        response.raise_for_status()
        return "🟢 API is running"

    except requests.RequestException:
        return "🔴 API is not available"


# =========================================================
# Predict CSV
# =========================================================


def predict_csv(file):
    if file is None:
        return None, "❌ Please upload a CSV file."

    try:
        df = pd.read_csv(file.name)

        if df.empty:
            return None, "❌ CSV file is empty."

        # CSV column -> API field
        column_mapping = {
            "Age at enrollment": "age_at_enrollment",
            "Application order": "application_order",

            "Curricular units 1st sem (credited)": "curricular_units_1st_sem_credited",
            "Curricular units 1st sem (enrolled)": "curricular_units_1st_sem_enrolled",
            "Curricular units 1st sem (evaluations)": "curricular_units_1st_sem_evaluations",
            "Curricular units 1st sem (approved)": "curricular_units_1st_sem_approved",
            "Curricular units 1st sem (grade)": "curricular_units_1st_sem_grade",
            "Curricular units 1st sem (without evaluations)": "curricular_units_1st_sem_without_evaluations",

            "Curricular units 2nd sem (credited)": "curricular_units_2nd_sem_credited",
            "Curricular units 2nd sem (enrolled)": "curricular_units_2nd_sem_enrolled",
            "Curricular units 2nd sem (evaluations)": "curricular_units_2nd_sem_evaluations",
            "Curricular units 2nd sem (approved)": "curricular_units_2nd_sem_approved",
            "Curricular units 2nd sem (grade)": "curricular_units_2nd_sem_grade",
            "Curricular units 2nd sem (without evaluations)": "curricular_units_2nd_sem_without_evaluations",

            "Unemployment rate": "unemployment_rate",
            "Inflation rate": "inflation_rate",
            "GDP": "gdp",

            "Marital status": "marital_status",
            "Application mode": "application_mode",
            "Course": "course",
            "Daytime/evening attendance": "daytime_evening_attendance",
            "Previous qualification": "previous_qualification",
            "Nacionality": "nacionality",
            "Mother's qualification": "mothers_qualification",
            "Father's qualification": "fathers_qualification",
            "Mother's occupation": "mothers_occupation",
            "Father's occupation": "fathers_occupation",

            "Displaced": "displaced",
            "Educational special needs": "educational_special_needs",
            "Debtor": "debtor",
            "Tuition fees up to date": "tuition_fees_up_to_date",
            "Gender": "gender",
            "Scholarship holder": "scholarship_holder",
            "International": "international",
        }

        # Take the first student
        row = df.iloc[0]

        # Build request body
        student_data = {}

        for csv_column, api_field in column_mapping.items():
            student_data[api_field] = row[csv_column]

        # Convert numpy values to normal Python values
        student_data = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in student_data.items()
        }

        print("Sending to API:")
        print(student_data)

        # Send ONE student
        response = requests.post(
            f"{API_URL}/predict",
            json=student_data,
            timeout=30,
        )

        print("API response:", response.text)

        if response.status_code == 422:
            return None, f"❌ Validation error: {response.text}"

        response.raise_for_status()

        prediction = response.json()

        # Convert result to DataFrame for display
        result_df = pd.DataFrame([
        {
        "Prediction": prediction["prediction"],
        "Dropout Probability": prediction["probabilities"]["Dropout"],
        "Enrolled Probability": prediction["probabilities"]["Enrolled"],
        "Graduate Probability": prediction["probabilities"]["Graduate"],
        }
        ])

        return (
            result_df,
            "🟢 Prediction successful."
        )

    except requests.RequestException as e:
        return None, f"❌ API request failed: {str(e)}"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"



# =========================================================
# UI
# =========================================================

with gr.Blocks(
    title="Student Success Prediction"
) as app:

    gr.Markdown(
        """
        # 🎓 Student Success Prediction

        Upload a CSV file containing student data
        and click **Predict**.
        """
    )

    # API status
    with gr.Row():

        api_status = gr.Textbox(
            label="API Status",
            value=check_api(),
            interactive=False,
        )

        check_button = gr.Button("Check API")

    check_button.click(
        fn=check_api,
        outputs=api_status,
    )

    # CSV upload
    gr.Markdown("## 📂 Upload Student Data")

    csv_file = gr.File(
        label="Upload CSV",
        file_types=[".csv"],
    )

    # Predict button
    predict_button = gr.Button(
        "Predict",
        variant="primary",
    )

    # Status
    status = gr.Textbox(
        label="Status",
        interactive=False,
    )

    # Result
    result = gr.Dataframe(
        label="Prediction Results",
        interactive=False,
    )

    predict_button.click(
        fn=predict_csv,
        inputs=csv_file,
        outputs=[
            result,
            status,
        ],
    )


# =========================================================
# Launch
# =========================================================

if __name__ == "__main__":
    app.launch()

