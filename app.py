from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load(
    "model/student_performance_model.pkl"
)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "gender": request.form["gender"],
        "age": int(request.form["age"]),
        "study_hours": float(request.form["study_hours"]),
        "attendance": float(request.form["attendance"]),
        "previous_score": float(request.form["previous_score"]),
        "sleep_hours": float(request.form["sleep_hours"]),
        "assignments_completed": int(
            request.form["assignments_completed"]
        ),
        "internet_access": request.form["internet_access"],
        "parent_education": request.form["parent_education"],
        "extracurricular": request.form["extracurricular"]
    }

    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)