from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

model = joblib.load("models/lung_cancer_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

HISTORY_FILE = "prediction_history.csv"

def get_local_explanation(result, probability, age, gender, positive_features):
    
    gender_text = "male" if gender == "M" else "female"

    if positive_features:
        feature_text = ", ".join(positive_features)

        feature_message = (
            f"The input contains the following features marked as Yes: "
            f"{feature_text}."
        )

    else:
        feature_message = (
            "None of the listed symptom or risk-factor fields were marked Yes."
        )

    if probability >= 70:

        probability_message = (
            "The model produced a relatively high probability for the YES class "
            "for this particular input."
        )

    elif probability >= 40:

        probability_message = (
            "The model produced an intermediate probability for the YES class "
            "for this particular input."
        )

    else:

        probability_message = (
            "The model produced a relatively low probability for the YES class "
            "for this particular input."
        )

    explanation = (
        f"The Random Forest model generated the prediction: {result}. "
        f"The model's probability for the YES class is {probability:.2f}%. "
        "This percentage represents the model output for this input and is "
        "not a medical risk percentage. "
        f"The input information provided was for a {gender_text} aged {age}. "
        f"{feature_message} "
        f"{probability_message} "
        "This is an educational machine learning prediction and must not be "
        "used as a medical diagnosis. "
        "For real medical concerns, consult a qualified healthcare professional."
    )

    return explanation

def save_prediction(age, gender, result, probability):

    new_record = pd.DataFrame(
        [
            {
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Age": age,
                "Gender": gender,
                "Prediction": result,
                "Probability": f"{probability:.2f}%"
            }
        ]
    )

    if os.path.exists(HISTORY_FILE):

        try:
            history = pd.read_csv(HISTORY_FILE)

        except pd.errors.EmptyDataError:

            history = pd.DataFrame()

    else:

        history = pd.DataFrame()

        history = pd.concat(
        [history, new_record],
        ignore_index=True
    )

    history.to_csv(
        HISTORY_FILE,
        index=False
    )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    gender = request.form["gender"]

    age = int(request.form["age"])

    smoking = int(request.form["smoking"])
    yellow_fingers = int(request.form["yellow_fingers"])
    anxiety = int(request.form["anxiety"])
    peer_pressure = int(request.form["peer_pressure"])
    chronic_disease = int(request.form["chronic_disease"])
    fatigue = int(request.form["fatigue"])
    allergy = int(request.form["allergy"])
    wheezing = int(request.form["wheezing"])
    alcohol_consuming = int(request.form["alcohol_consuming"])
    coughing = int(request.form["coughing"])
    shortness_of_breath = int(request.form["shortness_of_breath"])
    swallowing_difficulty = int(request.form["swallowing_difficulty"])
    chest_pain = int(request.form["chest_pain"])

    gender_value = 1 if gender == "M" else 0

    patient_data = pd.DataFrame(
        [[
            gender_value,
            age,
            smoking,
            yellow_fingers,
            anxiety,
            peer_pressure,
            chronic_disease,
            fatigue,
            allergy,
            wheezing,
            alcohol_consuming,
            coughing,
            shortness_of_breath,
            swallowing_difficulty,
            chest_pain
        ]],
        columns=feature_names
    )

    prediction = model.predict(patient_data)[0]

    probability = model.predict_proba(patient_data)[0][1] * 100

    if prediction == 1:

        result = "LUNG CANCER - YES"

    else:

        result = "LUNG CANCER - NO"


    symptoms = {
        "Smoking": smoking,
        "Yellow Fingers": yellow_fingers,
        "Anxiety": anxiety,
        "Peer Pressure": peer_pressure,
        "Chronic Disease": chronic_disease,
        "Fatigue": fatigue,
        "Allergy": allergy,
        "Wheezing": wheezing,
        "Alcohol Consuming": alcohol_consuming,
        "Coughing": coughing,
        "Shortness of Breath": shortness_of_breath,
        "Swallowing Difficulty": swallowing_difficulty,
        "Chest Pain": chest_pain
    }


    positive_features = [
        name
        for name, value in symptoms.items()
        if value == 2
    ]


    explanation = get_local_explanation(
        result,
        probability,
        age,
        gender,
        positive_features
    )


    save_prediction(
        age,
        gender,
        result,
        probability
    )


    patient_summary = {
        "Gender": "Male" if gender == "M" else "Female",
        "Age": age,
        "Smoking": "Yes" if smoking == 2 else "No",
        "Yellow Fingers": "Yes" if yellow_fingers == 2 else "No",
        "Anxiety": "Yes" if anxiety == 2 else "No",
        "Peer Pressure": "Yes" if peer_pressure == 2 else "No",
        "Chronic Disease": "Yes" if chronic_disease == 2 else "No",
        "Fatigue": "Yes" if fatigue == 2 else "No",
        "Allergy": "Yes" if allergy == 2 else "No",
        "Wheezing": "Yes" if wheezing == 2 else "No",
        "Alcohol Consuming": "Yes" if alcohol_consuming == 2 else "No",
        "Coughing": "Yes" if coughing == 2 else "No",
        "Shortness of Breath": "Yes" if shortness_of_breath == 2 else "No",
        "Swallowing Difficulty": "Yes" if swallowing_difficulty == 2 else "No",
        "Chest Pain": "Yes" if chest_pain == 2 else "No"
    }


    return render_template(
        "index.html",
        result=result,
        probability=f"{probability:.2f}",
        explanation=explanation,
        patient_summary=patient_summary
    )



@app.route("/history")
def history():

    if os.path.exists(HISTORY_FILE):

        try:

            history_data = pd.read_csv(HISTORY_FILE)

            records = history_data.to_dict(
                orient="records"
            )

        except pd.errors.EmptyDataError:

            records = []

    else:

        records = []

    return render_template(
        "history.html",
        records=records
    )

@app.route("/clear-history", methods=["POST"])
def clear_history():

    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

    return render_template(
        "history.html",
        records=[]
    )

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/performance")
def performance():
    metrics = {
        "Accuracy": "92.86%",
        "Precision": "97.83%",
        "Recall": "93.75%",
        "F1 Score": "95.74%",
        "ROC-AUC": "96.35%"
    }

    confusion_matrix = {
        "True Negative": 7,
        "False Positive": 1,
        "False Negative": 3,
        "True Positive": 45
    }

    return render_template(
        "performance.html",
        metrics=metrics,
        confusion_matrix=confusion_matrix
    )


@app.route("/feature-importance")
def feature_importance():

    importance_data = [
        {"feature": "AGE", "importance": 0.157524},
        {"feature": "ALLERGY", "importance": 0.109562},
        {"feature": "ALCOHOL CONSUMING", "importance": 0.096349},
        {"feature": "PEER PRESSURE", "importance": 0.074473},
        {"feature": "SWALLOWING DIFFICULTY", "importance": 0.070578},
        {"feature": "CHRONIC DISEASE", "importance": 0.070348},
        {"feature": "FATIGUE", "importance": 0.069986},
        {"feature": "WHEEZING", "importance": 0.055073},
        {"feature": "YELLOW FINGERS", "importance": 0.054103},
        {"feature": "CHEST PAIN", "importance": 0.047661},
        {"feature": "ANXIETY", "importance": 0.046249},
        {"feature": "COUGHING", "importance": 0.046092},
        {"feature": "SHORTNESS OF BREATH", "importance": 0.042731},
        {"feature": "GENDER", "importance": 0.030313},
        {"feature": "SMOKING", "importance": 0.028958}
    ]

    chart_labels = [
        item["feature"]
        for item in importance_data
    ]

    chart_values = [
        item["importance"]
        for item in importance_data
    ]

    return render_template(
        "feature_importance.html",
        importance_data=importance_data,
        chart_labels=chart_labels,
        chart_values=chart_values
    )


if __name__ == "__main__":
    app.run(debug=True)