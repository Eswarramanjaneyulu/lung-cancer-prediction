import joblib
import pandas as pd

# Load saved model

model = joblib.load("models/lung_cancer_model.pkl")

# Load feature names

feature_names = joblib.load("models/feature_names.pkl")

def get_local_explanation(result, probability, age, gender, patient_data):
    explanation = []


    if result == "LUNG CANCER - YES":
        explanation.append(
            "The Random Forest model classified this input as LUNG CANCER - YES."
        )
    else:
        explanation.append(
            "The Random Forest model classified this input as LUNG CANCER - NO."
        )

    explanation.append(
        f"The model probability for YES is {probability:.2f}%."
    )

    explanation.append(
        "This probability is only the output of the trained machine learning model "
        "and is not a medical risk percentage."
    )

    positive_features = []

    feature_labels = {
        "SMOKING": "Smoking",
        "YELLOW FINGERS": "Yellow Fingers",
        "ANXIETY": "Anxiety",
        "PEER_PRESSURE": "Peer Pressure",
        "CHRONIC DISEASE": "Chronic Disease",
        "FATIGUE": "Fatigue",
        "ALLERGY": "Allergy",
        "WHEEZING": "Wheezing",
        "ALCOHOL CONSUMING": "Alcohol Consumption",
        "COUGHING": "Coughing",
        "SHORTNESS OF BREATH": "Shortness of Breath",
        "SWALLOWING DIFFICULTY": "Swallowing Difficulty",
        "CHEST PAIN": "Chest Pain"
    }

    for feature in feature_names:
        if feature in feature_labels and patient_data[feature].iloc[0] == 2:
            positive_features.append(feature_labels[feature])

    if positive_features:
        explanation.append(
            "The input contains the following selected factors: "
            + ", ".join(positive_features)
            + "."
        )
    else:
        explanation.append(
            "No symptom or risk-factor inputs were selected as YES."
        )

    explanation.append(
        "This is an educational machine learning project and cannot diagnose "
        "lung cancer."
    )

    explanation.append(
        "For real medical concerns, please consult a qualified healthcare professional."
    )

    return " ".join(explanation)


print("\n===================================")
print("   LUNG CANCER PREDICTION SYSTEM")
print("===================================")

gender = input("Gender (M/F): ").upper()

age = int(input("Age: "))

smoking = int(input("Smoking (1=No, 2=Yes): "))
yellow_fingers = int(input("Yellow Fingers (1=No, 2=Yes): "))
anxiety = int(input("Anxiety (1=No, 2=Yes): "))
peer_pressure = int(input("Peer Pressure (1=No, 2=Yes): "))
chronic_disease = int(input("Chronic Disease (1=No, 2=Yes): "))
fatigue = int(input("Fatigue (1=No, 2=Yes): "))
allergy = int(input("Allergy (1=No, 2=Yes): "))
wheezing = int(input("Wheezing (1=No, 2=Yes): "))
alcohol_consuming = int(input("Alcohol Consuming (1=No, 2=Yes): "))
coughing = int(input("Coughing (1=No, 2=Yes): "))
shortness_of_breath = int(input("Shortness of Breath (1=No, 2=Yes): "))
swallowing_difficulty = int(input("Swallowing Difficulty (1=No, 2=Yes): "))
chest_pain = int(input("Chest Pain (1=No, 2=Yes): "))

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

print("\n===================================")
print("          PREDICTION RESULT")
print("===================================")

print(f"Prediction: {result}")
print(f"Probability of YES: {probability:.2f}%")

print("\n===================================")
print("          AI EXPLANATION")
print("===================================")

explanation = get_local_explanation(
result,
probability,
age,
gender,
patient_data
)

print(explanation)

print("\n===================================")
print("NOTE")
print("===================================")
print("This is an educational machine learning project.")
print("The prediction is not a medical diagnosis.")
print("For real medical concerns, consult a qualified healthcare professional.")
print("===================================")
