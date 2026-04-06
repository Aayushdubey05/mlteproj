import joblib
import numpy as np

model = joblib.load("ml_model.pkl")
le = joblib.load("label_encoder.pkl")

# same symptoms list used in training
symptoms = model.feature_names_in_

def build_input_vector(user_input):
    vector = {feature: 0 for feature in symptoms}

    for item in user_input:
        sym = item['symptom'].lower().strip()

        present_key = f"{sym}_present"
        severity_key = f"{sym}_severity"
        duration_key = f"{sym}_duration"

        if present_key in vector:
            vector[present_key] = 1
            vector[severity_key] = item.get('severity', 1)
            vector[duration_key] = item.get('days', 1)

    return np.array(list(vector.values())).reshape(1, -1)


def ml_predict(user_input, top_k=5):
    vec = build_input_vector(user_input)
    probs = model.predict_proba(vec)[0]

    top_idx = np.argsort(probs)[-top_k:][::-1]
    return [(le.inverse_transform([i])[0], probs[i]) for i in top_idx]