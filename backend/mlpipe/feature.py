# import pandas as pd
# import numpy as np

# matrix = pd.read_csv("weighted_matrix.csv", index_col=0)

# def diagnose(user_input, matrix):
#     scores = {}

#     for disease in matrix.index:
#         log_score = 0.0
        
#         for item in user_input:
#             sym = item['symptom'].lower().strip()
#             days = item.get('days', 1)
#             severity = item.get('severity', 1)
#             if sym in matrix.columns:
#                 p = matrix.loc[disease, sym]
#                 p = min(max(p, 1e-6), 1 - 1e-6)
#                 weight = severity * (1 + days / 10)
#                 log_score += weight * np.log(p)
#         scores[disease] = log_score

#     return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# user_input = [
#     {'symptom': 'persistent cough', 'days': 21, 'severity': 4},
#     {'symptom': 'fever', 'days': 14, 'severity': 3},
#     {'symptom': 'night sweats', 'days': 14, 'severity': 3},
#     {'symptom': 'weight loss', 'days': 20, 'severity': 4},
#     {'symptom': 'fatigue', 'days': 18, 'severity': 3},
#     {'symptom': 'chest pain', 'days': 10, 'severity': 2},
#     {'symptom': 'coughing blood', 'days': 7, 'severity': 4}
# ]

# result = diagnose(user_input, matrix)

# for d, s in result[:5]:
#     print(d, s)


import pandas as pd
import numpy as np
import joblib

# -------------------------
# Load data + models
# -------------------------
matrix = pd.read_csv("weighted_matrix.csv", index_col=0)

ml_model = joblib.load("ml_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -------------------------
# Expert System
# -------------------------
def diagnose(user_input, matrix):
    scores = {}

    for disease in matrix.index:
        log_score = 0.0
        
        for item in user_input:
            sym = item['symptom'].lower().strip()
            days = item.get('days', 1)
            severity = item.get('severity', 1)

            if sym in matrix.columns:
                p = matrix.loc[disease, sym]
                p = min(max(p, 1e-6), 1 - 1e-6)

                weight = severity * (1 + days / 10)
                log_score += weight * np.log(p)

        scores[disease] = log_score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# -------------------------
# ML Feature Builder
# -------------------------
def build_input_vector(user_input, matrix):
    vector = {}

    # initialize all features
    for sym in matrix.columns:
        vector[f"{sym}_present"] = 0
        vector[f"{sym}_severity"] = 0
        vector[f"{sym}_duration"] = 0

    # fill user input
    for item in user_input:
        sym = item['symptom'].lower().strip()

        if sym in matrix.columns:
            vector[f"{sym}_present"] = 1
            vector[f"{sym}_severity"] = item.get('severity', 1)
            vector[f"{sym}_duration"] = item.get('days', 1)

    return np.array(list(vector.values())).reshape(1, -1)

# -------------------------
# ML Prediction
# -------------------------
def ml_predict(user_input, matrix, top_k=5):
    vec = build_input_vector(user_input, matrix)
    probs = ml_model.predict_proba(vec)[0]

    top_idx = np.argsort(probs)[-top_k:][::-1]

    return [
        (label_encoder.inverse_transform([i])[0], probs[i])
        for i in top_idx
    ]

# -------------------------
# Hybrid Fusion (IMPORTANT)
# -------------------------
def hybrid_predict(user_input, matrix, alpha=0.5, top_k=5):
    expert_results = diagnose(user_input, matrix)
    ml_results = ml_predict(user_input, matrix, top_k=len(matrix.index))

    # normalize expert scores (log → softmax)
    diseases = [d for d, _ in expert_results]
    exp_scores = np.array([s for _, s in expert_results])
    exp_probs = np.exp(exp_scores - np.max(exp_scores))
    exp_probs = exp_probs / exp_probs.sum()

    expert_dict = dict(zip(diseases, exp_probs))
    ml_dict = dict(ml_results)

    final_scores = {}

    for disease in diseases:
        e_score = expert_dict.get(disease, 0)
        m_score = ml_dict.get(disease, 0)

        final_scores[disease] = alpha * e_score + (1 - alpha) * m_score

    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

# -------------------------
# Example Input
# -------------------------
user_input = [
    {'symptom': 'persistent cough', 'days': 21, 'severity': 4},
    {'symptom': 'fever', 'days': 14, 'severity': 3},
    {'symptom': 'night sweats', 'days': 14, 'severity': 3},
    {'symptom': 'weight loss', 'days': 20, 'severity': 4},
    {'symptom': 'fatigue', 'days': 18, 'severity': 3},
    {'symptom': 'chest pain', 'days': 10, 'severity': 2},
    {'symptom': 'coughing blood', 'days': 7, 'severity': 4}
]

# -------------------------
# Run All Models
# -------------------------
print("=== Expert System ===")
expert_result = diagnose(user_input, matrix)
for d, s in expert_result[:5]:
    print(d, round(s, 3))

print("\n=== ML Model ===")
ml_result = ml_predict(user_input, matrix)
for d, p in ml_result:
    print(d, round(p, 3))

print("\n=== Hybrid Model ===")
hybrid_result = hybrid_predict(user_input, matrix)
for d, s in hybrid_result:
    print(d, round(s, 3))