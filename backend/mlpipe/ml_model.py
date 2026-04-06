import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# -------------------------
# Load matrix
# -------------------------
matrix = pd.read_csv("weighted_matrix.csv", index_col=0)
symptoms = matrix.columns.tolist()

# -------------------------
# Synthetic patient generator
# -------------------------
def generate_patient(disease):
    probs = matrix.loc[disease].values
    probs = probs / probs.sum()  # ensure valid distribution

    chosen = np.random.choice(symptoms, size=5, p=probs)

    row = {}
    for sym in symptoms:
        if sym in chosen:
            row[f"{sym}_present"] = 1
            row[f"{sym}_severity"] = np.random.randint(1, 5)
            row[f"{sym}_duration"] = np.random.randint(1, 30)
        else:
            row[f"{sym}_present"] = 0
            row[f"{sym}_severity"] = 0
            row[f"{sym}_duration"] = 0

    row["disease"] = disease
    return row


def generate_dataset(samples_per_disease=200):
    data = []
    for disease in matrix.index:
        for _ in range(samples_per_disease):
            data.append(generate_patient(disease))
    return pd.DataFrame(data)

# -------------------------
# Generate data
# -------------------------
df = generate_dataset()

# -------------------------
# Train model
# -------------------------
X = df.drop(columns=["disease"])
y = df["disease"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# Save model + encoder
joblib.dump(model, "ml_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model trained and saved.")