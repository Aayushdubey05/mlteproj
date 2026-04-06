import pandas as pd
import numpy as np

matrix = pd.read_csv("weighted_matrix.csv", index_col=0)

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


user_input = [
    {'symptom': 'fever', 'days': 5, 'severity': 3},
    {'symptom': 'joint pain', 'days': 2, 'severity': 2}
]

result = diagnose(user_input, matrix)

for d, s in result[:5]:
    print(d, s)

# def softmax(scores):
#     values = np.array(list(scores.values()))
#     exp_vals = np.exp(values - np.max(values))  # stability
#     probs = exp_vals / exp_vals.sum()
#     return dict(zip(scores.keys(), probs))


# results = dict(diagnose(['fever', 'joint pain'], matrix))
# probs = softmax(results)

# for d, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
#     print(d, round(p*100, 2), "%")