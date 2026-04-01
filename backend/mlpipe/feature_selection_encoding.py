import pandas as pd
import os 
import numpy 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


input_data = "./datasets/inputFile.csv"
output_data = "processed_data.csv"

df = pd.read_csv(input_data)

not_important_cols = [
    'patient_id', 'name', 'doctor', 'hospital',
    'room_number', 'source_file',
    'admission_date', 'discharge_date', 'discharge_date_1',
    'date_of_admission'
]


# arr  = list(df.columns)
# for item in arr:
#    print(f"{item}: {df[item].dtype}")



df = df.drop(columns=not_important_cols, errors='ignore')

target = 'medical_condition'

binary_cols = [
    'diabetes', 'hypertension', 'obesity', 'smoking',
    'alcohol_consumption', 'family_history',
    'heart_attack_history', 'health_insurance'
]

numeric_cols = [
    'age', 'physical_activity', 'diet_score',
    'cholesterol_level', 'triglyceride_level',
    'ldl_level', 'hdl_level',
    'systolic_bp', 'diastolic_bp',
    'air_pollution_exposure', 'stress_level',
    'healthcare_access', 'emergency_response_time',
    'annual_income', 'year_of_admission',
    'satisfaction', 'average_duration_days',
    
    # Liver features: This is Liver Column
    'total_bilirubin', 'direct_bilirubin',
    'alkaline_phosphotase',
    'alamine_aminotransferase',
    'aspartate_aminotransferase',
    'total_protiens', 'albumin',
    'albumin_and_globulin_ratio'
]

# Cardinality of the dataset  
low_card_cat = [
    'gender', 'admission_type', 'severity',
    'common_in_region', 'language_availability',
    'blood_type'
]

high_card_cat = [
    'state_name', 'patient_state',
    'insurance_provider', 'medication',
    'medication_1', 'condition'
]

text_cols = ["symptom"]

df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
df['chol_ratio'] = df['ldl_level'] / (df['hdl_level'] + 1)
df['lifestyle_risk'] = (
    df['smoking'] + df['alcohol_consumption'] + df['obesity']
)

# This Lines perform the One-Hot encoding on the Low Cardinality data point
df = pd.get_dummies(df, columns=low_card_cat, drop_first=True)


# This Perform the Frequency Encoding(High Cardinality)
for col in high_card_cat:
    freq = df[col].value_counts()
    df[col] = df[col].map(freq)

df[binary_cols] = df[binary_cols].astype(int)


## Performing the TF-> IDF encoding 
def tf(df):
    tfidf = TfidfVectorizer(max_features=100)

    symptom_vec = tfidf.fit_transform(df['symptom'].astype(str))

    symptom_df = pd.DataFrame(
        symptom_vec.toarray(),
        columns=[f"symptom_{i}" for i in range(100)]
    )

    df = pd.concat([df.reset_index(drop=True), symptom_df], axis=1)
    df = df.drop(columns=['symptom'])


    return df

## This is used for the scaling for numerical cols

def scaling_handler(df):

    scalar = StandardScaler()
    df[numeric_cols] = scalar.fit_transform(df[numeric_cols])

    return df

df = tf(df)
df = scaling_handler(df)

df.to_csv(output_data, index=False)
