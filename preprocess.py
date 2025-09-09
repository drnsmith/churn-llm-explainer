import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the cleaned CSV
df = pd.read_csv('data/churn_cleaned.csv')

# Make a copy to preserve original
df_encoded = df.copy()

# Encode categorical variables
categorical_cols = df_encoded.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le  # Save encoders in case you want to decode later

# Save to CSV
df_encoded.to_csv('data/churn_encoded.csv', index=False)
print("Encoded file saved as 'data/churn_encoded.csv'")
