import os
import logging
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from openai import OpenAI

# ─────────────────────────────────────
# Optional Debug Mode
# ─────────────────────────────────────
DEBUG = os.getenv("DEBUG_SHAP", "false").lower() == "true"
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)

# ─────────────────────────────────────
# Load + Train Model
# ─────────────────────────────────────
df = pd.read_csv('data/churn_encoded.csv')
X = df.drop('churned', axis=1)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)

# ─────────────────────────────────────
# OpenAI Client Init
# ─────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_name = os.getenv("GPT_MODEL", "gpt-3.5-turbo")

# ─────────────────────────────────────
# Main Function
# ─────────────────────────────────────
def explain_customer_risk(customer_row, top_n=5):
    """
    Return a GPT explanation + SHAP summary for one customer's churn risk.
    """
    try:
        churn_prob = model.predict_proba([customer_row])[0][1]

        # Get SHAP values
        shap_values = explainer(customer_row.to_frame().T)
        values = shap_values.values[0]
        features = X.columns[:len(values)]

        # Create SHAP DataFrame
        shap_df = pd.DataFrame({
            'feature': features,
            'shap_value': values
        }).sort_values(by='shap_value', key=abs, ascending=False).head(top_n)

        if DEBUG:
            logging.debug("Top SHAP features:\n%s", shap_df)

        feature_summary = "\n".join(
            f"- {row['feature']}: SHAP value {row['shap_value']:.3f}"
            for _, row in shap_df.iterrows()
        )

        # GPT Prompt
        prompt = f"""
You are an AI assistant helping a data science team explain churn risk.
The predicted churn probability for this customer is {churn_prob:.2f}.

The top contributing features and their SHAP values are:
{feature_summary}

Generate a short, plain-English explanation.
Avoid jargon. Be clear, helpful, and precise.
"""

        # Call GPT
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content.strip()

        # Return explanation + SHAP values as list of tuples
        return response.choices[0].message.content, shap_df[['feature', 'shap_value']].values.tolist()


    except Exception as e:
        logging.error("Failed to generate explanation: %s", str(e))
        return f"(Error generating explanation: {str(e)})", []
        # Return empty explanation and SHAP values on error
