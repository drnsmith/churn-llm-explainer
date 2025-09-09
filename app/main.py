# app/main.py

import streamlit as st
import pandas as pd
from datetime import datetime
import os
from fpdf import FPDF
import matplotlib.pyplot as plt
from app.explainer import explain_customer_risk, X, model
from app.utils.email import send_email_report  # <-- make sure this file exists!
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────
# Setup
# ─────────────────────────────────────
st.set_page_config(page_title="Churn Explainer", layout="centered")
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ─────────────────────────────────────
# PDF Export Helper
# ─────────────────────────────────────
def save_to_pdf(customer_id, explanation, top_features, output_dir="reports"):
    filename = f"{output_dir}/churn_report_{customer_id}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, f"Churn Risk Report for Customer {customer_id}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Explanation:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, explanation)
    pdf.ln(5)

    if top_features:
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Top SHAP Features:", ln=True)
        pdf.set_font("Arial", size=12)
        for feature, value in top_features:
            pdf.cell(0, 10, f"- {feature}: {value:.3f}", ln=True)

    pdf.output(filename)
    return filename

# ─────────────────────────────────────
# UI: Title + Customer Selector
# ─────────────────────────────────────
st.title("Customer Churn Risk Explainer")
customer_idx = st.selectbox("Select a customer (by index):", range(len(X)))
customer_row = X.iloc[customer_idx]
customer_df = pd.DataFrame(customer_row).transpose()

# ─────────────────────────────────────
# Display Selected Customer
# ─────────────────────────────────────
st.subheader("Customer Details")
st.dataframe(customer_df)

# ─────────────────────────────────────
# Predict Churn
# ─────────────────────────────────────
churn_prob = model.predict_proba([customer_row])[0][1]
st.metric(label="Predicted Churn Risk", value=f"{churn_prob:.2%}")

# ─────────────────────────────────────
# Explain + Log + Export
# ─────────────────────────────────────
if st.button("Explain This"):
    with st.spinner("Talking to GPT-3.5..."):
        explanation_text, top_features = explain_customer_risk(customer_row)
        st.subheader("Why this customer might churn:")
        st.write(explanation_text)

        # Log explanation
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "customer_index": int(customer_idx),
            "risk_score": float(churn_prob),
            "explanation": explanation_text
        }
        log_df = pd.DataFrame([log_entry])
        log_df.to_csv("logs/explanations_log.csv", mode='a', index=False, header=False)
        st.success("Logged explanation.")

        # Get SHAP features
        top_features = explain_customer_risk.shap_values_ if hasattr(explain_customer_risk, "shap_values_") else []
        top_features = top_features[['feature', 'shap_value']].values.tolist() if len(top_features) > 0 else []

        # Show SHAP bar chart
        if top_features:
            st.subheader("Top Features Contributing to Churn Risk")
            shap_df = pd.DataFrame(top_features, columns=["Feature", "SHAP Value"])
            st.bar_chart(shap_df.set_index("Feature"))

        # Save PDF
        pdf_path = save_to_pdf(customer_idx, explanation_text, top_features)
        st.write(explanation_text)
        st.success("PDF report generated.")

        # Download button
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f,
                file_name=f"churn_report_{customer_idx}.pdf",
                mime="application/pdf"
            )

        # Email (optional)
        if st.checkbox("Send via Email"):
            recipient = st.text_input("Enter recipient email:")
            if st.button("Send Report"):
                try:
                    send_email_report(
                        recipient_email=recipient,
                        subject="Customer Churn Risk Report",
                        body="Attached is the churn explanation PDF.",
                        pdf_path=pdf_path
                    )
                    st.success("Email sent successfully.")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")

        st.balloons()
# ─────────────────────────────────────
# Footer
# ─────────────────────────────────────
st.markdown("---")
st.markdown("Made by Natasha Smith")

