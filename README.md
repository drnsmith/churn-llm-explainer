# Churn Risk Explainer — GPT + SHAP + Streamlit

This AI-powered app predicts **customer churn risk** and explains it using **plain English**, powered by SHAP + OpenAI GPT (3.5/4).

Perfect for CX, retention, and data science teams who need more than black-box predictions — they need **narratives that drive action**.

---

## Productisation Ready

This app is designed to evolve beyond a demo. Here's how it's **ready for production**:

- **Modular codebase** — Easy to wrap in Flask, deploy via Docker, or integrate into a customer service dashboard.
- **Explainable AI** — Transparent feature-level reasoning via SHAP, ready for compliance-heavy industries (telco, banking, insurance).
- **Pluggable Output** — Export churn reports as PDFs, email them to account managers, or log them into CRM systems.
- **LLM Swap Option** — Use GPT-4, Claude, or local LLMs by toggling one variable.
- **Log + Audit Trail** — Every explanation is timestamped and logged for internal review or analyst re-use.
- **Hugging Face Deployable** — Hosted UX in minutes, no infra needed.
- **Upcoming**: evaluation dashboards, email workflows, CSV batch explainers, multi-user mode.

If you're a business or consultancy working with churn, **this tool can go from prototype to pilot in under a week**.

---

## Features

 - Churn prediction using XGBoost  
 - SHAP-based feature attribution  
 - Natural language explanation via GPT  
 - PDF report generation  
 - CSV logging of predictions  
 - Interactive Streamlit UI  
 - Hugging Face-compatible deployment  

---

## Quickstart

### Local Installation

```bash
git clone https://github.com/drnsmith/ChurnAnalysis.git
cd ChurnAnalysis
pip install -r requirements.txt
export OPENAI_API_KEY=your-openai-key
streamlit run app/main.py
````

---

## Hugging Face Deployment

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) → **Create New Space**
2. Set:

   * **Space SDK**: `Streamlit`
   * **Repo name**: `churn-llm-explainer`
3. Add your `OPENAI_API_KEY` under **Settings → Secrets**
4. Push code:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/churn-llm-explainer
git push hf main
```

---

## Project Structure

```
app/
├── main.py              ← Streamlit UI
├── explainer.py         ← SHAP + GPT logic
├── utils/
│   └── email.py         ← (Coming soon) Email handler
├── data/churn_encoded.csv
├── logs/explanations_log.csv
├── reports/*.pdf
app.py                   ← Launcher for Hugging Face
requirements.txt
```

---

## Coming Soon

* Batch CSV explainer + download
* Email delivery to account managers
* Evaluation dashboard (model accuracy, AUC, recall)
* GPT-4 toggle + fallback logic
* Flask + API-ready version

---

## License

MIT License

```text
MIT License

Copyright (c) 2025 Natasha Smith

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

---

## Credits

Built by [@drnsmith](https://github.com/drnsmith) as part of a production-grade GenAI portfolio.

```

> **Note:** This repository was cleaned and reset on *2025-09-09* to provide a clear, review-ready version of the project.


