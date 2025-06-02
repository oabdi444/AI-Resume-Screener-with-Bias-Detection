I Resume Screener with Bias Detection and Explainability

## Overview
This project uses NLP and machine learning to:
- Match resumes to job descriptions
- Rank candidates
- Detect age-related bias
- Provide explanations using SHAP (optional extension)

## Features
- Resume parsing and cleaning
- Embedding-based semantic matching
- Age bias flagging based on text patterns
- Streamlit frontend

## Run the App
```bash
pip install -r requirements.txt
streamlit run src/app.py