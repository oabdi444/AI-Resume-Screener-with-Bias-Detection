AI Resume Screener with Bias Detection and Explainability
Overview
This project applies Natural Language Processing (NLP) and Machine Learning (ML) techniques to streamline the resume screening process. In addition to ranking candidates based on relevance to job descriptions, the system includes a mechanism to detect potential age-related biases and supports model interpretability using SHAP.

This tool is intended to assist HR teams and recruiters in making more data-driven, transparent, and equitable hiring decisions.

Key Features
Resume Parsing and Preprocessing
Efficiently processes and cleans raw resume text for semantic analysis.

Job Relevance Matching
Utilizes embedding-based semantic similarity to assess how well each candidate matches the job description.

Candidate Ranking
Automatically ranks resumes based on their semantic similarity scores.

Age Bias Detection
Flags resumes that may reflect age-related bias using rule-based heuristics (e.g., frequent references to dates, excessive years of experience).

Model Explainability (Optional Extension)
Integrates SHAP (SHapley Additive exPlanations) for interpreting the model’s decision-making process.

Web-Based User Interface
Built with Streamlit, allowing for interactive resume uploads, job description input, and result visualization.

Project Structure
bash
Copy
Edit
resume-screener/
│
├── src/
│   ├── app.py                # Streamlit app main script
│   ├── bias_detector.py      # Age bias detection logic
│   ├── ranker.py             # Candidate scoring and ranking
│   └── utils.py              # Utility functions (cleaning, embeddings)
│
├── data/
│   └── sample_resumes/       # Example resumes for testing
│
├── models/
│   └── embeddings.pkl        # Pretrained or cached embeddings
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
Getting Started
Prerequisites
Ensure you have Python 3.7+ installed. It is recommended to use a virtual environment.

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/resume-screener.git
cd resume-screener
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the application:

bash
Copy
Edit
streamlit run src/app.py
Usage
Open the Streamlit interface in your browser.

Upload one or more resumes (text or PDF format).

Input a job description in the provided field.

View the ranked list of candidates and any bias flags.

(Optional) Enable explainability to view SHAP outputs, if implemented.

Potential Enhancements
Integrate GPT or Claude for natural language job fit feedback.

Extend bias detection to include gender or ethnicity-related patterns.

Implement a recruiter feedback system for human-in-the-loop updates.

Deploy via Docker or to a cloud platform (e.g., Streamlit Cloud, Render, AWS).

Add multi-metric dashboards for HR analytics.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions, suggestions, or collaboration inquiries, please contact:
Osman (GITHUB-OABDI444)

