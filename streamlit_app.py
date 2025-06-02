import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re
from src.qa import get_resume_feedback, ask_resume_question

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data from folder
def load_text_from_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                text = file.read()
                data.append((filename, text))
    return data

# Improved similarity scoring using embeddings
def get_similarity_scores(resume_text, job_text):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_emb = model.encode(job_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(resume_emb, job_emb)
    return float(score)

# Improved age bias detection using broader regex
AGE_KEYWORDS = [
    r"\bage\s*[:\-]?\s*\d{2}\b",
    r"\b\d{2}\s*years old\b",
    r"\byears of age\b",
    r"\bborn in \d{4}\b",
    r"\bdob\s*[:\-]?\s*\d{2}[/-]\d{2}[/-]\d{2,4}\b",
    r"\bdate of birth\b",
    r"\bborn\s*[:\-]?\s*\w+\s+\d{1,2},?\s+\d{4}\b",
    r"\b45\b", r"\b50\b", r"\b60\b"  # common direct age mentions
]

def detect_age_bias(resumes):
    biased = []
    for name, text in resumes:
        for pattern in AGE_KEYWORDS:
            if re.search(pattern, text, re.IGNORECASE):
                biased.append(name)
                break
    return biased

# Streamlit App
st.title("\U0001F4C4 AI Resume Screener with Bias Detection")

# Load data
resumes = load_text_from_folder("data/resumes")
jobs = load_text_from_folder("data/job_descriptions")

if not resumes or not jobs:
    st.warning("Please add some `.txt` resumes and job descriptions in the `data/` folders.")
else:
    for job_name, job_text in jobs:
        st.subheader(f"\U0001F9E0 Matching Resumes to Job: `{job_name}`")

        for resume_name, resume_text in resumes:
            score = get_similarity_scores(resume_text, job_text)
            st.write(f"**{resume_name}** — Similarity Score: `{score:.2f}`")

            # === Added AI Feedback and Q&A Section ===
            with st.expander(f"🔍 Resume Review for {resume_name} (Job: {job_name})"):
                feedback = get_resume_feedback(resume_text, job_text)
                st.markdown(f"**Feedback:**\n{feedback}")

                st.markdown("**Ask a Question About This Resume:**")
                user_question = st.text_input(f"Ask about {resume_name} (Job: {job_name}):", key=f"q_{job_name}_{resume_name}")

                if st.button(f"Ask about {resume_name} (Job: {job_name})", key=f"btn_{job_name}_{resume_name}"):
                    if user_question:
                        answer = ask_resume_question(resume_text, job_text, user_question)
                        st.markdown(f"**Answer:** {answer}")
                    else:
                        st.warning("Please type a question.")

    st.subheader("\U0001F6A8 Age Bias Detection")
    biased = detect_age_bias(resumes)
    if biased:
        st.error("\u26A0\uFE0F Biased resumes mentioning age:")
        for name in biased:
            st.write(f"- {name}")
    else:
        st.success("\u2705 No age-biased resumes detected.")
