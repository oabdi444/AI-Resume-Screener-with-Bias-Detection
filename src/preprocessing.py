import os
import re

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            documents.append((filename, clean_text(text)))
    return documents

# -------------------------------
# File: src/model.py
# -------------------------------
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_similarity_scores(resumes, job_description):
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    scores = []
    for name, resume_text in resumes:
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
        scores.append((name, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)
