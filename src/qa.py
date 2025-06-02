# src/qa.py

import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_resume_feedback(resume_text, job_text):
    prompt = f"""
You're a professional recruiter. Analyze the following resume and job description. Give feedback:
- How well does this resume match the job?
- Strengths?
- Weaknesses?
- What could improve?

Resume:
{resume_text}

Job Description:
{job_text}
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def ask_resume_question(resume_text, job_text, question):
    prompt = f"""
Given this resume and job description:

Resume:
{resume_text}

Job Description:
{job_text}

Answer this question: {question}
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()
