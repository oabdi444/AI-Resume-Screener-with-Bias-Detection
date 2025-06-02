import os

from src.model import get_similarity_scores
from src.bias_detection import detect_age_bias

def load_text_from_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                data.append((filename, text))
    return data

def main():
    resume_folder = "data/resumes"
    job_folder = "data/job_descriptions"

    resumes = load_text_from_folder(resume_folder)
    jobs = load_text_from_folder(job_folder)

    if not jobs:
        print("No job descriptions found.")
        return

    if not resumes:
        print("No resumes found.")
        return

    # Loop over every job description
    for job_name, job_text in jobs:
        print(f"\n🧠 Matching Resumes to Job Description: {job_name}\n")
        for resume_name, resume_text in resumes:
            score = get_similarity_scores(resume_text, job_text)
            print(f"Resume: {resume_name} — Similarity Score: {score:.2f}")

    # Check for age bias in all resumes
    biased = detect_age_bias(resumes)
    if biased:
        print("\n🚨 Age Bias Detected in Resumes:")
        for name in biased:
            print(f" - {name}")
    else:
        print("\n✅ No age-biased resumes detected.")

if __name__ == "__main__":
    main()
