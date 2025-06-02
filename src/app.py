import os

from src.model import get_similarity_scores
from src.bias_detection import detect_age_bias

# Load resume and job description files
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

    # For this example, just compare each resume to the first job description
    if not jobs:
        print("No job descriptions found.")
        return

    job_name, job_text = jobs[0]

    print(f"Comparing all resumes to job description: {job_name}\n")

    for resume_name, resume_text in resumes:
        score = get_similarity_scores(resume_text, job_text)
        print(f"Resume: {resume_name} — Similarity Score: {score:.2f}")

    # Check for age bias
    biased = detect_age_bias(resumes)
    if biased:
        print("\n⚠️ Biased Resumes Detected (mentioning age):")
        for name in biased:
            print(f" - {name}")
    else:
        print("\n✅ No age-biased resumes detected.")

if __name__ == "__main__":
    main()
