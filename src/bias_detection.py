import re

# Patterns that may indicate age bias
AGE_KEYWORDS = [
    r"\bage\b", r"\byears old\b", r"\b\d{2}\s*(years|yrs)\b", r"\bborn in\b", r"\bDOB\b", r"\bbirth\b"
]

def detect_age_bias(resume_list):
    biased_resumes = []
    for name, text in resume_list:
        for pattern in AGE_KEYWORDS:
            if re.search(pattern, text, re.IGNORECASE):
                biased_resumes.append(name)
                break
    return biased_resumes
