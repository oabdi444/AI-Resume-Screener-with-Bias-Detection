from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_similarity_scores(resume_text, job_description):
    """
    Calculates the similarity score between a resume and a job description
    using TF-IDF vectorization and cosine similarity.
    """
    documents = [resume_text, job_description]
    
    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Compute cosine similarity between the resume and job description
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = similarity_matrix[0][0]  # Extract the single similarity value
    
    return score
