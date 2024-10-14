from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_text

def detect_plagiarism(user_text, db_texts):
    processed_user_text = preprocess_text(user_text)
    processed_db_texts = [preprocess_text(text) for text in db_texts]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_user_text] + processed_db_texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    results = []
    for idx, score in enumerate(similarity_scores):
        results.append((db_texts[idx], score))
    
    return results, similarity_scores

