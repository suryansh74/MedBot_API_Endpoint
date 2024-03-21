from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS

from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
# Load dataset
data = pd.read_json("1.format.json")

# Preprocess text data
for col in data.columns:
    if col != 'name':
        data[col] = data[col].astype(str).apply(lambda x: x.lower())

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit vectorizer on preprocessed text data
tfidf_matrix = vectorizer.fit_transform(data['description'] + " " + data['symptoms'] + " " + data['treatment'] + " " + data['medicine'] + " " + data['prescription'] + " " + data['prevalence_incidence'] + " " + data['causes_risk_factors'] + " " + data['diagnostic_tests'] + " " + data['age_gender_distribution'] + " " + data['geographical_distribution'] + " " + data['comorbidities'] + " " + data['genetic_factors'] + " " + data['environmental_factors'] + " " + data['patient_reported_outcomes'] + " " + data['research_studies_clinical_trials'])

def get_responses(user_query, top_n=5):
    # Preprocess user query
    processed_query = user_query.lower()
    
    # Transform user query into TF-IDF vector
    query_vector = vectorizer.transform([processed_query])
    
    # Calculate cosine similarity between user query vector and dataset vectors
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    
    # Get indices of top N rows with highest similarity scores
    top_indices = similarity_scores.argsort(axis=1)[0][-top_n:][::-1]
    
    responses = []
    for index in top_indices:
        # Get information from the matched row
        matched_row = data.iloc[index]
        
        # Construct response
        response = {"disease": matched_row['name'], "similarity_score": similarity_scores[0][index]}
        
        responses.append(response)
    
    return responses

@app.route('/get_responses', methods=['POST'])
def process_query():
    user_query = request.json['user_query']
    if not user_query:
        return jsonify({"error": "Please provide a user query."}), 400
    
    responses = get_responses(user_query, top_n=5)
    return jsonify({"responses": responses})

if __name__ == '__main__':
    app.run(debug=True)
