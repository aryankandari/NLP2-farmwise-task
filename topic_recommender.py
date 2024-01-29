import json
import requests
import langchain
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score, jaccard_score
from langchain.retrievers import WikipediaRetriever
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Step 2: Load the training data from the JSON file
with open('Reddit_data_train.json') as f:
    train_data = json.load(f)

# Step 3: Preprocess the training data
user_profiles = []
posts = []

for user, user_data in train_data.items():
    for post in user_data:
        posts.append(post['text'])
        user_profiles.append(' '.join([p['text'] for p in user_data]))

# Step 4: Generate topics using a generative AI model
def generate_topics(text):
    response = requests.post(
        'https://api.openai.com/v1/engines/davinci-002/completions',
        headers={'Authorization': f'Bearer {api_key}'},
        json={'prompt': text, 'max_tokens': 20, 'temperature': 0.7}
    )
    response_json = response.json()
    print(response_json)  # Add this line to inspect the response
    topics = response_json['choices'][0]['text'].strip().split('\n')
    return topics

generated_topics = [generate_topics(post) for post in posts]

# Step 5: Verify topics from Wikipedia
# Verify topics from Wikipedia
wikipedia_retriever = WikipediaRetriever()
def verify_topics(topics):
    verified_topics = []
    for topic in topics:
        if wikipedia_retriever.get_page(topic):
            verified_topics.append(topic)
    return verified_topics

verified_generated_topics = [verify_topics(topic_list) for topic_list in generated_topics]

# Step 6: Build the recommendation engine
def recommend_posts(post_index, user_profiles, post_topics_list, posts):
    post_topics = ' '.join(post_topics_list[post_index])
    tfidf_vectorizer = TfidfVectorizer()
    user_profiles_tfidf = tfidf_vectorizer.fit_transform(user_profiles.values())
    post_topics_tfidf = tfidf_vectorizer.transform([post_topics])

    similarities = cosine_similarity(post_topics_tfidf, user_profiles_tfidf)[0]
    top_users_indices = np.argsort(similarities)[-10:][::-1]
    recommended_posts = [posts[i] for i in top_users_indices]
    return recommended_posts

# Step 7: Evaluate the recommendation engine
# Assuming test_data, test_labels, and labels are properly defined
with open('Reddit_data_test.json') as f:
    test_data = json.load(f)

test_posts = [post_data['text'] for post_data in test_data]
test_generated_topics = [generate_topics(post) for post in test_posts]
test_verified_topics = [verify_topics(topic_list) for topic_list in test_generated_topics]

def evaluate_recommendations(user_profiles, test_verified_topics, test_labels, labels):
    total_ndcg = 0
    total_jaccard = 0

    tfidf_vectorizer = TfidfVectorizer()
    user_profiles_tfidf = tfidf_vectorizer.fit_transform(user_profiles.values())

    for i, post_topics in enumerate(test_verified_topics):
        post_topics_tfidf = tfidf_vectorizer.transform([' '.join(post_topics)])
        similarities = cosine_similarity(post_topics_tfidf, user_profiles_tfidf)[0]
        top_users_indices = np.argsort(similarities)[-10:][::-1]
        recommended_labels = [labels[index] for index in top_users_indices]

        true_labels = [test_labels[i]]
        ndcg = ndcg_score([true_labels], [recommended_labels], k=10)
        jaccard = jaccard_score([true_labels], [recommended_labels], average='samples')

        total_ndcg += ndcg
        total_jaccard += jaccard

    avg_ndcg = total_ndcg / len(test_data)
    avg_jaccard = total_jaccard / len(test_data)

    return avg_ndcg, avg_jaccard

# Run the evaluation
# avg_ndcg, avg_jaccard = evaluate_recommendations(user_profiles, test_verified_topics, test_labels, labels)
# print(f"Average NDCG: {avg_ndcg}")
# print(f"Average Jaccard Similarity: {avg_jaccard}")