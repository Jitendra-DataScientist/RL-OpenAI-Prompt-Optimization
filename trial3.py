import openai
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy for linguistic evaluation
nlp = spacy.load("en_core_web_sm")

# Set your OpenAI API key
openai.api_key = "your-openai-api-key"

# Define an expected topic or keywords
EXPECTED_KEYWORDS = ["artificial intelligence", "AI", "machine learning", "technology"]

def get_response(prompt):
    """
    Interacts with the OpenAI API using the given prompt.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def semantic_relevance(response, keywords):
    """
    Measures semantic similarity between the response and expected keywords.
    """
    if not response:
        return 0
    tfidf = TfidfVectorizer()
    corpus = [" ".join(keywords), response]
    vectors = tfidf.fit_transform(corpus)
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return similarity  # Returns a value between 0 and 1

def linguistic_quality(response):
    """
    Evaluates linguistic clarity using spaCy.
    Penalizes responses with poor grammar or low token count.
    """
    if not response:
        return 0
    doc = nlp(response)
    grammar_errors = sum(1 for token in doc if token.is_alpha and token.pos_ == "X")
    length_penalty = max(0, len(response.split()) - 5)  # Penalize very short responses
    return max(0, 1 - grammar_errors / (len(doc) + 1) + length_penalty / 50)

def reward_function(response):
    """
    Combines semantic relevance and linguistic quality for a comprehensive reward.
    """
    if not response:
        return -1  # Penalize empty responses
    relevance_score = semantic_relevance(response, EXPECTED_KEYWORDS)
    quality_score = linguistic_quality(response)
    combined_score = 0.7 * relevance_score + 0.3 * quality_score
    return combined_score

# Example usage
initial_prompt = "Explain artificial intelligence in simple terms."
response = get_response(initial_prompt)

if response:
    print("Response:", response)
    reward = reward_function(response)
    print("Reward:", reward)
else:
    print("No response received.")
