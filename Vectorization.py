import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Download necessary NLTK resources
nltk.download('stopwords')
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample text data
text_data = [
    "This is a sample sentence.",
    "Text vectorization is awesome!",
    "Let's build a POC for text vectorization."
]

# Preprocess function
def preprocess(text):
    """
    Preprocesses the input text by lowering case, removing non-alphanumeric characters,
    and removing stopwords.
    """
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = ' '.join([word for word in text.split() if word not in stopwords_set])  # Remove stopwords
    return text

# Function to preprocess a list of texts
def preprocess_texts(texts):
    """
    Preprocesses a list of text data.
    """
    return [preprocess(text) for text in texts]

# Preprocess the text data
preprocessed_texts = preprocess_texts(text_data)
print("preprocessed_texts:", preprocessed_texts)

tokenizer = sentence_model.tokenizer
encoding = tokenizer(preprocessed_texts, padding=True, truncation=False, return_tensors='pt', max_length=None)

# Convert token IDs back to readable tokens
for idx, input_ids in enumerate(encoding['input_ids']):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(f"Tokens for Sentence {idx+1}:", tokens)


# Print the tokenized input IDs (numerical IDs assigned to each token)
print("Tokenized Input IDs:", encoding['input_ids'])


# ---- TF-IDF Vectorization ----
# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform preprocessed texts into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

# Convert to array and get feature names (words)
tfidf_vectors = tfidf_matrix.toarray()
feature_names = vectorizer.get_feature_names_out()

print("TF-IDF Vectors:")
print(tfidf_vectors)
print("Feature Names (Words):", feature_names)

# ---- Word2Vec Vectorization ----
# Tokenize the texts
tokenized_texts = [word_tokenize(text) for text in preprocessed_texts]

# Train Word2Vec model
word2vec_model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Get the vector for a specific word (e.g., "sample")
word_vector = word2vec_model.wv['sample']

print("\nWord2Vec Vector for 'sample':")
print(word_vector)

# ---- Sentence-BERT Vectorization (Using all-MiniLM-L6-v2) ----
# Load the local Sentence-BERT model

# Generate sentence embeddings
embeddings = sentence_model.encode(preprocessed_texts)

print("\nSentence Embeddings:")
print(embeddings)

# ---- Cosine Similarity Comparisons ----
# Cosine similarity between TF-IDF vectors (first two sentences)
cosine_sim_tfidf = cosine_similarity([tfidf_vectors[0]], [tfidf_vectors[1]])
print("\nCosine Similarity (TF-IDF) between first two sentences:", cosine_sim_tfidf)

# Cosine similarity between Word2Vec vectors (for example, 'sample' and 'awesome')
word_similarity = word2vec_model.wv.similarity('sample', 'awesome')  # You can choose other words
print("\nWord2Vec Similarity between 'sample' and 'awesome':", word_similarity)

# Cosine similarity between Sentence-BERT embeddings (first two sentences)
cosine_sim_bert = cosine_similarity([embeddings[0]], [embeddings[1]])
print("\nCosine Similarity (Sentence-BERT) between first two sentences:", cosine_sim_bert)
