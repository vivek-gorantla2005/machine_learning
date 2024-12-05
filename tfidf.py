from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer instance
vec = TfidfVectorizer()

# Random one-line strings
data = [
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is transforming the world",
    "Python is a versatile programming language",
    "Machine learning is a subset of AI"
]

# Fit and transform the data
tfidf = vec.fit_transform(data)

# Display the resulting cosine similarity matrix
print((tfidf * tfidf.T).toarray())

