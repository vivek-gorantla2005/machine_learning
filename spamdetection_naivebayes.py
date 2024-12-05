import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    'text': [
        "win money now", "get rich quick, click here!", "limited time offer, buy now and save big",
        "congratulations, you've won a free gift card", "get unlimited free money, claim now",
        "click here to claim your prize!", "exclusive offer: earn $1000 a day", "don't miss out on this amazing deal!",
        "the weather looks great today", "we have a meeting scheduled at 3 PM", "let's catch up over the weekend",
        "can you send me the project details?", "i am feeling a bit tired today", "the movie was really good, highly recommend it",
        "we should plan a trip to the beach", "the homework for tomorrow is due at noon"
    ],
    'label': [
        'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam',  # Spam labels
        'not spam', 'not spam', 'not spam', 'not spam', 'not spam', 'not spam', 'not spam','spam'  # Not Spam labels
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)


count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)


nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)
y_pred = nb_classifier.predict(X_test_tfidf)

# Calculate and print the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classify new text data 
new_text = [
    "Congratulations, you have won a free vacation!",
    "Let's schedule a call for tomorrow",
    "Get unlimited cash now by clicking here"
]

new_text_counts = count_vectorizer.transform(new_text)
new_text_tfidf = tfidf_transformer.transform(new_text_counts)
new_predictions = nb_classifier.predict(new_text_tfidf)

# Display the classification results for the new text
for text, category in zip(new_text, new_predictions):
    print(f"Text: '{text}' => Predicted Category: '{category}'")
