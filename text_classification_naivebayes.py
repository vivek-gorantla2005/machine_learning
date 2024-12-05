import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Try specifying a different encoding
try:
    athesium = pd.read_table(
        r'C:\Users\vivek\Downloads\archive (9)\alt.atheism.txt',
        header=None,
        encoding='ISO-8859-1',
        on_bad_lines='skip'
    )

    religion = pd.read_table(
        r'C:\Users\vivek\Downloads\archive (9)\talk.religion.misc.txt',
        header=None,
        encoding='ISO-8859-1',
        on_bad_lines='skip'
    )

    graphics = pd.read_table(
        r'C:\Users\vivek\Downloads\archive (9)\comp.graphics.txt',
        header=None,
        encoding='ISO-8859-1',
        on_bad_lines='skip'
    )

except Exception as e:
    print(f"Error: {e}")

# Combine the text data into a single DataFrame
all_text = pd.concat([athesium, religion, graphics], axis=0)
all_text.columns = ['text']  # Assign column name
all_text['label'] = ['atheism'] * len(athesium) + ['religion'] * len(religion) + ['graphics'] * len(graphics)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    all_text['text'], all_text['label'], test_size=0.2, random_state=42
)

# Initialize CountVectorizer and TfidfTransformer
count_vector = CountVectorizer()
tfidf_transformer = TfidfTransformer()

# Fit-transform the training data
X_train_counts = count_vector.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train the Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Transform the test data
X_test_counts = count_vector.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Predict on the test set
y_pred = clf.predict(X_test_tfidf)

# Evaluate model accuracy
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classify new text
new_text = ["vivek is bad", "Computer graphics are fascinating"]
new_text_counts = count_vector.transform(new_text)
new_text_tfidf = tfidf_transformer.transform(new_text_counts)
new_predictions = clf.predict(new_text_tfidf)

# Display classification of new text
for text, category in zip(new_text, new_predictions):
    print(f"Text: '{text}' => Predicted Category: '{category}'")
