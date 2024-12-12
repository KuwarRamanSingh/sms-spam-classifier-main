from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import pandas as pd

# Load your dataset (make sure to replace 'spam.csv' with the actual path)
df = pd.read_csv('spam.csv')

# Extract features (X) and labels (y)
X = df['v2']  # assuming 'v2' column has the SMS text
y = df['v1'].map({'ham': 0, 'spam': 1})  # convert 'ham' to 0, 'spam' to 1

# Initialize TfidfVectorizer and model
tfidf = TfidfVectorizer(stop_words='english')  # you can add other preprocessing steps here
model = MultinomialNB()

# Fit the vectorizer on the training data
X_tfidf = tfidf.fit_transform(X)

# Fit the model on the vectorized data
model.fit(X_tfidf, y)

# Save the vectorizer and model to pickle files
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer saved successfully!")
