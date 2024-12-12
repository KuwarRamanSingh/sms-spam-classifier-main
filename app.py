import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize PorterStemmer
ps = PorterStemmer()


# Preprocessing function to clean the input text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    y = []
    for i in text:
        if i.isalnum():  # Only keep alphanumeric characters
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Perform stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Add custom background and professional styling
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #1E1E1E; /* Dark gray background */
    color: #F5F5F5; /* Light text color for readability */
    font-family: 'Arial', sans-serif;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0); /* Transparent header */
    color: #F5F5F5;
}
[data-testid="stSidebar"] {
    background-color: #252525; /* Slightly darker sidebar */
    color: white;
}
h1 {
    font-family: 'Arial Black', sans-serif;
    color: #00C8FF; /* Cool blue for headings */
    text-align: center;
    margin-bottom: 20px;
}
.stTextInput>div>div>input {
    background-color: #333333; /* Darker text input background */
    color: #F5F5F5; /* White text for inputs */
    border-radius: 8px;
    border: 1px solid #555555;
    font-size: 16px;
}
.stButton>button {
    background-color: #00C8FF; /* Blue button */
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #007ACC; /* Darker blue on hover */
}
.stMarkdown h3 {
    font-size: 22px;
    color: #F5F5F5;
    margin-top: 20px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Streamlit app UI
st.title("Email/SMS Spam Classifier")

# Input field for the SMS message
input_sms = st.text_area("Enter the message here:")

# Predict button to trigger classification
if st.button('Predict'):
    # Step 1: Preprocess the input SMS
    transformed_sms = transform_text(input_sms)

    # Step 2: Vectorize the preprocessed SMS using the loaded TF-IDF vectorizer
    vector_input = tfidf.transform([transformed_sms])

    # Step 3: Predict using the loaded model
    result = model.predict(vector_input)[0]

    # Step 4: Display the result (Spam or Not Spam)
    if result == 1:
        st.markdown("<h3 style='color: #FF4C4C;'>This message is classified as Spam.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: #4CAF50;'>This message is classified as Not Spam.</h3>", unsafe_allow_html=True)
