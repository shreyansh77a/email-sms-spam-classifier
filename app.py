import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure nltk data is available (downloads only if missing)
for pkg in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)

ps = PorterStemmer()

# Function for preprocessing text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load saved models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit Page Config
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
            padding: 2rem;
            border-radius: 12px;
        }
        h1 {
            color: #2C3E50;
            text-align: center;
        }
        .stTextInput input {
            border: 2px solid #3498db !important;
            border-radius: 10px !important;
            padding: 0.6rem !important;
        }
        .result-box {
            text-align: center;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1.3rem;
            margin-top: 1rem;
        }
        .spam {
            background-color: #ffe6e6;
            color: #c0392b;
            font-weight: bold;
        }
        .not-spam {
            background-color: #eafaf1;
            color: #27ae60;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("📧 Email / SMS Spam Classifier")
st.write("### Instantly check if a message is **Spam or Not Spam** using a trained ML model 🔍")

# Input Area
sms_input = st.text_area("Enter your message below:", height=150, placeholder="Type or paste your message here...")

# Predict Button
if st.button("🔍 Classify Message"):
    if sms_input.strip() == "":
        st.warning("⚠️ Please enter a message before prediction.")
    else:
        # Preprocess and Predict
        transformed_sms = transform_text(sms_input)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Display Result
        if result == 1:
            st.markdown('<div class="result-box spam">🚨 This message is **SPAM**!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box not-spam">✅ This message is **NOT SPAM**!</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    ---
    💡 *Built with ❤️ using Streamlit & Machine Learning.*
""")

