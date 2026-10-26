import pickle
import re
import nltk
from nltk.corpus import stopwords
import streamlit as st

# Download stopwords if not present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Load the saved model and vectorizer
@st.cache_resource
def load_models():
  model = pickle.load(open('sentiment_model.pkl', 'rb'))
  vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
  return model, vectorizer


model, vectorizer = load_models()


# Text cleaning function (must match training cleaning)
def clean_text(text):
  text = re.sub(r'<.*?>', '', text)
  text = text.lower()
  text = re.sub(r'[^a-zA-Z\s]', '', text)
  words = text.split()
  words = [w for w in words if w not in stop_words]
  return ' '.join(words)


# --- Streamlit UI Design ---
st.set_page_config(
    page_title='Movie Review Sentiment Analyzer', page_icon='🎬', layout='centered'
)

st.title('🎬 Movie Review Sentiment Analyzer')
st.write(
    'Type any movie review below and click **Predict** to see if it is'
    ' **Positive** or **Negative**!'
)

# Text box for user input
user_review = st.text_area(
    'Enter your movie review here:',
    placeholder='Type or paste your review here...',
    height=150,
)

if st.button('Predict Sentiment', type='primary'):
  if user_review.strip() == '':
    st.warning('⚠️ Please enter a review before clicking predict.')
  else:
    # 1. Clean review
    cleaned = clean_text(user_review)

    # 2. Vectorize
    vectorized = vectorizer.transform([cleaned])

    # 3. Predict sentiment & probabilities
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    # Get confidence score for the predicted class
    if prediction == 'positive':
      confidence = probabilities[1] * 100
    else:
      confidence = probabilities[0] * 100

    # 4. Display Results with styling
    st.markdown('---')
    st.subheader('Result:')

    if prediction == 'positive':
      st.success(f'🟢 **Positive Review** (Confidence: {confidence:.2f}%)')
    else:
      st.error(f'🔴 **Negative Review** (Confidence: {confidence:.2f}%)')