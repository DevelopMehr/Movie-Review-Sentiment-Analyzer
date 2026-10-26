
import pickle
import re
import nltk
from nltk.corpus import stopwords
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB


# Download stopwords list if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ==========================================
# STEP 1 — LOAD AND EXPLORE
# ==========================================
df = pd.read_csv('IMDB Dataset.csv')

print('Shape:', df.shape)
print('\nColumns:', df.columns.tolist())
print('\nFirst 3 rows:')
print(df.head(3))
print('\nSentiment counts:')
print(df['sentiment'].value_counts())
print('\nSample review:')
print(df['review'].iloc[0][:300])


# ==========================================
# STEP 2 — CLEAN THE TEXT
# ==========================================
def clean_text(text):
  # 1. Remove HTML tags
  text = re.sub(r'<.*?>', '', text)

  # 2. Lowercase everything
  text = text.lower()

  # 3. Remove punctuation and numbers (keep only letters and spaces)
  text = re.sub(r'[^a-zA-Z\s]', '', text)

  # 4. Remove stopwords
  words = text.split()
  words = [w for w in words if w not in stop_words]

  return ' '.join(words)


# Test cleaning on our sample review
sample_cleaned = clean_text(df['review'].iloc[0])
print('\n--- Before Cleaning ---')
print(df['review'].iloc[0][:300])
print('\n--- After Cleaning ---')
print(sample_cleaned[:300])

# Apply cleaning to the dataset
print('\nCleaning dataset reviews (this may take a minute)...')
df['cleaned_review'] = df['review'].apply(clean_text)
print('Cleaning complete!')
print('Sample cleaned review:', df['cleaned_review'].iloc[0][:200])

# ==========================================
# STEP 3 — SPLIT DATA & CONVERT TO NUMBERS (TF-IDF)
# ==========================================
print('\nSplitting data into Train and Test sets...')
X = df['cleaned_review']
y = df['sentiment']

# 80% for training the model, 20% for testing it
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Training samples: {len(X_train)}')
print(f'Testing samples: {len(X_test)}')

print('\nConverting text to numbers using TF-IDF...')
vectorizer = TfidfVectorizer(max_features=5000)  # Keeping top 5,000 words

# Fit and transform on training data, transform only on test data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print('TF-IDF Matrix Shape (Train):', X_train_vec.shape)
print('TF-IDF Matrix Shape (Test):', X_test_vec.shape)
print('Step 3 Complete!')

# ==========================================
# STEP 4 — TRAIN 3 MODELS AND COMPARE
# ==========================================
print('\nTraining Model 1: Naive Bayes...')
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

nb_acc = accuracy_score(y_test, nb_preds)
nb_f1 = f1_score(y_test, nb_preds, pos_label='positive')
print(f'Naive Bayes -> Accuracy: {nb_acc:.4f} | F1 Score: {nb_f1:.4f}')

print('\nTraining Model 2: Logistic Regression...')
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_preds = lr_model.predict(X_test_vec)

lr_acc = accuracy_score(y_test, lr_preds)
lr_f1 = f1_score(y_test, lr_preds, pos_label='positive')
print(f'Logistic Regression -> Accuracy: {lr_acc:.4f} | F1 Score: {lr_f1:.4f}')

print('\nTraining Model 3: Random Forest...')
# Using n_estimators=100 for a balance between speed and performance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_vec, y_train)
rf_preds = rf_model.predict(X_test_vec)

rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds, pos_label='positive')
print(f'Random Forest -> Accuracy: {rf_acc:.4f} | F1 Score: {rf_f1:.4f}')

print('\nStep 4 Complete!')

# ==========================================
# STEP 5 — SAVE THE BEST MODEL & VECTORIZER
# ==========================================
print('\nSaving the best model (Logistic Regression) and vectorizer...')

# Save Logistic Regression model
with open('sentiment_model.pkl', 'wb') as model_file:
  pickle.dump(lr_model, model_file)

# Save TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as vec_file:
  pickle.dump(vectorizer, vec_file)

print('Files saved successfully: sentiment_model.pkl & tfidf_vectorizer.pkl')
print('\n🎉 All training steps (1-5) complete!')


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