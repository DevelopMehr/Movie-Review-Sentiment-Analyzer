# 🎬 Movie Review Sentiment Analyzer

A production-ready Natural Language Processing (NLP) web application that analyzes user-submitted movie reviews and instantly predicts whether they are **Positive** or **Negative** with a confidence score.

Built using Python, Scikit-learn, and Streamlit, and trained on the IMDB 50k Movie Reviews dataset.

---

## 🚀 Features
* **Real-time Prediction:** Type or paste any review and get instant sentiment classification.
* **Confidence Scoring:** Displays the exact model confidence percentage for every prediction.
* **Color-Coded UI:** Clean, modern interface with distinct visual indicators (🟢 Positive / 🔴 Negative).
* **Robust Text Preprocessing:** Automatically handles HTML tags, lowercasing, punctuation removal, and stopword filtering.

---

## 🛠️ Tech Stack
* **Python:** Core programming language
* **Pandas:** Data loading and manipulation
* **Scikit-learn:** TF-IDF feature extraction and machine learning models (`LogisticRegression`)
* **NLTK:** Natural language toolkit for stopword removal
* **Pickle:** Model serialization and persistence
* **Streamlit:** Interactive web application framework

---

## 📊 Model Comparison & Selection
Before building the web app, three different machine learning models were trained and evaluated on an 80/20 train-test split:

| Model | Accuracy | F1 Score | Status |
| :--- | :--- | :--- | :--- |
| **Naive Bayes** | 85.06% | 0.8521 | Baseline |
| **Logistic Regression** | **88.76%** | **0.8899** | 🏆 **Selected (Best Performance)** |
| **Random Forest** | 84.88% | 0.8481 | Slower / Prone to Overfitting |

*Logistic Regression was selected for final deployment due to its superior accuracy, stability, and speed.*

---

## 📁 Project Structure

```text
sentiment-analysis/
├── IMDB Dataset.csv          # Original 50k IMDB reviews dataset
├── sentiment_analysis.py     # Data pipeline, model training & evaluation code
├── app.py                    # Streamlit web application
├── sentiment_model.pkl       # Saved Logistic Regression model
├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer vocabulary
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
