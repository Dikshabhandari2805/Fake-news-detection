# Fake News Detection Project
# Simple implementation for academic purposes

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download stopwords
try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except:
    stop_words = {'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}

def preprocess_text(text):
    """Clean text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def main():
    print("FAKE NEWS DETECTION PROJECT")
    print("=" * 40)
    
    # Step 1: Load datasets
    print("1. Loading datasets...")
    fake_df = pd.read_csv('Fake.csv')
    fake_df['label'] = 0  # Fake = 0
    
    true_df = pd.read_csv('True.csv')
    true_df['label'] = 1  # Real = 1
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    print(f"   Total articles: {len(df)}")
    print(f"   Fake articles: {len(fake_df)}")
    print(f"   Real articles: {len(true_df)}")
    
    # Step 2: Text preprocessing
    print("\n2. Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['processed_text'].str.len() > 0]
    print(f"   Articles after cleaning: {len(df)}")
    
    # Step 3: Feature extraction
    print("\n3. Converting text to features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label']
    print(f"   Feature matrix shape: {X.shape}")
    
    # Step 4: Split data
    print("\n4. Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    
    # Step 5: Train models
    print("\n5. Training models...")
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"   Logistic Regression trained")
    
    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print(f"   Naive Bayes trained")
    
    # Step 6: Evaluate models
    print("\n6. Model Results:")
    print("=" * 40)
    
    print("LOGISTIC REGRESSION:")
    print(f"Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_pred))
    print("Classification Report:")
    print(classification_report(y_test, lr_pred, target_names=['Fake', 'Real']))
    
    print("\nNAIVE BAYES:")
    print(f"Accuracy: {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, nb_pred))
    print("Classification Report:")
    print(classification_report(y_test, nb_pred, target_names=['Fake', 'Real']))
    
    # Step 7: Compare models
    print("\nMODEL COMPARISON:")
    print(f"Logistic Regression: {lr_accuracy*100:.2f}%")
    print(f"Naive Bayes: {nb_accuracy*100:.2f}%")
    
    if lr_accuracy > nb_accuracy:
        best_model = lr_model
        best_name = "Logistic Regression"
        print(f"Winner: Logistic Regression")
    else:
        best_model = nb_model
        best_name = "Naive Bayes"
        print(f"Winner: Naive Bayes")
    
    # Step 8: Prediction function
    def predict_news(article_text):
        processed = preprocess_text(article_text)
        if not processed:
            return "Unable to process text", 0.5
        
        vector = vectorizer.transform([processed])
        prediction = best_model.predict(vector)[0]
        probability = best_model.predict_proba(vector)[0]
        confidence = max(probability)
        
        result = "REAL" if prediction == 1 else "FAKE"
        return result, confidence
    
    # Step 9: Sample predictions
    print("\n7. Sample Predictions:")
    print("=" * 40)
    
    samples = [
        "Scientists discover new planet with water",
        "Aliens land in New York demanding pizza",
        "Stock market rises after Fed announcement",
        "Man flies after eating magic beans",
        "Climate change affects global temperatures"
    ]
    
    for i, sample in enumerate(samples, 1):
        result, confidence = predict_news(sample)
        print(f"Sample {i}: {result} ({confidence*100:.1f}%)")
        print(f"Text: {sample}")
        print()
    
    # Step 10: Interactive prediction
    print("8. Interactive Prediction:")
    print("=" * 40)
    print("Enter news articles to classify (type 'quit' to exit)")
    
    while True:
        user_input = input("\nEnter article: ")
        if user_input.lower() == 'quit':
            break
        
        if len(user_input.strip()) < 10:
            print("Please enter a longer article")
            continue
        
        result, confidence = predict_news(user_input)
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2f} ({confidence*100:.1f}%)")
        print(f"Model: {best_name}")
    
    print("\nProject completed successfully!")
    print(f"Final accuracies:")
    print(f"- Logistic Regression: {lr_accuracy*100:.2f}%")
    print(f"- Naive Bayes: {nb_accuracy*100:.2f}%")

if __name__ == "__main__":
    main()