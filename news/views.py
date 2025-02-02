from django.shortcuts import render
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download NLTK stopwords if not already available
nltk.download('stopwords')

# Load dataset
DATASET_PATH = r'news\Datasets\train.csv'
dataset = pd.read_csv(DATASET_PATH)

# Handle missing values
dataset.fillna('', inplace=True)

# Combine 'author' and 'title' into a single 'content' column
dataset['content'] = dataset['author'] + ' ' + dataset['title']

# Separate features and labels
X = dataset['content']
Y = dataset['label']

# Initialize Porter Stemmer
port_stem = PorterStemmer()

def preprocess_text(content):
    """
    Cleans and preprocesses the input text:
    - Removes special characters and numbers
    - Converts to lowercase
    - Tokenizes and removes stopwords
    - Applies stemming
    """
    content = re.sub(r'[^a-zA-Z]', ' ', content).lower().split()
    content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# Apply text preprocessing
X = X.apply(preprocess_text)

# Convert text data into numerical form using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate model accuracy
train_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)
print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

def predict_news(request):
    """
    Handles user input, processes it, and returns the prediction result.
    """
    if request.method == 'POST':
        news_text = request.POST.get('news_input', '')
        processed_text = preprocess_text(news_text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]
        result = '🚨 FAKE NEWS 🚨' if prediction == 0 else '✅ REAL NEWS ✅'
        return render(request, 'index.html', {'news_input': news_text, 'result': result})
    
    return render(request, 'index.html', {'result': None})
