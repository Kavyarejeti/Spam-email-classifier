import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Clean text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

df['clean_text'] = df['text'].apply(clean_text)

# Convert labels to 0/1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict custom messages
def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])
    pred = model.predict(msg_vec)
    return "Spam" if pred[0] == 1 else "Ham"

# Example
print(predict_message("Congratulations! You won a free ticket. Reply now!"))
print(predict_message("Hey, are we still meeting tomorrow?"))
