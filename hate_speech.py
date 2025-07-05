# 📦 Import libraries
import pandas as pd
import numpy as np
import string
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ Download stopwords if not already
nltk.download('stopwords')

# 📂 Load Data
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

print("📊 Sample data:")
print(train_df.head())

# 🧹 Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# 🧼 Apply cleaning
train_df['clean_text'] = train_df['tweet'].apply(clean_text)
test_df['clean_text'] = test_df['tweet'].apply(clean_text)
print("✅ Text cleaning done!")

# 🔠 Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])

y_train = train_df['label']

# 🧠 Model training
model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)

# ✅ Prediction only (test set has no labels)
predictions = model.predict(X_test)
print("\n📈 Predictions on test set (first 10):", predictions[:10])

# 🧪 Evaluate on train set (for reference)
train_preds = model.predict(X_train)
print("\n🎯 Training Accuracy:", accuracy_score(y_train, train_preds))
print("\n📊 Classification Report on Train Data:\n", classification_report(y_train, train_preds))

# 📉 Confusion Matrix on training data
sns.heatmap(confusion_matrix(y_train, train_preds), annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix (Train Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
import joblib

# Save model and vectorizer
joblib.dump(model, 'hate_speech_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("✅ Model and vectorizer saved successfully!")
