import re

def clean_text(text):
    text = str(text)
    text = text.lower()

    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("Fake Dataset Shape:")
print(fake.shape)

print("\nTrue Dataset Shape:")
print(true.shape)

print("\nColumns:")
print(fake.columns)

print("\nDataset Info:")
print(fake.info())

# Labels Add Karna
fake["label"] = 0
true["label"] = 1

print("\nFake Dataset:")
print(fake.head())

print("\nTrue Dataset:")
print(true.head())

# Merge

data = pd.concat([fake, true], axis=0)

# Shuffle

data = data.sample(frac=1, random_state=42)

# Reset Index

data.reset_index(drop=True, inplace=True)

print("\nMerged Dataset Shape:")
print(data.shape)

print(data.head())

# Text Cleaning Apply

data["text"] = data["text"].apply(clean_text)

print("\nCleaned Text Sample:\n")
print(data["text"].iloc[0][:500])

# TF-IDF Vectorization

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["text"])

y = data["label"]

print("\nTF-IDF Shape:")
print(X.shape)

print("\nLabels Shape:")
print(y.shape)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# Train Model

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

import pickle

pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved Successfully!")

print("\nModel Training Complete!")

# Prediction

y_pred = model.predict(X_test)

print("\nPrediction Complete!")

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print(f"\nAccuracy: {accuracy*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

news = input("\nEnter News Text: ")

news = clean_text(news)

news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)

if prediction[0] == 0:
    print("\nFake News")
else:
    print("\nReal News")

