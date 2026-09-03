from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset
df = pd.read_csv("news_dataset.csv")
df = df.dropna(subset=["label", "text"])

# Convert labels: REAL=1, FAKE=0
df["label"] = df["label"].map({"REAL": 1, "FAKE": 0})
df = df.dropna(subset=["label"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Test accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    news = ""

    if request.method == "POST":
        news = request.form.get("news", "").strip()

        if news:
            news_tfidf = vectorizer.transform([news])
            prediction_value = model.predict(news_tfidf)[0]
            probabilities = model.predict_proba(news_tfidf)[0]
            confidence = max(probabilities) * 100

            if prediction_value == 1:
                prediction = "REAL NEWS"
            else:
                prediction = "FAKE NEWS"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        accuracy=accuracy * 100,
        news=news
    )


if __name__ == "__main__":
    app.run(debug=True)
