# Fake News Detection Website

## Files

Place these files in one folder:

    fake_news_website/
    ├── app.py
    ├── news_dataset.csv
    ├── requirements.txt
    ├── templates/
    │   └── index.html
    └── static/
        └── style.css

The `news_dataset.csv` file must contain these columns:

- `label`
- `text`

Labels should be `REAL` and `FAKE`.

## Run in VS Code

Open the project folder in VS Code and open Terminal.

Install dependencies:

    pip install -r requirements.txt

Start the website:

    python app.py

You should see:

    Running on http://127.0.0.1:5000

Open that address in your browser.

## How it works

1. Loads `news_dataset.csv`
2. Splits the data into training and test sets
3. Converts text to TF-IDF features
4. Trains a Logistic Regression classifier
5. Calculates test accuracy
6. Accepts news through the webpage
7. Predicts REAL NEWS or FAKE NEWS
8. Shows model confidence

Note: Model confidence is a probability estimate from the classifier, not proof that an article is factually true or false.
