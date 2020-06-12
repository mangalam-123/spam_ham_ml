import jinja2
from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from logging import FileHandler, WARNING
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer

app = Flask(__name__)
if not app.debug:
    file_handle = FileHandler('errorlog.txt')
    file_handle.setLevel(WARNING)

    app.logger.addHandler(file_handle)



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/email')
def index():
    return render_template('email.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('spam_or_not_spam.csv', encoding="utf-8")
    df['email'].fillna("NO message", inplace=True)
    df['label'].fillna("NO message", inplace=True)

    df["new_email"] = df['email'].str.replace('[^\w\s]', '')
    X = df['new_email']
    y = df['label']

    cv = CountVectorizer()
    tfidf = TfidfTransformer()
    X = cv.fit_transform(X)
    # X = tfidf.fit_transform(X)

    # Fit the Data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    # Getting data from Post method

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vector = cv.transform(data).toarray()
        my_prediction = clf.predict(vector)

    return render_template('result.html', prediction=my_prediction)


if __name__ == "__main__":
    app.run()
