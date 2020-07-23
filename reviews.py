import time
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from preproc import preprocessing


def tfidf(token_list):
    texts = []
    for token in token_list:
        text = " ".join(i for i in token).strip()
        texts.append(text)
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    return vec, X


def fit_models(train_x, train_y, test_x, test_y):
    log = LogisticRegression()
    log.fit(train_x, train_y)
    print("Logistic ->", (log.predict(test_x) == test_y).sum() / len(test_y))
    multinb = MultinomialNB()
    multinb.fit(train_x, train_y)
    print(
        "Multinomial Naive Bayes ->",
        (multinb.predict(test_x) == test_y).sum() / len(test_y),
    )
    bernb = BernoulliNB()
    bernb.fit(train_x, train_y)
    print(
        "Bernouili Naive Bayes ->",
        (bernb.predict(test_x) == test_y).sum() / len(test_y),
    )


if __name__ == "__main__":
	
    file_path = "/content/IMDB Dataset.csv"
    df = pd.read_csv(file_path)
    df = df.sample(frac=1).reset_index(drop=True)
    df["labels"] = df["sentiment"].apply(lambda row: 1 if row == "positive" else 0)
    train_data = df[:-1000]
    test_data = df[-1000:]
    train_y = train_data["labels"]
    test_y = test_data["labels"]

    train_tokens = Parallel(n_jobs=2, backend="multiprocessing")(
        delayed(preprocessing)(text)
        for i, text in tqdm(enumerate(train_data["review"]), total=len(train_data))
    )

    test_tokens = Parallel(n_jobs=2, backend="multiprocessing")(
        delayed(preprocessing)(text)
        for i, text in tqdm(enumerate(test_data["review"]), total=len(test_data))
    )

    vect, train_x = tfidf(train_tokens)

    texts = []
    for token in test_tokens:
        text = " ".join(i for i in token).strip()
        texts.append(text)
    test_x = vect.transform(texts)

    fit_models(train_x, train_y, test_x, test_y)
