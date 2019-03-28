# -*- coding: utf-8 -*-
'''
Author: Ian Shen
Github: ianshan0915
Date: March 28, 2019

'''

import os
import warnings
import sys

# import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    rec = recall_score(actual, pred, average="weighted")
    f1 = f1_score(actual, pred, average="weighted")
    return acc, rec, f1



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load data from sklearn
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)


    max_percent = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
    min_count = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    num_feats = int(sys.argv[3]) if len(sys.argv) > 3 else None

    with mlflow.start_run():
        # NLP pipeline + prediction modeling
        vectModel = CountVectorizer(max_df = max_percent, min_df = min_count, max_features=num_feats)
        tfdfModel = TfidfTransformer()
        nlp_clf = Pipeline([
            ('vect', vectModel),
            ('tfidf', tfdfModel),
            ('clf', MultinomialNB()),
        ])
        nlp_clf.fit(train.data, train.target)

        predicted = nlp_clf.predict(test.data)

        (acc, rec, f1) = eval_metrics(test.target, predicted)

        print("nlp pipeline (max_df=%f, min_df=%s):" % (max_percent, min_count))
        print("  ACCURACY: %s" % acc)
        print("  RECALL: %s" % rec)
        print("  F1: %s" % f1)

        mlflow.log_param("max_df", max_percent)
        mlflow.log_param("min_df", min_count)
        mlflow.log_param("num_features", num_feats)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1-score", f1)

        # mlflow.sklearn.log_model(lr, "model")
