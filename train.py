# -*- coding: utf-8 -*-
'''
Author: Ian Shen
Github: https://github.com/ianshan0915
Date: March 28, 2019

'''

import os 
import warnings
import sys
import re
from time import time

# import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from tokenizers import spacy_tokenizer, LemmaTokenizer

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
    start_time = time()

    # Load data from sklearn
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)


    max_percent = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
    min_count = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    num_feats = int(sys.argv[3]) if len(sys.argv) > 3 else None
    nlp_type = sys.argv[4] if len(sys.argv) > 4 else "sklearn"

    if nlp_type =='nltk':
        tokenizer = LemmaTokenizer()
    elif nlp_type =='spacy':
        tokenizer = spacy_tokenizer
    else:
        tokenizer = None

    with mlflow.start_run():
        # NLP pipeline + prediction modeling
        vectModel = CountVectorizer(max_df = max_percent, 
                                    min_df = min_count, 
                                    max_features=num_feats,
                                    tokenizer = tokenizer,
                                    ngram_range=(1,2),
                                    stop_words="english")
        tfdfModel = TfidfTransformer()
        nlp_clf = Pipeline([
            # ('cleanText', CleanTextTransformer()),
            ('vect', vectModel),
            ('tfidf', tfdfModel),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                alpha=1e-3, random_state=42,
                                max_iter=5, tol=None)),
        ])
        nlp_clf.fit(train.data, train.target)
        training_time = time() - start_time

        vect_features = ",".join(nlp_clf.named_steps["vect"].get_feature_names())
        stop_words = ",".join(nlp_clf.named_steps["vect"].get_stop_words())

        predicted = nlp_clf.predict(test.data)

        (acc, rec, f1) = eval_metrics(test.target, predicted)

        print("feature names: %s" % vect_features)
        print("stop words: %s" % stop_words)
        print("\n")
        print("nlp pipeline (max_df=%f, min_df=%s):" % (max_percent, min_count))
        print("  Duration: %0.3f" % training_time)
        print("  ACCURACY: %s" % acc)
        print("  RECALL: %s" % rec)
        print("  F1: %s" % f1)

        mlflow.log_param("max_df", max_percent)
        mlflow.log_param("min_df", min_count)
        mlflow.log_param("num_features", num_feats)
        mlflow.log_param("nlp_type", nlp_type)
        mlflow.log_metric("duration", training_time)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1-score", f1)

        # mlflow.sklearn.log_model(lr, "model")
