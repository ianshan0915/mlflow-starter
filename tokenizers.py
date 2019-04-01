# -*- coding: utf-8 -*-
'''
Author: Ian Shen
Github: https://github.com/ianshan0915
Date: March 29, 2019

This is an NLP pipeline based on SpaCy, the pipeline will be used as tokenizer for CountVectorizer
'''

import spacy
spacy.load('en_core_web_sm')

lemmatizer = spacy.lang.en.English()


# nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

def token_filter(token):
  return not (token.is_punct | token.is_stop | len(token.text.strip()) <= 3 )

def spacy_tokenizer(input):
  tokens = lemmatizer(input)
  lemmas = [token.lemma_ for token in tokens if token_filter(token) and token.lemma_.strip() and token.is_alpha]
  # tokens = [ent.text for ent in doc.ents]
  # tokens = []
  # for token in doc:
  #   if token.pos_ in ["PROPN", "VERB", "NOUN", "ADJ","NUM","ADV"] and token.is_alpha and not token.is_stop:
  #     tokens.append(token.lemma_.lower().strip())
  return lemmas

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        lemmas = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        return [lemma for lemma in lemmas if len(lemma)>3]