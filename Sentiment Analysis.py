#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:02:10 2020

@author: omri
"""
# For this sentiment analysis project, i'm using an off-line movie review corpus 
# I'm implementing Naive Bayes or let's say Multinomial Naive Bayes classifier using NLTK which 
# stands for Natural Language Toolkit. It is a library dedicated to NLP and NLU related tasks

# Load and prepare the dataset
import nltk
# I'm starting by importing the movie reviews 
from nltk.corpus import movie_reviews
import random
# After that i have constructed a list of documents, labeled with the appropriate categories(Pos or Neg)
documents = [(list(movie_reviews.words(fileid)), category) 
for category in movie_reviews.categories() 
for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)


# Next, i have defined a feature extractor for documents, so the classifier will know which aspects of the
# data it should pay attention too
# In this case,i can define a feature for each word, indicating whether the document contains that word
# Define the feature extractor
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
# To limit the number of features that the classifier needs to process, i have started by constructing a list 
# of the 2000 most frequent words in the overall corpus
word_features = list(all_words)[:2000]

def document_features(document):
    # Checking whether a word occurs in a SET is much faster than checking whether it happens in a LIST
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# I have defined the feature extractor. Now, i'm going to use it to train a Naive Bayes classifier to predict
# the sentiments of new movie reviews
# Train Naive Bayes classifier
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
# Test the classifier
# I'm checking the classifier performance by computing its accuracy on the test set
print(nltk.classify.accuracy(classifier, test_set))
# NLTK provides show_most_informative_features() to see which features the classifier found to be most informative
# Show the most important features as interpreted by Naive Bayes
classifier.show_most_informative_features(10)
