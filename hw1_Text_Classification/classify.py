#!/bin/python
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_classifier(X, y,c):
	"""Train a classifier using the given training data.

	Trains a logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	#param_grid = [{'C': [0.5,0.7,0.8,0.9, 1,5,10,50,100]}]
	#cls = LogisticRegression(penalty='l2',multi_class =  'ovr', solver = 'lbfgs', C = 8.1)#tfidf #.96#0.40#0.41516

	cls = LogisticRegression(penalty='l2', multi_class='ovr', class_weight ='balanced',solver='lbfgs', C = c)


	#cls = LogisticRegression(penalty='l2',class_weight ='balanced',solver='lbfgs') #.96#0.40#0.41516


	# param_grid = [{'tol': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001], 'C': [0.5, 1, 2.5, 10, 50]}]

	#cls = GridSearchCV(LogisticRegression(penalty='l2', solver='lbfgs',multi_class =  'ovr'), param_grid)
	#cls = LogisticRegression()
	# The conjugate gradient method is often implemented as an iterative algorithm, applicable to sparse systems
	cls.fit(X, y)
	#print(cls.best_params_)
	return cls

def evaluate(X, yt, cls):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	m = metrics.confusion_matrix(yt,yp)
	print("  Accuracy", acc)
	return m,acc
