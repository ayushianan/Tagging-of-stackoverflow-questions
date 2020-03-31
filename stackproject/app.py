import pandas as pd
from flask import Flask,render_template,url_for,request
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.lines as mlines
from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import seaborn as sns
import warnings
import pickle
import time
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.cluster import KMeans
import logging
import os
from scipy.sparse import hstack


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
vector1 = pickle.load(open('vector1.pkl','rb'))
vector2= pickle.load(open('vector2.pkl','rb'))
tagst= pickle.load(open('tagst.pkl','rb'))





@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		fitted=vector1.transform(data)
		fitted2=vector2.transform(data)
		x_t = hstack([fitted,fitted2])
		my_prediction = model.predict(x_t)
	res = []
	xstr1 = ""
	for labels in (my_prediction):
		for i in range(len(labels)):
			if labels[i] == 1:
				res.append(tagst[i])
	return render_template('result.html',res=res)



if __name__ == '__main__':
  app.run(debug=True)
