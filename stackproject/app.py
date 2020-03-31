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
#from sklearn.externals import joblib
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
from sklearn.cluster import AgglomerativeClustering
import re
import textdistance
from bs4 import BeautifulSoup
from token import *
token=ToktokTokenizer()
punct = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
lemma=WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
vector1 = pickle.load(open('vector1.pkl','rb'))
vector2= pickle.load(open('vector2.pkl','rb'))
tagst= pickle.load(open('tagst.pkl','rb'))
clf= pickle.load(open('clf.pkl','rb'))
multilabel_binarizer= pickle.load(open('multilabel_binarizer.pkl','rb'))

def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))

def stopWordsRemove(text):

    stop_words = set(stopwords.words("english"))

    words=token.tokenize(text)

    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered))

def clean_punct(text):
    words=token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    for w in words:
        if w in tagst:
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))

    filtered_list = strip_list_noempty(punctuation_filtered)

    return ' '.join(map(str, filtered_list))

def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		comment = request.form['comment']
		four=stopWordsRemove(comment)
		three=lemitizeWords(four)
		two=clean_text(three)
		one=clean_punct(two)
		data=([one])
		data1 =([one])
		#data=[comment]
		#data1=[comment]
		fitted=vector1.transform(data)
		fitted2=vector2.transform(data1)
		x_t = hstack([fitted,fitted])
		my_prediction = clf.predict(x_t)
	res = []
	for labels in(my_prediction):
		for i in range(len(labels)):
			if(labels[i]==1):
				res.append(multilabel_binarizer.classes_[i])
	#for i in range(my_prediction.shape[1]):
	#	for j in range(len(my_prediction))
	#		if(my_prediction[i][j])
	#	res.append(multilabel_binarizer.classes_[i])

	return render_template('result.html',res=res)



if __name__ == '__main__':
  app.run(debug=True)
