
# coding: utf-8

# In[1]:

import nltk
import pandas as pd
import numpy as np


# In[2]:

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#from sklearn.pipeline import Pipeline


# In[3]:

#import sys
#import os
#import argparse
#from scipy.sparse import csr_matrix
import six
from abc import ABCMeta
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy import sparse


# In[4]:

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC


# In[5]:

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras import backend as K
#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')


# In[6]:

# https://www.kaggle.com/aaron7sun/stocknews/data
data = pd.read_csv('D:/DA/PGDBA/IIT/BM69006_DATA_SCIENCE_LABORATORY/dsl_project/data/stocknews/Combined_News_DJIA.csv')
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']


# In[7]:

data.head()


# In[8]:

#transforming the news headlines to number of words as input
trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))


# In[9]:

#repeating the preprossesing steps for test set
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))


# In[10]:

#visualising the append
print(trainheadlines[0])


# In[11]:

#using basic vectorizer preprocessing : that is just using the count of the each word as vectors
basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
basictest = basicvectorizer.transform(testheadlines)
#checking the size of the vector we have created
print(basictrain.shape, basictest.shape)


# In[17]:

print basictrain.todense()[2]


# In[79]:

#advance preprocessing
#advancedvectorizer = TfidfVectorizer()
#advancedvectorizer = TfidfVectorizer( min_df=0.01, max_df=1, ngram_range = (2, 2))
advancedvectorizer = TfidfVectorizer( min_df=0.03, max_df=0.97, ngram_range = (2, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
advancedtest = advancedvectorizer.transform(testheadlines)
print(advancedtrain.shape, advancedtest.shape)


# In[19]:

#advanced model 2 : trigrams
advancedvectorizer3 = TfidfVectorizer( min_df=0.0039, max_df=0.17, ngram_range = (3, 3))
advancedtrain3 = advancedvectorizer3.fit_transform(trainheadlines)
advancedtest3 = advancedvectorizer3.transform(testheadlines)

print(advancedtrain3.shape, advancedtest3.shape)


# In[20]:

basicmodel = LogisticRegression()
basicmodel = basicmodel.fit(basictrain, train["Label"])

#doing the predictions
preds1 = basicmodel.predict(basictest)
#measing the accuracy
acc1=accuracy_score(test['Label'], preds1)
print('Logic Regression 1 accuracy: ',acc1 )


# In[80]:

advancedmodel = LogisticRegression()
advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])
preds2 = advancedmodel.predict(advancedtest)
acc2=accuracy_score(test['Label'], preds2)
print('Logic Regression 2 accuracy: ', acc2)


# In[22]:

advancedmodel3 = LogisticRegression()
advancedmodel3 = advancedmodel3.fit(advancedtrain3, train["Label"])
preds3 = advancedmodel3.predict(advancedtest3)
acc3 = accuracy_score(test['Label'], preds3)
print('Logic Regression 3 accuracy: ', acc3)


# In[25]:

#Naive bayes model with basic pre processing
basicmodel = MultinomialNB(alpha=0.0001)
basicmodel = basicmodel.fit(basictrain, train["Label"])
preds4 = basicmodel.predict(basictest)
acc4 = accuracy_score(test['Label'], preds4)
print('NBayes 2 accuracy: ', acc4)


# In[26]:

#Naive bayes model with advance pre processing
advancedmodel = MultinomialNB(alpha=0.0001)
advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])
preds5 = advancedmodel.predict(advancedtest)
acc5 = accuracy_score(test['Label'], preds5)
print('NBayes 2 accuracy: ', acc5)


# In[27]:

#Naive bayes model with advance pre processing
advancedmodel3 = MultinomialNB(alpha=0.0001)
advancedmodel3 = advancedmodel3.fit(advancedtrain3, train["Label"])
preds6 = advancedmodel3.predict(advancedtest3)
acc6 = accuracy_score(test['Label'], preds6)
print('NBayes 3 accuracy: ', acc6)


# In[28]:

#SVM

class NBSVM(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):

    def __init__(self, alpha=1.0, C=1.0, max_iter=10000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.C = C
        self.svm_ = [] # fuggly

    def fit(self, X, y):
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # so we don't have to cast X to floating point
        Y = Y.astype(np.float64)

        # Count raw events from data
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.ratios_ = np.full((n_effective_classes, n_features), self.alpha,
                                 dtype=np.float64)
        self._compute_ratios(X, Y)

        # flugglyness
        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            svm = LinearSVC(C=self.C, max_iter=self.max_iter)
            #svm = SVC(kernel='sigmoid', C=self.C, max_iter=self.max_iter)
            Y_i = Y[:,i]
            svm.fit(X_i, Y_i)
            self.svm_.append(svm) 

        return self

    def predict(self, X):
        n_effective_classes = self.class_count_.shape[0]
        n_examples = X.shape[0]

        D = np.zeros((n_effective_classes, n_examples))

        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            D[i] = self.svm_[i].decision_function(X_i)
        
        return self.classes_[np.argmax(D, axis=0)]
        
    def _compute_ratios(self, X, Y):
        """Count feature occurrences and compute ratios."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")

        self.ratios_ += safe_sparse_dot(Y.T, X)  # ratio + feature_occurrance_c
        normalize(self.ratios_, norm='l1', axis=1, copy=False)
        row_calc = lambda r: np.log(np.divide(r, (1 - r)))
        self.ratios_ = np.apply_along_axis(row_calc, axis=1, arr=self.ratios_)
        check_array(self.ratios_)
        self.ratios_ = sparse.csr_matrix(self.ratios_)

        #p_c /= np.linalg.norm(p_c, ord=1)
        #ratios[c] = np.log(p_c / (1 - p_c))


# In[29]:

svmbasicmodel = NBSVM(C=0.01)
svmbasicmodel = svmbasicmodel.fit(basictrain, train["Label"])
preds12 = svmbasicmodel.predict(basictest)
acc12 = accuracy_score(test['Label'], preds12)
print('NBSVM 1: ', acc12)


# In[78]:

svmadvancedmodel = NBSVM(C=0.01)
svmadvancedmodel = svmadvancedmodel.fit(advancedtrain, train["Label"])
preds13 = svmadvancedmodel.predict(advancedtest)
acc13 = accuracy_score(test['Label'], preds13)
print('NBSVM 1: ', acc13)


# In[37]:

svmadvancedmodel = NBSVM(C=0.01)
svmadvancedmodel = svmadvancedmodel.fit(advancedtrain3, train["Label"])
preds14 = svmadvancedmodel.predict(advancedtest3)
acc14 = accuracy_score(test['Label'], preds14)
print('NBSVM 1: ', acc14)


# In[32]:

#further improvement
advancedvectorizer4 = TfidfVectorizer( min_df=0.031, max_df=0.2, ngram_range = (2, 2))
advancedtrain4 = advancedvectorizer4.fit_transform(trainheadlines)
print(advancedtrain4.shape)


# In[33]:

advancedmodel = NBSVM(C=0.01)
advancedmodel = advancedmodel.fit(advancedtrain4, train["Label"])
advancedtest4 = advancedvectorizer4.transform(testheadlines)
preds15 = advancedmodel.predict(advancedtest4)
acc15 = accuracy_score(test['Label'], preds15)
print('NBSVM 2: ', acc15)


# In[106]:

type(preds15)


# In[86]:

#Deep Learning
np.random.seed(88)

batch_size = 32
nb_classes = 2


advancedvectorizer5 = TfidfVectorizer( min_df=0.04, max_df=0.3, ngram_range = (2, 2))
advancedtrain5 = advancedvectorizer5.fit_transform(trainheadlines)
advancedtest5 = advancedvectorizer5.transform(testheadlines)
print(advancedtrain5.shape)



X_train = advancedtrain5.toarray()
X_test = advancedtest5.toarray()

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(train["Label"])
y_test = np.array(test["Label"])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.mean(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]

model = Sequential()
model.add(Dense(256, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',  metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam')
#model.compile(loss='binary_crossentropy', optimizer='rmsprop')
#model.compile(loss='binary_crossentropy', optimizer='adam')

print("Training...")
model.fit(X_train, Y_train, nb_epoch=10, batch_size=100, validation_split=0.15, verbose=1)

print("Generating test predictions...")
preds20 = model.predict_classes(X_test, verbose=0)
acc20 = accuracy_score(test["Label"], preds20)
print('prediction accuracy: ', acc20)


# In[95]:

#Deep Learning  basic
np.random.seed(88)

batch_size = 32
nb_classes = 2


advancedvectorizer5 = TfidfVectorizer()# min_df=0.04, max_df=0.3, ngram_range = (3, 3))
advancedtrain5 = advancedvectorizer5.fit_transform(trainheadlines)
advancedtest5 = advancedvectorizer5.transform(testheadlines)
print(advancedtrain5.shape)



X_train = advancedtrain5.toarray()
X_test = advancedtest5.toarray()


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(train["Label"])
y_test = np.array(test["Label"])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# pre-processing: divide by max and substract mean
scale = np.max(X_train)
scale = scale.astype(int)
X_train /= scale
X_test /= scale

mean = np.mean(X_train, dtype=np.int64)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]

model = Sequential()
model.add(Dense(256, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',  metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam')
#model.compile(loss='binary_crossentropy', optimizer='rmsprop')
#model.compile(loss='binary_crossentropy', optimizer='adam')

print("Training...")
model.fit(X_train, Y_train, nb_epoch=10, batch_size=100, validation_split=0.15, verbose=1)

print("Generating test predictions...")
preds20 = model.predict_classes(X_test, verbose=0)
acc20 = accuracy_score(test["Label"], preds20)
print('prediction accuracy: ', acc20)


# In[88]:

type(X_train)


# In[70]:

#ensemble by max votes
predsvotes = np.vstack((preds2, preds3, preds4, preds5, preds6, preds13, preds14, preds15, preds20)).T


# In[71]:

from scipy.stats import mode
x = []
for i in range(len(test)):
    x.append(mode(predsvotes[i])[0][0])
accvotes = np.array(x)


# In[72]:

test['Label'].shape


# In[73]:

accv = accuracy_score(test['Label'], accvotes)
print('voted acc ', accv)
