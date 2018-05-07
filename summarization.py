# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 20:02:40 2018

@author: User
"""

import pandas as pd
import csv
data1 = pd.read_csv("F:/DSL/bloomberg_news")

from pathlib import Path # import 'pathlib'
directory_in_str = "E:/xyz" # provide name of your directory
pathlist = Path(directory_in_str).glob('*') # read all files in this directory

for path in pathlist:
    path_in_str = str(path) # convert path into string object
    for i in path_in_str:
        print(i)# run your function on each file in the folder
    append.results # append your results to this variable one by one 
    

def getText(path_in_str):
    file_open = open(path_in_str, encoding = 'utf-8') # open each file 
    file_read = file_open.read() # read each file
    file_open.close() # close each file
    
file_open=open("F:/DSL/bloomberg_news")


from pathlib import Path # import 'pathlib'
directory_in_str = "F:/bnews/" # provide name of your directory
pathlist = Path(directory_in_str).glob('*')
from pathlib import Path # import 'pathlib'
directory_in_str = "F:/bnews/" # provide name of your directory
pathlist = Path(directory_in_str).glob('*') # read all files in this directory
for path in pathlist:
    path_in_str = str(path) 
    file_list=[]# convert path into string object
    r=print(path_in_str)
    append.file_list(r)
with open ("E:/xyz/doc1.txt", "r") as myfile:
    data=myfile.read().replace('\n', '')
        
from pathlib import Path # import 'pathlib'
directory_in_str = "E:/xyz" # provide name of your directory
pathlist = Path(directory_in_str).glob('*') # read all files in this directory
for path in pathlist:
    path_in_str = str(path) # convert path into string object
    for i in path_in_str:
        with open ("directory_in_str/", "r") as myfile:
        data=myfile.read()
        append.data
        
import sys
import glob
import errno

path = 'E:/xyz/*'   
files = glob.glob(path)   
for name in files: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
    try:
        with open(name) as f: # No need to specify 'r': this is the default.
            sys.stdout.write(f.read())
    except IOError as exc:
        if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
            raise # Propagate other kinds of IOError.
        
        
        
## final thing
import glob   
path = 'E:/bnews/*'   
files=glob.glob(path)  
result=[] 
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

import gensim
from gensim import corpora
import os
for file in files:
    newpath=file
    newfiles=glob.glob(newpath)
    for files in newfiles :
        f=open(files, 'r')  
    #m=f.readlines()
    # store final result in this
    text=f.read()
    f.close()
    text=clean(text)
    dictionary=corpora.Dictionary(text)
    
    
    
    result.append(m)
 
        
        
        
        
        
        
import os

path = "F:/bnews/*"
files=  glob.glob(path)
# Check current working directory.
retval = os.getcwd()
#print "Current working directory %s" % retval
n=len(files)
for i in range (0,n):
    subdirectory=files[i]
    pathlist=Path(subdirectory).glob('*')
    for path in pathlist:
        path_in_str=str(path)
        date=path_in_str[9:18]
        text=path_in_str[20:]
        text.read()
        with open ('path_in_str','r') as myfile:
            ace=myfile.read().replace('\n','')
    
    
    
    
    
    
    
    newfiles=glob.glob(path)
    
    newpath=os.chdir(path)  
        
        
from pathlib import Path # import 'pathlib'
directory_in_str = "your_directory_name" # provide name of your directory
pathlist = Path(directory_in_str).glob('*') # read all files in this directory

for path in pathlist:
    path_in_str = str(path) # convert path into string object
    your_function(path_in_str) # run your function on each file in the folder
    append.results # append your results to this variable one by one        
        
        
        
        
for file in folder:

pathlist=glob.glob('F:/bnews/*/*')
for path in pathlist:
    f=open(path,'r')
    text=f.read()
    path_in_str=str(path)
    print(path_in_str)
    
    
        
        
 file = open(filename, encoding="utf8")       
        
        
## This is working

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized   
import networkx as nx
import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
def textrank(document):
    
    # Sentence Splitting
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(document)
 
    # Converting to a Graph
    bow_matrix = CountVectorizer().fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
 
    similarity_graph = normalized * normalized.T
    
    # Pagerank
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    return sorted(((scores[i],s) for i,s in enumerate(sentences)),
                  reverse=True)   

import glob 
pathlist=glob.glob('E:/bnews/*/*')
result1=[]

for path in pathlist:
    #print(path)
    pathname=path
    date=pathname[9:19]
    
    f=open(path,encoding="utf8")
    #firstline=f.readline()
    text=f.read()
    #text=text[6:]        
    #date=pathlist[][9:18]  
    #text=text[0:4]
    text=clean(text)
    m=textrank(text)[0]# getting top 
    summary=(date,m[1])
    result1.append(summary)
    

import csv

#csvfile = "E:/news_summary.csv"

csvfile1 = "E:/news1.csv"
#Assuming res is a flat list
with open(csvfile1, "w",encoding="utf-8") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in result1:
        writer.writerow([val])    

#Assuming res is a list of lists
#with open(csvfile, "w") as output:
 #   writer = csv.writer(output, lineterminator='\n')
  #  writer.writerows(res)       
        
    
#def remove(input_file):
 #   for i, line in enumerate(input_file):
   #     if i == 0:
  #          output.write(line)
    #    else:
     #       if not line.startswith('#'):
      #          output.write(line)   
                    
                    
#remove("F:/bnews/2006-10-20/")                    
                    
                    
'''                    
with open('file.txt', 'r') as fin:
    data = fin.read().splitlines(True)
with open('file.txt', 'w') as fout:
    fout.writelines(data[1:])
    
'''
'''   
import textblob as tb
import tqdm
from tqdm import tqdm
import pandas as pd                  
def sent(x):
    t = tb.TextBlob(x)
    return t.sentiment.polarity, t.sentiment.subjectivity
df = pd.read_csv('C:/Users/User/Desktop/Bloomberg_news.csv')
file="C:/Users/User/Desktop/Bloomberg_news.csv"
sent(df[2])
'''                   
                    
import csv
from textblob import TextBlob

infile = 'C:/Users/User/Desktop/Bloomberg_news.csv'
result=[]
with open(infile, 'r',encoding="utf-8") as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        sentence = row[1]
        blob = TextBlob(sentence)
        result.append(blob.sentiment)                 
                    
csvfile = "E:/news_sentiment.csv"

with open(csvfile, "w",encoding="utf-8") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in result:
        writer.writerow([val]) 
                  

#headline extraction
import glob 
pathlist=glob.glob('E:/bnews/*/*')

headline=[]
for path in pathlist:
    #print(path)
    pathname=path
    date=pathname[9:19]
    f=open(path,encoding="utf8")
    firstline=f.readline()
    m=(date,firstline)
    headline.append(m)
    
csvfile = "E:/news_headlines.csv"

with open(csvfile, "w",encoding="utf-8") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in headline:
        writer.writerow([val])    








                    
                    

