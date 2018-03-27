# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 23:21:16 2018

@author: Puneeth
Implementation of textrank algorithm for summarization
"""

import PyPDF2
import networkx as nx
import numpy as np
 
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
 
filename = 'D:/DA/PGDBA/IIT/BM69006_DATA_SCIENCE_LABORATORY/references/Text_Summarization_Techniques-A_Brief_Survey.pdf' 
pdfFileObj = open(filename,'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
num_pages = pdfReader.numPages
count = 0
text = ""

#The while loop will read each page
while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()
document = ' '.join(text.strip().split('\n'))

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


textrank(document)[0:5]



