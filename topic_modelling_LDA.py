# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:57:04 2018

@author: Puneeth
Extracting words from pdf and topic modelling
"""

import PyPDF2 
#import textract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


filename = 'D:/DA/PGDBA/IIT/BM69006_DATA_SCIENCE_LABORATORY/dsl_project/code/sample_text.pdf' 
filename1 = 'D:/DA/PGDBA/IIT/BM69006_DATA_SCIENCE_LABORATORY/dsl_project/code/doc1.pdf' 
filename2 = 'D:/DA/PGDBA/IIT/BM69006_DATA_SCIENCE_LABORATORY/dsl_project/code/doc2.pdf' 

pdfFileObj = open(filename,'rb')
pdfFileObj1 = open(filename1,'rb')
pdfFileObj2 = open(filename2,'rb')
#The pdfReader variable is a readable object that will be parsed
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pdfReader1 = PyPDF2.PdfFileReader(pdfFileObj1)
pdfReader2 = PyPDF2.PdfFileReader(pdfFileObj2)

#discerning the number of pages will allow us to parse through all #the pages
num_pages = pdfReader.numPages
count = 0
text = ""

num_pages1 = pdfReader1.numPages
count1 = 0
text1 = ""

num_pages2 = pdfReader2.numPages
count2 = 0
text2 = ""


#The while loop will read each page
while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()
    
#The while loop will read each page
while count1 < num_pages1:
    pageObj1 = pdfReader1.getPage(count1)
    count1 +=1
    text1 += pageObj1.extractText()
    
#The while loop will read each page
while count2 < num_pages2:
    pageObj2 = pdfReader2.getPage(count2)
    count2 +=1
    text2 += pageObj2.extractText()


#This if statement exists to check if the above library returned #words.
#It's done because PyPDF2 cannot read scanned files.
#if text != "":
#   text = text
#If the above returns as False, we run the OCR library textract to #convert scanned/image based PDF files into text
#else:
#   text = textract.process(fileurl, method='tesseract', language='eng')


# Now we have a text variable which contains all the text derived #from our PDF file. 
# Type print(text) to see what it contains. It 
# likely contains a lot of spaces, possibly junk such as '\n' etc.
# Now, we will clean our text variable, and return it as a list of keywords.


#The word_tokenize() function will break our text phrases into #individual words
tokens = word_tokenize(text)
#we'll create a new list which contains punctuation we wish to clean
punctuations = ['(',')',';',':','[',']',',']
#We initialize the stopwords variable which is a list of words like 
#"The", "I", "and", etc. that don't hold much value as keywords
stop_words = stopwords.words('english')
#We create a list comprehension which only returns a list of words 
#that are NOT IN stop_words and NOT IN punctuations.
keywords = [word for word in tokens if not word in stop_words and  not 
            word in punctuations]


doc_complete = [text1, text2]


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

doc_clean = [clean(doc).split() for doc in doc_complete]        


# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=200)

print(ldamodel.print_topics(num_topics = 5, num_words = 10))







