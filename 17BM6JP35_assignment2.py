# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:19:27 2018

@author: Puneeth G, Roll No - 17BM6JP35
Course : Information Retrieval(CS60092), Assignment 2
"""

###############################################################################
# Inputs required :
topic = 'Topic1' # Topics - 'Topic1', 'Topic2', 'Topic3', 'Topic4', and 'Topic5'
algorithm = 'degree_centrality' # algorithms - 'degree_centrality' and 'text_rank'
threshold = 0.1 # possible thresholds - 0.1, 0.2 and 0.3 
dir_path = 'D:/DA/PGDBA/IIT/CS60092_INFORMATION_RETRIEVAL/ass/assignment_2/'
    # provide link to directory in which inputs files exist
###############################################################################


# Run the rest of the code to get output printed

# load libraries
from bs4 import BeautifulSoup # import beautiful soup to read files
from pathlib import Path
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from rouge import Rouge # import Rouge package to calculate scores
import numpy as np # import pandas
from nltk.tokenize import sent_tokenize


# define function 'getText' to read text/sentences from files
def getText(path_in_str):
    file_open = open(path_in_str, encoding = 'utf-8') # open each file 
    file_read = file_open.read() # read each file
    file_open.close() # close each file
    soup = BeautifulSoup(file_read, "lxml").find_all('p') # get content for attributes 'p'
    sentences = [] # initialize string
    for i in soup: # for loop to read all senteces with attribute 'p'
        sentences.append(i.text.strip()) # strip and get text
    sentences = ' '.join(sentences).replace('\n', ' ') # join the sentences and remove "/n"
    return sentences


# From the input files, extract all sentences for a given topic
directory_in_str = dir_path + topic # path to files of a given topic
pathlist = Path(directory_in_str).glob('*') # read all files in this directory
result = [] # initialize string
for path in pathlist: # run loop for all files in the folder for the topic
    path_in_str = str(path) # convert path into string object
    result.append(getText(path_in_str)) # combine sentences from all files
result = ''.join(result).replace('\n', ' ') # join the sentences and remove '\n'



# Tokenize all sentences using natural language toolkit
sentences = sent_tokenize(result) # returns list of all sentences


# Build tf-idf matrix for all these sentences
# step 1 - construct matrix with rows as sentences and each columns as a word
count_matrix = CountVectorizer().fit_transform(sentences) 
# step 2 - using this count matrix create tf-idf matrix and normalize it
# tf-idf(d, t) = tf(t) * idf(d, t); idf(d, t) = log [ n / df(d, t) ] + 1
# where n is the total number of documents; df(d, t) is the document frequency
# and tf(t) is the term frequency
# And the document frequency is the number of documents d that contain term t
normalized = TfidfTransformer().fit_transform(count_matrix)


# construct similarity graph from normalized tf-idf matrix
# multiply matrix with its transpose
#similarity_matrix = normalized * normalized.T

# convert the existing sparse 'normalized' matrix to its dense form
normalized_dense = normalized.todense()
# compute similarity matrix using the normalized matrices
similarity_matrix = normalized_dense * normalized_dense.T


# Create adjacency matrix based on the similarity between sentences such that, 
# if the similarity is greater than threshold, replace with '1'. 
# If not, replace with '0'.

num_sent = similarity_matrix.shape[0] # total number of sentences 
adj_matrix = np.zeros((num_sent, num_sent)) # initialize adjacency matrix
for row in range(num_sent):
    for col in range(num_sent):
        if  similarity_matrix[row, col] >= threshold:
            adj_matrix[row, col] = 1
        else :
            adj_matrix[row, col] = 0
# adjacency matrix - 'adj_matrix' is created 


# calculate sum of each row in adjacency matrix
sum_rows_adj = np.array([sum(adj_matrix[row]) for row in
                     range(adj_matrix.shape[0])])
zero_sum_rows = np.where(sum_rows_adj == 0)[0] 
    # rows with no connections to other rows
nan_value_rows = np.where(np.isnan(sum_rows_adj))[0] 
    # rows with 'nan' values
del_rows =  np.concatenate([zero_sum_rows, nan_value_rows])
for i in sorted(del_rows, reverse = True): # delete unnecessary rows
    adj_matrix = np.delete(adj_matrix, [i], axis = 0)
    adj_matrix = np.delete(adj_matrix, [i], axis = 1)
    # modified adjancency matrix

##############################################################################
#############degree centrality based approach to generate summary#############
##############################################################################

if algorithm == 'degree_centrality' :
   
    # store the node with highest degree centrality in a list
    rank_deg_centr = [] # initialize this list
    temp_adj_mat = adj_matrix 
    # create a temporary adjacency matrix to use in the while loop below
    
    # run a while loop to extract nodes with highes degree centralities
    for j in range(100):    
        
        # calculate sum of each row in adjacency matrix
        sum_rows_adj = np.array([sum(temp_adj_mat[row]) for row in 
                                 range(temp_adj_mat.shape[0])])
        
        # rank nodes/sentences based on decreasing order of degree centrality
        nodes_deg_centr = [b[0] for b in sorted(enumerate(sum_rows_adj), reverse = True, key = lambda i:i[1])]
        
        # append nodes with highest degree centrality to this list
        rank_deg_centr.append(nodes_deg_centr[0])
        
        # Extract row in adjacency matrix corresponding to the node with
        # highest degree centrality
        adj_mat_row = temp_adj_mat[nodes_deg_centr[0]]
        
        # From this row, extract list of all nodes connected to this node
        nodes_conctd = np.nonzero(adj_mat_row)[0]
        
        #To this list, append the node with highest degree centrality as well
        nodes_conctd = np.append(nodes_conctd, nodes_deg_centr[0])
        
        # set all the elements (both in  rows and columns) corresponding to indices
        # in the list 'nodes_conctd' to zeros
        
        for i in sorted(nodes_conctd, reverse = True): # delete unnecessary rows
            temp_adj_mat[i] = np.zeros(temp_adj_mat.shape[0])
            temp_adj_mat[:, i] = np.zeros(temp_adj_mat.shape[0])
            # modified temporary adjancency matrix
    
    # ranked list of nodes using degree centrality based approach
    ranked_text = ' '.join([sentences[i] for i in rank_deg_centr])
    # join the sentences in the ranked order to form summary 
    
    # Extract 250-word summary and print
    smry_deg_cent = ' '.join(ranked_text.split()[:250]) # 250-word summary
    
    # print this only if the algorithm is 'degree_centrality'
    print(' ' + '\n' + 
          'Summary of ' + topic + ' based on ' + algorithm + 
          ' algorithm and using threshold of ' + str(threshold) + ' :' + 
          '\n' + '##########################' + '\n' + smry_deg_cent + '\n' 
          + '##########################')
          
    # Read ground truth
    path_ground_truth = dir_path + 'GroundTruth/' + topic + '.1'
    smry_ground_truth = getText(path_ground_truth)
    
    # Evaluate generated summaries and print them
    rouge = Rouge()
    scores_deg_cent = rouge.get_scores(smry_ground_truth, smry_deg_cent)[0]
    # print this only if the algorithm is 'degree_centrality'
    print('Rouge scores for the summary are :' + '\n' + 
          '##########################'  + '\n' + 
          'Rouge-1 f-score is ' + str(scores_deg_cent['rouge-1']["f"])  + '\n' + 
          'Rouge-1 p-score is ' + str(scores_deg_cent['rouge-1']["p"])  + '\n' + 
          'Rouge-1 r-score is ' + str(scores_deg_cent['rouge-1']["r"])  + '\n' + 
          'Rouge-2 f-score is ' + str(scores_deg_cent['rouge-2']["f"])  + '\n' + 
          'Rouge-2 p-score is ' + str(scores_deg_cent['rouge-2']["p"])  + '\n' + 
          'Rouge-2 r-score is ' + str(scores_deg_cent['rouge-2']["r"])  + '\n' + 
          'Rouge-l f-score is ' + str(scores_deg_cent['rouge-l']["f"])  + '\n' + 
          'Rouge-l p-score is ' + str(scores_deg_cent['rouge-l']["p"])  + '\n' + 
          'Rouge-l r-score is ' + str(scores_deg_cent['rouge-l']["r"])  + '\n' + 
          '##########################')


##############################################################################
################Text Rank based approach to generate summary##################
##############################################################################

if algorithm == 'text_rank' :
    
    # convert adjacency matrix into transition matrix
    trans_matrix = adj_matrix/adj_matrix.sum(axis = 1, keepdims = True)
    # convert the values in the rows into probabilities
    # by dividing with sum of all the elements in the row
    
    # Power method to calculate eigen vector of transition matrix and hence 
    # text-rank of all the sentences
    M = trans_matrix # stochastic matrix
    N = trans_matrix.shape[0] # matrix size (N x N)
    eigen_vector = np.ones(N)/N # initialize eigen vector
    
    # Solve for largest eigen vector of transition matrix using power method 
    # using a function and stop loop based on number of iterations to run
    def power_method(mat, start, maxit):
        eigen_vector = start
        for i in range(maxit):
            eigen_vector = mat.dot(eigen_vector)/np.linalg.norm(eigen_vector)
            return eigen_vector
    
    # call the function to calculate largest eigen vector
    eigen_vector = power_method(mat = M, start = eigen_vector, maxit = 500000)
    
    # elements of this 'eigen_vector' represent the respective text-rank of 
    # the sentences
    # return indices/sentence IDs of the 'largest_eigen_vector' based on their score
    
    text_rank = np.array([b[0] for 
                          b in sorted(enumerate(eigen_vector), reverse = True, 
                                      key = lambda i:i[1])])
    # indices of elements with decreasing order of score(value in eigen_vector)
    # sentence IDs in ranked order
    
    ranked_text = ' '.join([sentences[i] for i in text_rank])
    # join the sentences in the ranked order to form summary 
    
    # Extract 250-word summary and print
    smry_txt_rank = ' '.join(ranked_text.split()[:250]) # 250-word summary
    print(' ' + '\n' + 
          'Summary of ' + topic + ' based on ' + algorithm + 
          ' algorithm and using threshold of ' + str(threshold) + ' :' + 
          '\n' +
          '##########################' + '\n' + smry_txt_rank + '\n' 
          + '##########################')
    
    # Read ground truth
    path_ground_truth = dir_path + 'GroundTruth/' + topic + '.1'
    smry_ground_truth = getText(path_ground_truth)
    
    # Evaluate generated summaries and print them
    rouge = Rouge()
    scores_txt_rank = rouge.get_scores(smry_ground_truth, smry_txt_rank)[0]
    print('Rouge scores for the summary are :' + '\n' + 
          '##########################'  + '\n' + 
          'Rouge-1 f-score is ' + str(scores_txt_rank['rouge-1']["f"])  + '\n' + 
          'Rouge-1 p-score is ' + str(scores_txt_rank['rouge-1']["p"])  + '\n' + 
          'Rouge-1 r-score is ' + str(scores_txt_rank['rouge-1']["r"])  + '\n' + 
          'Rouge-2 f-score is ' + str(scores_txt_rank['rouge-2']["f"])  + '\n' + 
          'Rouge-2 p-score is ' + str(scores_txt_rank['rouge-2']["p"])  + '\n' + 
          'Rouge-2 r-score is ' + str(scores_txt_rank['rouge-2']["r"])  + '\n' + 
          'Rouge-l f-score is ' + str(scores_txt_rank['rouge-l']["f"])  + '\n' + 
          'Rouge-l p-score is ' + str(scores_txt_rank['rouge-l']["p"])  + '\n' + 
          'Rouge-l r-score is ' + str(scores_txt_rank['rouge-l']["r"])  + '\n' + 
          '##########################')
