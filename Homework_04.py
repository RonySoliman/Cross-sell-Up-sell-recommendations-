#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info"><b></b>
# <h1><center> <font color='black'> Homework 04  </font></center></h1>
# <h2><center> <font color='black'> Cross-Selling/ Up-selling & Recommendation System</font></center></h2>   
# <h2><center> <font color='black'> MTAT.03.319 - Business Data Analytics</font></center></h2>
# <h2><center> <font color='black'> University of Tartu - Spring 2021</font></center></h2>
# </div>

# # Homework instructions
# 
# - Please provide the names and student IDs of the team-members (Maximum 2 person) in the field "Team mates" below. If you are not working in a team please insert only your name and student ID. 
# 
# - The accepted submission formats are Colab links or .ipynb files. If you are submitting Colab links please make sure that the privacy settings for the file is public so we can access your code. 
# 
# - The submission will automatically close on <font color='red'>**18 April at 23:59**</font>, so please make sure to submit before the deadline. 
# 
# - ONLY one of the teammates should submit the homework. We will grade the homework and the marks and feedback is applied for both the team members. So please communicate with your team member about marks and feedback if you are submit the homework.
# 
# - If a question is not clear, please ask us in Moodle ONLY. 
# 
# - After you have finished solving the Homework, please restart the Kernel and run all the cells to check if there is any persisting issues. 
# 
# - Plagiarism is <font color='red'>**PROHIBITED**</font>. Any form of plagiarism will be dealt according to the university policy (https://www.ut.ee/en/current-students/academic-fraud).
# 
# - Please <font color='red'>do not change</font> the template of this notebook file. You can download the .ipynb file and work on that.
# 

# **<h2><font color='red'>Team mates:</font></h2>**
# 
# 
# **<font color='red'>Name: Mohga Emam</font>&emsp;   <font color='red'>Student ID: C09505</font>**
# 
# 
# **<font color='red'>Name: Rewan Emam</font>&emsp;   <font color='red'>Student ID: C07851</font>**

# ### The homework is divided into four sections and the points are distributed as below:
# <pre>
# - Market Basket Analysis            -> 2.0 points
# - Collaborative Filtering           -> 3.5 points
# - Recommender Systems Evaluation    -> 1.0 points
# - Neural Network                    -> 2.5 points
# _________________________________________________
# Total                               -> 9.0 points
# </pre>

# # 1.  Market Basket Analysis (2 points)

# **1.1 Consider the following businesses and think about one case of cross selling and one case of up selling techniques they could use. This question is not restricted to only traditional, standard examples.(1 points)**

# ### <font color='red'> **I apologize for the inconvience but no matter what I do the text icon shows part of what I am writing so kindly click on the points [a, b, c, d] as you are editing them to see my full answer**</font> 

# a. An OnlineTravel Agency like Booking.com or AirBnB

# <font color='red'> **Cross selling: I booked a room in a certain hotel and it offered collection of effors for Taxi booking from the airport with good prices.**</font> 
# 
# <font color='red'> **Up selling: I booked a room in a certain hotel and it shows that it's not refundable but if I instead pick another room with more features the food coupon will increase and there's no payment needed, I can pay while checking in. and free cancelation before 2 days of the reservation. The difference between the two of them are less than $70.**</font> 

# b. A software company which produces products related to cyber security like Norton, Kaspersky, Avast and similar ones. 

# <font color='red'> **Cross selling: I wanted to purchase the basic package [Norton Anti-Virus] with $34.99, it shows me 2 other great packages [ Norton computer tune up] which helps my computer run like new again for $49.99 and the other one is [Norton family], which guarantee safe, secure connection for kids for $49.99.**</font> 
# 
# <font color='red'> **Up selling:[text is hidden kindly open the text] I wanted to purchase Norton package for $37.99 with %45 But the site recommended to instead purchase Norton 360 Premium Plus with 95% discount with 6 more features and only for $59.99**</font> 

# c. A company that sells cell phones
# 
# 
# 

# <font color='red'> **Cross selling: I added to the cart Iphone 11, and then down below the wesite shows adapters & headsets for Ipone 11 with vival colors**</font> 
# 
# <font color='red'> **Up selling: I clicked on the headsets icon to pick one with my Iphone 11, and I have selected one with the price of EarPods with 3.5 mm Headphone Plug for $19. The |site| showed| me |that |the |headset [Beats flex all day wireless] for |only $27.99**</font>

# d. A supermarket like Konsum, Rimi, Maxima etc. 

# <font color='red'> **Cross selling: I added to the cart chicken and it shows spicies of chicken for a great teste, 20% discount on the Rice [1 Kg]**</font>
# 
# <font color='red'> **Up selling: I added to the cart Tissue paper [8 pieces] to buy it for price 2.53 Euros, instead I found down below that if I took from different company Tissue paper [16 pieces] it would be with the price of 4.20 Euros.**</font> 

# **1.2 Let's suppose that our client is a retail company that has an online shop. They gave us a dataset about online sales of their products. The client wants to know which product bundles to promote. Find 5 association rules with the highest lift.**

# In[ ]:


import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/RewanEmam/Customer-Segmentation-files/main/OnlineRetailPurchase.csv', header=0, sep = ',')


# In[ ]:


df.head()


# **1.3 Use describe function from pandas to get statistical information about the values in the dataframe.(0.2 points)**

# In[ ]:


df.describe()


# **1.4 Create a dataframe name as "Basket", where each row has an distintive value of InvoiceNo and each column has a distinctive Description. The cells in the table contain the count of each item (Description) mentioned in one invoice. For example basket.loc['536365','WHITE HANGING HEART T-LIGHT HOLDER'] has a value of 1 because the product with WHITE HANGING HEART T-LIGHT HOLDER was entered  only once in the invoice 536365. (0.2 points)**

# In[328]:


Basket = df[['InvoiceNo', 'Description']]
basket = Basket.drop_duplicates(subset = ['InvoiceNo', 'Description'],keep= 'last').reset_index(drop = True)


# In[329]:


basket = pd.get_dummies(basket['Description'])
basket


# **1.5 Some products are mentioned more than once in one invoice. You can check the maximum number for each column to verify. Modify your dataframe such that every cell which has a value higher than one will be replaced with 1. If the cell has the value 0 it will remain the same. (0.2 points)** <br>
# NB: If your implementation in 1.4 already takes care of this, please skip the question. 

# In[ ]:


#TODO


# **1.5 We do not need to spend time on calculating the association rules by ourselves as there already exists a package for python to do so, called mlxtend. We are going to use the mlxtend package to find frequent items bought together and then create some rules on what to recomend to a user based on what he/she/they have bought. We have given you the first part of the code which calculates the frequent items bought together. (0.2 points)**

# In[ ]:


#!pip install mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml
import math


# In[ ]:


#TODO


# **Please read the documentation of the associaton rules function in mlextend [here](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/) and complete the code so we get the 5 rules with the highest lift. Print those rules. For example if user bought product basket A then  the algorithm recommends product basket B. (0.2 points)**

# In[ ]:




rules = ... #TODO



for index, row in (rules.iloc[:5]).iterrows():
    print("If the customer buys " + str(row['antecedents']))
    print("")
    print("The recommender recommends "+str(row['consequents']))
    print("")
    print("")
    print("")


# # 2. Collaborative filtering (3.5 points )

# We are going to use Books.csv dataset which contains  ratings from Amazon website and the data has the following features:
# 
# UserID: The ID of the users who read the books
# 
# BookTitle: The title of the book
# 
# Book-Rating: A rating given to the book in a scale from 0 to 10
# 
# Below we are going to perform the same steps we did with movies dataset in the practice session

# **2.0 Load the dataset and take a look at the books titles. And pick a favorite book (any book).(0.1 points)**

# In[277]:


df_book = pd.read_csv('https://raw.githubusercontent.com/RewanEmam/Customer-Segmentation-files/main/Books.csv', header=0, sep = ',', usecols=['UserID', 'Book-Rating', 'BookTitle'])
df_book.head()


# In[278]:


dfBook = df_book.drop_duplicates(subset = ['BookTitle', 'UserID'],keep= 'last').reset_index(drop = True)
dfBook


# **2.1 You have to apply KNN algorithm for collaborative filtering. As KNN algorithm does not accept strings, use a Label Encoder for BookTitle column.After that reshape the books matrix so that every column will be a UserID and every row a BookTitle. (0.45 points)**

# In[264]:


from sklearn import preprocessing

# label encounter
label = preprocessing.LabelEncoder()
dfBook['BookName'] = labelencoder.fit_transform(dfBook['BookTitle'])

# every column is userid
df_boo = dfBook.pivot(index = 'BookTitle', columns='UserID', values='Book-Rating').fillna(0)
df_boo.index.names = ['BookTitle']
df_boo.head()


# **2.2 Build a sparse matrix for books data and show it. (0.45 points)**

# In[265]:


from scipy.sparse import csr_matrix

df_boo_sparse = csr_matrix(df_boo.values)
print(f"Sparse matrix:\n{df_boo_sparse}")


# In[266]:


# create mapper from book title to index
# book: index
book_to_idx = {
    book: i for i, book in enumerate(list(dfBook.set_index('BookTitle').loc[df_boo.index].index))
}
book_to_idx


# **2.3 Initialize and train two different KNN models (use cosine metric for similarity for both) but with different n_neighbours, 2 and 10. Recommend top 5 books based on your favourite one in both cases (1 points)**<br>
# NB: You are free to choose a favorite book (any book) based on which you have to recommend 5 books.

# In[267]:


from sklearn.neighbors import NearestNeighbors

# define model: using cosine for similarity 
model_knn_null = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=2, n_jobs=-1)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)


# fit the model
print(model_knn.fit(df_boo_sparse))
print(model_knn_null.fit(df_boo_sparse))


# In[227]:


get_ipython().system('pip install fuzzywuzzy')


# In[268]:


# Import the required libraries:

import os
import time
import math
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors


# In[270]:


def fuzzy_matching(mapper, fav_book, verbose=True):
    # Get match
    match_tuple = []
    for title, idx in mapper.items():
        ratio = fuzz.ratio(BookTitle.lower(), fav_book.lower())
        if ratio >= 500:
           match_tuple.append((df_boo['BookTitle'], idx, ratio))
  
    # Sort
    match_tuple = sorted(match_tuple, key = lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]

def make_recommendation(model_knn, data, mapper, fav_book, n_recommendations):
    # data = df_boo
    model_knn.fit(data)

    # get input book index
    print('You have input book:', fav_book)
    idx = fuzzy_matching(mapper, fav_book, verbose=True)

    # Inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)

    # Get list of raw idx of recommendations
    raw_recommends =         sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}

    # print recommendation:
    print('Recommendations for {}:'.format(fav_book))
    for i, (idx, dist) in reversed(list(enumerate(raw_recommends))):
        #j =i
        print('{0}: {1}, with distance of {2}'.format(n_recommendations-i, reverse_mapper[idx], dist))


# In[ ]:


my_favorite = 'Matilda' # Matilda

make_recommendation(
    model_knn=model_knn, # trained model (model)
    data=df_boo_sparse, # sparse matrix (data)
    fav_book=my_favorite, # fav_book
    mapper=book_to_idx, # {book: index} (mapper)
    n_recommendations=5) 


# In[283]:


data = df_boo_sparse

def fuzzy_matching(mapper, fav_book, verbose=True):
    match_tuple = []
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_book.lower())
        if ratio >= 60:
           match_tuple.append((title, idx, ratio))
  
    match_tuple = sorted(match_tuple, key = lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]

def make_recommendation(model_knn_null, data, mapper, fav_book, n_recommendations):
    # data = df_boo
    model_knn_null.fit(data)

    # get input book index
    print('You have input book:', fav_book)
    idx = fuzzy_matching(mapper, fav_book, verbose=True)

    # Inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn_null.kneighbors(data[idx], n_neighbors=n_recommendations+1)

    # Get list of raw idx of recommendations
    raw_recommends =         sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}

    # print recommendation:
    print('Recommendations for {}:'.format(fav_book))
    for i, (idx, dist) in reversed(list(enumerate(raw_recommends))):
        #j =i
        print('{0}: {1}, with distance of {2}'.format(n_recommendations-i, reverse_mapper[idx], dist))


# In[ ]:


my_favorite = 'Shadowland' # The Da Vinci Code

make_recommendation(
    model_knn_null=model_knn_null, # trained model (model)
    data= df_boo_sparse, # sparse matrix (data)
    fav_book=my_favorite, # fav_book
    mapper=book_to_idx, # {book: index} (mapper)
    n_recommendations=5) 


# **2.4 Discuss the results you received from both models. Which one worked better? (0.25 points)**

# <font color='red'> **Answer: Based on the result, I found the recommendation are quite similar to the choice I have selected. Whether I have selected Matilda-The davnci code- Shadowland, etc. Thanks to the main factors I have here: Model_knn function & mapper. They are factors of the main factors that the recommendations mechanism are absed on.**</font> 

# **2.5 Add a new user (with user “UserID” = 6293) in your data. Using the two trained models in task 2.3 suggest which books should this user read if his ratings are:**
# 
# French Cuisine for All: 4
# 
# 
# Harry Potter and the Sorcerer's Stone Movie Poster Book: 5
# 
# 
# El Perfume: Historia De UN Asesino/Perfume : The Story of a Murderer: 1
# 
# **(1. 25 points)**
# 
# 

# In[ ]:


# Edit my dataset a little bit:

features = ['UserID', 'BookTitle', 'Book-Rating']

# Get each row as a string
def combine_features(row):
    return row['Book-Rating']+" "+row['UserID']+" "+row['BookTitle']

for feature in features:
    dfBook[feature] = dfBook[feature].fillna('')

dfBook["combined_features"] = dfBook.apply(combine_features, axis=1)


# In[ ]:


# In case model_knn case:

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return dfBook[dfBook['BookTitle'] == title]["index"].values[0]

book_user_likes = "Shadowland"

book_index = get_index_from_title(book_user_likes)
similar_books =  list(enumerate(cosine_sim[book_index]))

sorted_similar_books = sorted(similar_books,key=lambda x:x[1],reverse=True)[1:]
i=0
print("Top 5 similar movies to "+book_user_likes+" are:\n")
for element in sorted_similar_books:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>=5:
        break


# # 3. Recommender systems evaluation (1 points)

# We are going to compare different methods of recommender systems by their RMSE score. One useful package that has several recommender algorithms for Python is [Surprise](https://surprise.readthedocs.io/en/stable/getting_started.html). Below we have split the books dataset into training and test and used the KNNBasic algorithm to predict the ratings for the test set using surprise. 

# In[298]:


pip install surprise


# In[313]:


from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Reader
from surprise import Dataset
from surprise import SVD
from surprise import NormalPredictor
from surprise import KNNBasic

# The reader is necessary for surprise to interpret the ratings
reader = Reader(rating_scale=(0, 10))

# This function loads data from a pandas dataframe into surprise dataset structure
# The columns should always be ordered like this
data = Dataset.load_from_df(dfBook[['UserID', 'BookTitle', 'Book-Rating']], reader)

# Split in trainset and testset
# No need to define the label y because for surprise the last column is always the rating
trainset, testset = train_test_split(data, test_size=.25, random_state=0 )

knn = KNNBasic()
knn.fit(trainset)
predictions = knn.test(testset)
print('KNN RMSE', accuracy.rmse(predictions))


# **3.1 After taking a look at surprise documentation and the code above, follow the same steps as with KNN, and predict the ratings in test set using the NormalPredictor which predicts a random rating based on the distribution of the training set. Do the same for SVD which  is a matrix factorization technique. For both of them report RMSE. (1 points)**

# In[319]:


#TODO: Normal predictor
# First Recall the libraries:

from surprise.model_selection import cross_validate
from surprise.model_selection import KFold

# We can now use this dataset as we please, e.g. calling cross_validate
cross_validate(NormalPredictor(), data, cv=2)


# In[318]:


#TODO: SVD

# define a cross-validation iterator
kf = KFold(n_splits=3)

algo = SVD()

for trainset, testset in kf.split(data):

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)


# ### **Conclusion: RMSE for SVD is in range 4.2389 to 4.3355. Unlike the NormalPredictor that generates an array..**

# # 4. Neural Networks (2.5 Points)

# **4.1 We are now going to build a recommender system using Neural Networks. Being this dataset is really small in terms of features you might not see great improvements but it is a good starting point to learn. Please build  one of the neural network architechtures as we did in practice session part 3. You can for example choose the one which had the following layers:**
# - 2 Embedding
# - 2 Reshape
# - 1 Concatenation 
# - 1 Dense
# 
# **Use the Neural Network you built to learn from the train data of part 3 of this homework.  The column UserID should be used as input to your NN for the user embedding layer. For the books embedding layer we will use BookTitle column. Lastly, the ratings will be your target variable. Regarding the evaluation metric for the training phase use RMSE. To make your training fast you can use a batch size of 200 or above. (1.5 points)**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from keras import backend

from keras.layers import Input, Embedding, Flatten, Dot, Dense,multiply, concatenate, Dropout, Reshape
from keras.models import Model, Sequential

from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
#Method for RMSE calculation
def rmse(true_label, pred_label):
    return #TODO: RMSE function

#TODO: Data preparation

df = pd.read_csv('https://raw.githubusercontent.com/RewanEmam/Customer-Segmentation-files/main/Books.csv',
                 header=0, sep = ',', usecols=['UserID', 'Book-Rating', 'BookTitle'])

#TODO: Model
def RecommenderV1(user_id, title, ratings):
     user_id = Input(shape=(1,))
     u = Embedding(user_id, ratings, embeddings_initializer='he_normal',
                  embeddings_regularizer=l2(1e-6))(user_id)
     u = Reshape((n_factors,))(u)

     #TODO: Embedding user id
     title = Input(shape=(50,))
      m = Embedding(title,ratings, embeddings_initializer='he_normal',
                  embeddings_regularizer=l2(1e-6))(title)
      m = Reshape((n_factors,))(m)


      x = Dot(axes=1)([u, m])
  
      model = Model(inputs = (id_em, title_em), outputs = out)
      model.compile(optimizer = 'Adam', loss = rmse, metrics = ['accuracy'])

#TODO: Train model
history = model.fit(x=X_train_array, y=y_train, batch_size=200, epochs=150,
                    verbose=1, validation_data=(X_test_array, y_test))
    
    
#TODO: pass data, batch_size=200, epochs=150)


# **4.2 Plot the RMSE values during the training phase, as well as the model loss. Report the best RMSE. Is it better than the RMSE from the models we built in Section 2 and 3 ? (0.5 points)**

# In[ ]:


from matplotlib import pyplot
#TODO


# **4.3 Use your trained model to recommend books for user with ID 6293. (0.5 points)**

# In[ ]:


#TODO


# ## How long did it take you to solve the homework?
# 
# * Please answer as precisely as you can. It does not affect your points or grade in any way. It is okay, if it took 0.5 hours or 24 hours. The collected information will be used to improve future homeworks.
# 
# <font color='red'> **Answer: X hours**</font>
# 
# 
# ## What is the level of difficulty for this homework?
# you can put only number between $0:10$ ($0:$ easy, $10:$ difficult)
# 
# <font color='red'> **Answer:**</font>

# In[ ]:




