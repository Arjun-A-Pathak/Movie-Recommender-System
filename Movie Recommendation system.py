#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


#Reading csv data 
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[5]:


#Merging movies and credit dataset in to single dataset
movies = pd.merge(movies,credits, on='title')
movies.head(1)


# In[6]:


movies.columns


# In[7]:


#Removing unnessery columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[8]:


#checking for null values
movies.isnull().sum()


# In[9]:


#dropping null values
movies.dropna(inplace=True)


# In[10]:


#checking for duplicates values
movies.duplicated().sum()


# In[11]:


movies.iloc[0].genres


# In[12]:


import ast

def convert(obj):  #Function to extract the value of name keywoprds only
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[13]:


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[14]:


def convert3(obj):   #Function to extract the value of name keywoprds only for first 3 cast from cast columns
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:       
            l.append(i['name'])
            counter +=1
        else:
            break
    return l


# In[15]:


movies['cast'] = movies['cast'].apply(convert3)
movies.head()


# In[16]:


def fetch_director(obj):  #Function to extract the value of name keywoprds only if job id Director from crew columns
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':       
            l.append(i['name'])
            break
    return l


# In[17]:


movies['crew'] = movies['crew'].apply(fetch_director)
movies.head(2)


# In[18]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.head(2)


# In[19]:


#Removing space from cast, crew, genres, keywords to make a single keyword for our analysis

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies.head()


# In[20]:


"""combining/adding cast, crew, genres, keywords and overview columns in to single column and 
also drop these columns from dataset"""

movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

new_df = movies[['movie_id','title','tags']]
new_df.head()


# In[21]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df.head()


# In[22]:


#converting value of tags column in lowercase
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
new_df.head()


# In[23]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[24]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)
        


# In[25]:


new_df['tags'] = new_df['tags'].apply(stem)
new_df


# ### text vectorization

# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[28]:


vectors = cv.fit_transform(new_df['tags']).toarray()
vectors


# In[29]:


cv.get_feature_names_out()


# In[30]:


new_df.head(2)


# In[31]:


from sklearn.metrics.pairwise import cosine_similarity


# In[32]:


similarity = cosine_similarity(vectors)


# In[33]:


similarity[1]


# In[34]:


def recommend(movie):  #this function that reccomend movie based on there cosine similarity
    movie_index = new_df[new_df['title']==movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)),reverse=True, key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
       
    


# In[37]:


recommend('Avatar')


# In[38]:


recommend('Batman Begins')


# In[ ]:




