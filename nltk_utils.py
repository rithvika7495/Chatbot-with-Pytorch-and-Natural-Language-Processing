#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nltk
from nltk.stem import PorterStemmer


# In[18]:


print(torch.__version__)


# In[19]:


import nltk
nltk.download('punkt')


# In[20]:


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


# In[21]:


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# In[22]:


def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog = [0, 1 , 0 , 1 , 0 , 0 , 0]
    """
    
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


# In[23]:


a = "How long does shipping takes?"
print(a)


# In[24]:


a = tokenize(a)
print(a)


# In[25]:



words = ["Organize", "organizes" , "organizing"]
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in words]

print(stemmed_words)

