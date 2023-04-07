#!/usr/bin/env python
# coding: utf-8

# ## Train

# In[1]:


import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from torch.utils.data import Dataset, DataLoader


with open('C:/Users/Dell/Downloads/pytorch-chatbot-master/pytorch-chatbot-master/intents.json', 'r') as f:
    intents = json.load(f)


# In[2]:


import torch
import torch.nn as nn


# In[3]:


print(intents)


# In[4]:


stemmer = PorterStemmer()
tags = []
xy = []
all_words = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = word_tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stemmer.stem(w) for w in all_words if w not in ignore_words]


# In[5]:


all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)


# In[6]:


from collections import Counter

def bag_of_words(sentence, words):
    """
    Returns a bag-of-words vector for a sentence
    """
    sentence_words = sentence.lower().split()
    # Count the frequency of each word in the sentence
    bag = Counter(sentence_words)

    # Create a vector of zeros with the same length as the vocabulary
    vector = [0] * len(words)

    # Update the vector to represent the sentence
    for i, word in enumerate(words):
        if word in bag:
            vector[i] = bag[word]

    return vector


# In[7]:


X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(" ".join(pattern_sentence), all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss


# In[8]:


X_train = np.array(X_train)
y_train = np.array(y_train)


# In[9]:


class ChatDataset(Dataset):
    def __init__(self):
        super(ChatDataset, self).__init__()
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

#Hyperparameters
batch_size = 8
hidden_size = 8
output_size_size = len(tags)
input_size = len(X_train[0])
print(input_size, len(all_words))
print(output_size_size, tags)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers=2)


# In[11]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## Model

# In[10]:


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()  # add this line
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(input_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)  
        out = self.l3(out)
    
        #no activitation and no softmax
        return out
    
model = NeuralNet(input_size, hidden_size, output_size_size).to(device)




# In[12]:
#loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[14]:


import torch

# Define the number of epochs
num_epochs = 1000

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'epoch {epoch+1}/{num_epochs}, Loss={loss.item()}')


print (f'epoch {epoch+1}/{num_epochs}, Loss={loss.item()}')

# In[ ]:
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')





# %%
