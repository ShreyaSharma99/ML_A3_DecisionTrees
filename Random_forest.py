#!/usr/bin/env python
# coding: utf-8

# In[1]:


from xclib.data import data_utils
import pandas as pd
from numpy import *
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[2]:


def read_labels(f_name):
    f = pd.read_csv(f_name, header = None,  encoding='ISO-8859-1') 
    f = f.to_numpy() 
    return f


# In[3]:


Y_test = read_labels('/home/shreya/Sem6/COL774/A3/virus/ass3_parta_data/test_y.txt')
Y_train = read_labels('/home/shreya/Sem6/COL774/A3/virus/ass3_parta_data/train_y.txt')

x_test = data_utils.read_sparse_file('/home/shreya/Sem6/COL774/A3/virus/ass3_parta_data/test_x.txt', force_header=True)
x_train = data_utils.read_sparse_file('/home/shreya/Sem6/COL774/A3/virus/ass3_parta_data/train_x.txt', force_header=True)


# In[4]:


# X_train = np.vstack((x_train[0].toarray(),  x_train[1].toarray()))
# for i in range(2,x_train.shape[0]):
#     l = x_train[i].toarray()
#     X_train = np.vstack((X_train,  x_train[i].toarray()))
    
X_train = x_train.toarray()


# In[5]:


# X_test = np.vstack((x_test[0].toarray(),  x_test[1].toarray()))
# for i in range(2,x_test.shape[0]):
#     l = x_test[i].toarray()
#     X_test = np.vstack((X_test,  x_test[i].toarray())) 
X_test = x_test.toarray()


# In[6]:


print(X_train.shape)


# In[7]:


# clf = RandomForestClassifier(max_depth=50, random_state=0,  min_samples_leaf=10)
# clf.fit(X_train, Y_train)
# print(clf.feature_importances_)


# In[8]:


#check accuracy 
def get_accuracy(X_test, Y_test, clf):
    accuracy = 0.0
    predict = clf.predict(X_test)
    for i in range(len(Y_test)):
        if (predict[i]==Y_test[i]):
            accuracy += 1.0
    accuracy = accuracy/(len(Y_test))
    print("Accuracy is: ", accuracy)
    return accuracy


# In[9]:


param_grid = {'n_estimators': [50, 150, 250, 350, 450], 'max_features': [0.1, 0.3, 0.5, 0.7, 0.9], 'min_samples_split':[2, 4, 6, 8, 10]}
rf = RandomForestClassifier(criterion = 'entropy', oob_score=True)
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2, refit = True)


# In[10]:


grid_search.get_params()


# In[ ]:


grid_search.fit(X_train, Y_train)
# acc_new = get_accuracy(X_test, Y_test, grid_search)

