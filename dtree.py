#!/usr/bin/env python
# coding: utf-8

# In[1]:


from xclib.data import data_utils
import pandas as pd
from numpy import *
import numpy as np
import copy
import statistics
import matplotlib.pyplot as plt


# In[2]:


def read_labels(f_name):
    f = pd.read_csv(f_name, header = None,  encoding='ISO-8859-1') 
    f = f.to_numpy() 
    return f


# In[3]:


Y_test = read_labels('/home/shreya/Sem6/COL774/A3/virus/ass3_parta_data/test_y.txt')
Y_train = read_labels('/home/shreya/Sem6/COL774/A3/virus/ass3_parta_data/train_y.txt')


# In[4]:


x_test = data_utils.read_sparse_file('/home/shreya/Sem6/COL774/A3/virus/ass3_parta_data/test_x.txt', force_header=True)
x_train = data_utils.read_sparse_file('/home/shreya/Sem6/COL774/A3/virus/ass3_parta_data/train_x.txt', force_header=True)


# In[5]:


Y_valid = read_labels('/home/shreya/Sem6/COL774/A3/virus/ass3_parta_data/valid_y.txt')
x_valid = data_utils.read_sparse_file('/home/shreya/Sem6/COL774/A3/virus/ass3_parta_data/valid_x.txt', force_header=True)


# In[6]:


f_size = x_train.shape[1]
print(f_size)
X_train = []
for i in range(x_train.shape[0]):
    l = x_train[i].toarray()
    X_train.append(l)
X_test = []
for i in range(x_test.shape[0]):
    l = x_test[i].toarray()
    X_test.append(l)    
X_valid = []
for i in range(x_valid.shape[0]):
    l = x_valid[i].toarray()
    X_valid.append(l)   


# In[7]:


# #to get H_y
# p_y0 = 1.0
# p_y1 = 1.0
# for label in Y_train:
#     if(label==0):
#         p_y0 += 1.0

# p_y1 = len(Y_train) - p_y0
# print("p_y0 = ", p_y0)
# print("p_y1 = ", p_y1)
# print(len(Y_train))
# p_y0 = p_y0/(len(Y_train)+1)
# p_y1 = p_y1/(len(Y_train)+1)


# H_y = (-1)*p_y0*(np.log(p_y0)/np.log(2))
# H_y = H_y - p_y1*(np.log(p_y1)/np.log(2))
# print('H of y = ', H_y)


# In[7]:


def feature_median(f_index, i_list, X):
    med_list = []
    if len(i_list)==0:
        return -1
    for i in i_list:
        temp = X[i]
        med_list.append(temp[0][f_index])
#     print(max(med_list))
    return median(med_list)


# In[9]:


# print(X_train[1][0][0])
# print(X_train[2][0][0])
# print(X_train[4][0][0])
# print(X_train[5][0][0])

# print(Y_train[1])
# print(Y_train[2])
# print(Y_train[3])
# print(Y_train[4])

# feature_median(0,[1,2,4,5],X_train)


# In[8]:


def H_entropy(i_list, Y):
    p0 = 0.0
    p1 = 0.0
    for i in i_list:
        if(Y[i]==1):
            p1 += 1.0
        else:
            p0 += 1.0
    if(p0==0 or p1==0):
        H = 0
    else:
        p0 = p0/(len(i_list)+2)
        p1 = p1/(len(i_list)+2)    
        H = -(p0*(np.log(p0)/np.log(2))) - (p1*(np.log(p1)/np.log(2))) 
        
#     print("p0 ", p0)
#     print("p1 ", p1)
    
#     print(H)
    return H


# In[11]:


print(H_entropy([1,2,3,4], Y_train))
# f_med = feature_median(list(range(0,len(Y_train))), X_train)
# print(f_med)
# print(H_entropy(1, f_med, list(range(0,len(Y_train))), X_train))


# In[9]:


def mutual_info(f_index, i_list, X, Y):
    
    f_median = feature_median(f_index, i_list, X)
#     median1 = feature_median(f_index, i1_list, X)
#     if f_median==0:
#         return 1000000, [], []
    f0_list = []
    f1_list = []
    
    for i in i_list:
        temp = X[i]
        if(temp[0][f_index]<=f_median):
            f0_list.append(i)
        else:
            f1_list.append(i)
    
#     print('med_feature: ', f_median)
#     print('med1: ',median1)
    if(len(f0_list)==0 or len(f1_list)==0):
        H_y_x =  100000
    else:
        H_y_x0 = H_entropy(f0_list, Y)
        H_y_x1 = H_entropy(f1_list, Y)    
    
        f0 = float(len(f0_list))/len(i_list)
        f1 = float(len(f1_list))/len(i_list)
        H_y_x = ((f0*H_y_x0) + (f1*H_y_x1))
    
    p0 = 0.0
    p1 = 0.0
    for i in i_list:
        if(Y[i]==1):
            p1 += 1.0
        else:
            p0 += 1.0
    if(p0>p1):
        predict = 0
    else:
        predict = 1
#     print('Entropy of y given x is = ', H_y_x)
    return H_y_x, f0_list, f1_list, f_median, predict


# In[10]:


#find best feature
def best_feature(i_list, X, Y):
    best_f = 0
    best_med = 0
    best_predict = 0
    min_entropy = 1000000
    f0_best = []
    f1_best = []
    
#     i0_list = []
#     i1_list = []
    
#     #to calculate H(y)
#     for i in i_list:
#         if(Y[i]==0):
#             i0_list.append(i)
#         else:
#             i1_list.append(i)
            
#     p_y0 = float(len(i0_list)+1)/(len(Y)+2)
#     p_y1 = float(len(i1_list)+1)/(len(Y)+2)
    
#     H_y = (p_y0*(np.log(1/p_y0)/np.log(2))) + (p_y1*(np.log(1/p_y1)/np.log(2)))  
    
    for i in range(f_size):
#         if parent_list[i]==0:
        val, f0, f1, med_temp, predict = mutual_info(i, i_list, X, Y)
        if (min_entropy>val):
            min_entropy = val
            best_f = i
            f0_best = f0
            f1_best = f1
            best_med = med_temp
            best_predict = predict
#     print(H_y-)
#     median_val = feature_median(best_f, i_list, X)
    return best_f, best_med, f0_best, f1_best, best_predict


# In[21]:


# print(best_feature(list(range(0,len(Y_train))), X_train, Y_train))


# In[11]:


class Node:
    def __init__(self, feature, median, height, max_predict):
        self.left = None
        self.right = None
        self.feature = feature
        self.median = median
        self.height = height
#         self.parent = parent_list
        self.max_predict = max_predict
    def insert_left(self, node):
        if self.left is None:
            self.left = node
    
    def insert_right(self, node):
        if self.right is None:
            self.right = node  


# In[12]:


def make_tree(i_list, X, Y, height):
    if(height==0 or len(i_list)==0):
        case0 = 0
        case1 = 0
        for i in i_list:
            if Y[i]==0:
                case0 += 1
            else:
                case1 += 1
        if case0 > case1:
            return 0
        else:
            return 1
    else:
        feature, f_median, f0_list, f1_list, max_predict = best_feature(i_list, X, Y)
        f_node = Node(feature, f_median, height, max_predict)
#         parent_list[feature] = 1
        print("At height: ", height, " feature: ", feature, " f_median: ", f_median)
        left_node = make_tree(f0_list, X, Y, height-1)
        right_node = make_tree(f1_list, X, Y, height-1)
        f_node.insert_left(left_node)
        f_node.insert_right(right_node)        
        return f_node


# In[13]:


print(len(X_valid))


# In[14]:


def decision_tree(dTree, x):
    node = dTree
    while(type(node)!=int):
        if(x[node.feature] > node.median): 
            node = node.right
        else:
            node = node.left
#     if(x[node.feature] > node.median):
#         return node.right
#     else:
#         return node.left
    return node


# In[15]:


def get_accuracy(X_test, Y_test, dtree):
    accuracy = 0.0
    for i in range(len(Y_test)):
        predict = decision_tree(dtree, X_test[i][0])
        if (predict==Y_test[i]):
            accuracy += 1.0
    accuracy = accuracy/(len(Y_test))
    print("Accuracy is: ", accuracy)
    return accuracy


# In[16]:


#define decision tree:
Tree_list = []
h = 20
height = h
i_list = range(0, len(Y_train))
#     parent_list = np.zeros(f_size)
Tree = make_tree(i_list, X_train, Y_train, height)
Tree_list.append(Tree)
#     val_acc.append(get_accuracy(X_valid, Y_valid, Tree))
#     test_acc.append(get_accuracy(X_test, Y_test, Tree))    


# In[17]:


# print("Validation accuracies are: ", get_accuracy(X_valid, Y_valid, Tree_list[0]))
# print("Test accuracies are: ", get_accuracy(X_test, Y_test, Tree_list[0]))
# print("Test accuracies are: ", get_accuracy(X_train, Y_train, Tree_list[0]))

print(Tree_list[0])


# In[18]:


big_tree20 = copy.deepcopy(Tree_list[0])
print(big_tree20.height)


# In[46]:


# def max_predit(i_list):
#     case0 = 0
#     case1 = 0
#     for i in i_list:
#         if Y_train[i]==0:
#             case0 += 1
#         else:
#             case1 += 1
#     if case0 > case1:
#         return 0
#     else:
#         return 1


# In[19]:


def get_tree(node, h):
    if type(node) == int:
        return node
    elif node.height<=(50-h):
        return node.max_predict
    else:
        node.left = get_tree(node.left, h)
        node.right = get_tree(node.right, h)
        return node


# In[21]:


temp = copy.deepcopy(big_tree20)
t = get_tree(temp, 2)
print(t)


# In[ ]:


print(t.right.left)


# In[ ]:


#generate trees of different heights from big_tree50
h_tree_list = []
h = 49
temp = copy.deepcopy(big_tree50)

while h>=0:
    temp = get_tree(temp, h)
    t_new = copy.deepcopy(temp)
    h_tree_list.append(t_new)
    h -= 1


# In[22]:


def count_node(node):
    count = 0
    if(type(node)==int):
        return 0
    else:
        count += 1
        count += count_node(node.left)
        count += count_node(node.right) 
        return count


# In[23]:


count_node(Tree_list[0])
# print(count_node(h_tree_list[30]))


# In[73]:


val_acc = []
test_acc = []
node_count = []
train_acc = []
for i in range(len(h_tree_list)):
    val_acc.append(get_accuracy(X_valid, Y_valid, h_tree_list[i]))
    test_acc.append(get_accuracy(X_test, Y_test, h_tree_list[i])) 
    train_acc.append(get_accuracy(X_train, Y_train, h_tree_list[i]))
    node_count.append(count_node(h_tree_list[i]))


# In[79]:


plt.scatter(node_count, val_acc)
plt.xlabel('Number of Nodes')
plt.ylabel('Validation Accuracy')
plt.title('Plot2')
plt.show(block=False)
# time.sleep(5)
plt.close()


# In[80]:


plt.scatter(node_count, test_acc)
plt.xlabel('Number of Nodes')
plt.ylabel('Test Accuracy')
plt.title('Plot3')
plt.show(block=False)
# time.sleep(5)
plt.close()


# In[81]:


plt.scatter(node_count, train_acc)
plt.xlabel('Number of Nodes')
plt.ylabel('Train Accuracy')
plt.title('Plot1')
plt.show(block=False)
# time.sleep(5)
plt.close()


# In[24]:


def prune_tree(dtree, Y):
    prune_list = []
    left_list = []
    right_list = []
    
#     print("Height: ", dtree.height)
    if dtree.height>1:
        tree = copy.deepcopy(dtree)
        tree.height = 1
#         case0 = 0
#         case1 = 0
#         print("Left: ", dtree.left)
        if (type(tree.left) != int):
            tree.left = tree.left.max_predict
#             for i in tree.left.i_list:
#                 if Y[i]==0:
#                     case0 += 1
#                 else:
#                     case1 += 1
#             if case0 > case1:
#                 tree.left = 0
#             else:
#                 tree.left = 1
            
#         case0 = 0
#         case1 = 0
#         print("Right: ", dtree.right)
        if (type(tree.right) != int):
            tree.right = tree.right.max_predict
#             for i in tree.right.i_list:
#                 if Y[i]==0:
#                     case0 += 1
#                 else:
#                     case1 += 1
#             if case0 > case1:
#                 tree.right = 0
#             else:
#                 tree.right = 1
       
        prune_list.append(tree)
        if (type(dtree.left) != int):
            left_list = prune_tree(dtree.left, Y)
        if (type(dtree.right) != int):
            right_list = prune_tree(dtree.right, Y)
        
        for node in left_list:
            temp = copy.deepcopy(dtree)
            temp.left = node
            if (type(temp.right) == int):
                temp.height = max(temp.left.height, 0)+1
            else:
                temp.height = max(temp.left.height, temp.right.height)+1
                prune_list.append(temp)
        
        for node in right_list:
            temp = copy.deepcopy(dtree)
            temp.right = node
            if (type(temp.left) == int):
                temp.height = max(temp.right.height, 0)+1
            else:
                temp.height = max(temp.left.height, temp.right.height)+1
            prune_list.append(temp)
        
    return prune_list


# In[ ]:


#pruning using validaton set----->
prune_val_acc = []
prune_test_acc = []
prune_train_acc = []
prune_num = []
best_accuracy = get_accuracy(X_valid, Y_valid, Tree_list[0])
best_tree = copy.deepcopy(Tree_list[0])
old_accuracy = 0
count = 0
while(old_accuracy<best_accuracy):
    old_accuracy = best_accuracy
    prune_train_acc.append(get_accuracy(X_train, Y_train, best_tree))
    prune_test_acc.append(get_accuracy(X_test, Y_test, best_tree))
    prune_val_acc.append(get_accuracy(X_valid, Y_valid, best_tree))
    prune_num.append(count_node(best_tree))
    
    prune_list = prune_tree(best_tree, Y_train)
    print("Prune size: ",len(prune_list))
    for tree in prune_list:
        acc = get_accuracy(X_valid, Y_valid, tree)
        if acc>best_accuracy:
            best_accuracy = acc
            best_tree = copy.deepcopy(tree)
    count += 1
    print("Accuracy after ", count, " prunes is ", best_accuracy)
    
print("Best accuracy: ", best_accuracy)


# In[25]:


print(best_tree.height)


# In[ ]:


def prune_tree(dtree, Y):
    prune_list = []
    left_list = []
    right_list = []

    if dtree.height>1:
        tree = copy.deepcopy(dtree)
        tree.height = 1
        if (type(tree.left) != int):
            tree.left = tree.left.max_predict
        if (type(tree.right) != int):
            tree.right = tree.right.max_predict
        prune_list.append(tree)
        
        if (type(dtree.left) != int):
            left_list = prune_tree(dtree.left, Y)
        if (type(dtree.right) != int):
            right_list = prune_tree(dtree.right, Y)
        
        for node in left_list:
            temp = copy.deepcopy(dtree)
            temp.left = node
            if (type(temp.right) == int):
                temp.height = max(temp.left.height, 0)+1
            else:
                temp.height = max(temp.left.height, temp.right.height)+1
                prune_list.append(temp)
        
        for node in right_list:
            temp = copy.deepcopy(dtree)
            temp.right = node
            if (type(temp.left) == int):
                temp.height = max(temp.right.height, 0)+1
            else:
                temp.height = max(temp.left.height, temp.right.height)+1
            prune_list.append(temp)
        
    return prune_list

