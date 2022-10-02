#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np

def find_accuracy(y_true, y_pred):
    
    true_pred =0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            true_pred+=1
    return true_pred/len(y_true)  

def find_precision(y_true, y_pred):
    
    tpos_count = []
    pos_count = []
    precision = []
    for i in set(y_true):
        pos_count.append(list(y_pred).count(i))
        x = 0
        for j in range(len(y_true)):
            if i == y_true[j] and i == y_pred[j]:
                x += 1
        tpos_count.append(x)
    for i in range(len(tpos_count)):
        try:
            x = tpos_count[i]/ pos_count[i]
            precision.append(x)
        except:
            precision.append(0)
            
        
    return np.array(precision)

def find_recall(y_true, y_pred):
    
    tpos_count = []
    pos_count =[]
    for i in set(y_true):
        pos_count.append(list(y_true).count(i))
        x = 0
        for j in range(len(y_true)):
            if i == y_true[j] == y_pred[j]:
                x += 1
            
        tpos_count.append(x)
        
    return np.array(tpos_count)/np.array(pos_count)

def find_f1score(y_true, y_pred):
    
    precision = find_precision(y_true, y_pred)
    recall = find_recall(y_true, y_pred)
    
    return 2* ((precision * recall)/(precision + recall))


if __name__ == "__main__":
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

    # load iris features and target as numpy array 
    data = np.loadtxt('iris.data', delimiter=',', usecols=(0,1))
    target = np.loadtxt('iris.data', delimiter=',', usecols=(4), dtype='str')

    # convert target (in str) to number (int)
    le = LabelEncoder()
    y = le.fit_transform(target)
    print(y, type(y.dtype), le.classes_)

    # split data to create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    # build logistic regression model and make prediction
    clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred, y_test)

    # evaluate 
    print(f'Accuracy = {accuracy_score(y_test, y_pred):.3f}')
    print(f'Accuracy = {find_accuracy(y_test, y_pred):.3f}')
    print(f'Precision = \n{precision_score(y_test, y_pred, average=None)}')
    print(f'Precision = \n{find_precision(y_test, y_pred)}')
    print(f'Recall = \n{recall_score(y_test, y_pred, average=None)}')
    print(f'Recall = \n{find_recall(y_test, y_pred)}')
    print(f'F1-score = \n{f1_score(y_test, y_pred, average=None)}')
    print(f'F1-score = \n{find_f1score(y_test, y_pred)}')


# In[ ]:





# In[ ]:




