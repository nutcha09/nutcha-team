#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


def normalization(data):
    """
    Normalize the given data
    Input:  data = a 2d numpy array of shape (n_samples, n_features).
    Output:  the normalized data on each column separately
    """
    nor = []
    
    for i in range(data.shape[0]):
        if np.max(data) == np.min(data):
            x = 0
        else:
            x = (data[i] - np.min(data))/(np.max(data) - np.min(data))
        nor.append(x)
        
    normalize = np.asarray(nor)      
    return normalize

def standardization(data):
    """
    Standardize the given data
    Input:  data = a 2d numpy array of shape (n_samples, n_features).
    Output:  the standardized data on each column separately
    """
    std = []
    for i in range(data.shape[0]):
        if np.std(data) == 0:
            x = 0
        else:
            x = (data[i] - np.mean(data))/np.std(data)
        std.append(x)
        
    standarde = np.asarray(std)    
    return standarde

def label_encoding(data):
    """
    Label encoding the given categorial data in the alphabetical order
    Input:  data = a 1d numpy array of str.
    Output:  the 1d array of encoded labels and a 1d array of class label
    """
    class_encode = np.unique(data[:], return_inverse=True)
    classes = class_encode[0]
    encoded = class_encode[1]
    
    return encoded, classes

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, KBinsDiscretizer

    # load iris features and target as numpy array 
    data = np.loadtxt('iris.data', delimiter=',', usecols=(0,1,2,3))
    target = np.loadtxt('iris.data', delimiter=',', usecols=(4), dtype='str')
    scaler = MinMaxScaler()
    skl = scaler.fit_transform(data[:,0:1])
    our = normalization(data[:,0:1])
    print(skl[-5:,:])
    print(our[-5:,:])
    assert np.allclose(skl, our)

    scaler = StandardScaler()
    skl = scaler.fit_transform(data[:,1:3])
    our = standardization(data[:,1:3])
    print(skl[-5:,:])
    print(our[-5:,:])
    assert np.allclose(skl, our)

    le = LabelEncoder()
    skl = le.fit_transform(target)
    our, cls = label_encoding(target)
    print(skl[-5:])
    print(our[-5:])
    print(le.classes_)
    print(cls)


# In[ ]:




