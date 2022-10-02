#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def distance(a, b):
    """
    Find Euclidian distance between two vectors
    Input:  a = 1D array vector
            b = another 1D array vector
    Output: Euclidian distance between vectors a and b
    """
    distance = 0.0
    
    for i in range(len(b)):
        distance += (a[i] - b[i])**2
        
    return sqrt(distance)
    
    
    

def knn(sample, data, target, k=1):
    """
    Make prediction using K-Nearest Neighbor technique
    Input:  sample = 1D array vector of the sample we want to predict its class
            data = 2D array of data, each row is the instance, each column is the attribute.
            target = 1D array indicating the class of each row in data
            k = the parameter
    Output: the predicted class of the sample
    """
    distances =[]
    for train_row in data:
        dist = distance(sample, train_row)
        distances.append(dist)
    distances = list(enumerate(distances))
    distances.sort(key=lambda tup: tup[1])
    output = []
    
    for i in range(k):
        output.append(target[distances[i][0]])
    prediction = max(set(output), key = output.count)    
    
    return prediction

    
    
        
  

  
if __name__=='__main__':
    data = np.loadtxt('iris.data', delimiter=',', usecols=(0,1))
    target = np.loadtxt('iris.data', delimiter=',', usecols=(4), dtype='str')

    sample = np.array([4.5, 2.])
    print(knn(sample, data, target, k=3)) 
        
    classes = np.unique(target)
    plt.figure()
    for c in classes:
        plt.scatter(data[target==c,0],data[target==c,1], label=c)
    plt.scatter(sample[0], sample[1], color='r', marker='x')
    plt.legend()
    plt.title('Prediction = %s' % knn(sample, data, target, k=3))
    plt.axis('square')
    plt.show()
    
    
    


# In[ ]:





# In[ ]:




