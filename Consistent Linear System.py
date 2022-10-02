#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
n = int(input())
li = []
var_list = []
op_list = []

for i in range(n) :
    inp = input()
    inp = list(inp.split())
    if len(inp) == n+1:
        li.append(list(map(int,inp)))
    else:
        break
    var_list.append(li[i][0:n])
    op_list.append(li[i][-1])
    
a = np.array(var_list)
b = np.array(op_list)
x = np.linalg.solve(a,b)

for j in range(len(x)) :
    print("{:.2f}".format(x[j]))
    


# In[ ]:




