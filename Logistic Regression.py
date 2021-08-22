#!/usr/bin/env python
# coding: utf-8

# # Implementación del método de la máxima verosimilitud para la regresión logísitica

# Definir  la función de entorno

# In[ ]:


from IPython.display import display, Math, Latex
display(Math(r"L(\beta)=\sum_{i=1}^n P_i^{y_i}{1-Pi}^{y_i}"))


# In[ ]:





# In[ ]:



def likelihood(y, pi):
    import numpy as np
    total_sum=1
    sum_in= list(range(1, len(y)+1))
    for i in range(len(y)):
        sum_in[i]=np.where(y[i]==1, pi[i], 1-pi[i])
        total_sum=total_sum * sum_in[i]
                                    
    return total_sum



        
    


# In[ ]:





# #### Calcular ls probabilidades para cada observación  

# In[ ]:


display(Math(r"Pi = P(x_i) \frac{1}{1+e^{-\beta \cdot \ x_i}}"))


# In[ ]:


def logitprobs(X, beta):
    import numpy as np
    n_rows = np.shape(X)[0]
    n_cols = np.shape(X)[1]
    pi = list(range (1, n_rows + 1))
    expon = list(range(1, n_rows + 1))
    for i in range(n_rows):
        expon[i]=0
        for j in range (n_cols):
            ex = X[i][j] * beta[j]
            expon[i]=ex + expon[i]
        with np.errstate(divide = "ignore", invalid = "ignore"):
            pi[i]= 1/(1+np.exp(-expon[i]))
    return pi


# In[ ]:





# # Calcular la matri diagonal W

# In[ ]:


display(Math(r"W=diag(Pi \cdot (1-P_i))_{i=1}^n"))


# In[ ]:


def findW(pi):
    import numpy as np
    n=len(pi)
    W= np.zeros(n*n).reshape(n,n)
    for i in range (n):
        print(i)
        W[i,i]=pi[i]*pi[i]
        W[i,i].astype(float)
    return W


# In[ ]:





# # Obtener la solución de la función logísitica

# In[ ]:


display(Math(r"x_{n+1} = x_n -\frac {f(x_n)}{f'(x_n)} "))


# In[ ]:


def logistics(X, Y, limit):
    import numpy as np
    from numpy import linalg
    nrow = np.shape(X)[0]
    bias = np.ones(nrow).reshape(nrow, 1)
    X_new = np.append(X, bias, axis=1)
    ncol = np.shape(X_new)[1]
    beta = np.zeros(ncol).reshape(ncol, 1)
    root_dif = np.array(range(1,ncol+1)).reshape(ncol,1)
    inter_i = 10000
    while(inter_i > limit):
        print(str(inter_i) + "," + str(limit))
        pi = logitprobs(X_new, beta)
        print ("Pi: " + str(pi))
        W=findW(pi)
        print("W: "+ str(W))
        num=(np.transpose(np.matrix(X_new))*np.matrix(Y-np.transpose(pi)).transpose())
        den = (np.matrix(np.transpose(X_new)))*np.matrix(W)*np.matrix(X_new)
        inc = np.array(linalg.inv(den)*num)
        print("Beta:" + strin(beta))
        iter_i = np.sum(root_dif * root_dif)
        print(iter_i)
        ll=likelihood(Y, pi)
    return beta


# In[ ]:





# In[ ]:





# In[ ]:





# # Comprobacion experimental 

# In[1]:


import numpy as np


# In[ ]:





# In[2]:


X = np.array(range(10)).reshape(10,1)
X


# In[ ]:





# In[3]:


Y = [0,0,0,0,1,0,1,0,1,1]


# In[ ]:





# In[4]:


bias = np.ones(10).reshape(10,1)


# In[ ]:





# In[5]:


X_new = np.append(X,bias, axis=1)


# In[ ]:





# In[ ]:


a=logistics(X,Y,0.0001)


# # Con el paquete de python

# In[7]:


import statsmodels.api as sm


# In[8]:


logit_model = sm.Logit(Y,X_new)


# In[9]:


result = logit_model.fit()


# In[10]:


print(result.summary())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




