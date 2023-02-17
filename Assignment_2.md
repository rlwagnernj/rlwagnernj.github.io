## Cross Validated Robust LOWESS Model


```python
# Import Libraries

import numpy as np
import pandas as pd
from scipy import linalg
from math import ceil
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts, KFold
```


```python
# function that computes the Euclidean distance between all the observation in u and v
def dist(u,v):
  if len(v.shape)==1: # force v into column vector
    v = v.reshape(1,-1)
  d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))]) # distance between all points in u and v
  return d
```


```python
def lw_ag_md_kfold(x, y, f=2/3, ter=3, intercept=True, n_splits=5, shuffle=True, random_state=411):
    ''' Computes a cross validated multivariate LOWESS model for x and y
        RETURNS list of mean squared error for each fold
    '''

    kf = KFold(n_splits = n_splits, shuffle=shuffle, random_state=random_state) # set KFOLD
    mse_list = [] # initialize MSE list

    for idxtrain, idxtest in kf.split(x): # run for each of the splits
        xtrain = x[idxtrain]
        ytrain = y[idxtrain]
        ytest = y[idxtest]
        xtest = x[idxtest]

        n = len(xtrain) # number of observations/data points
        r = int(ceil(f * n)) # number of points that define local neighborhood
        yest = np.zeros(n)

        if len(ytrain.shape)==1: # here we make column vectors
            ytrain = ytrain.reshape(-1,1)

        if len(xtrain.shape)==1:
            xtrain = xtrain.reshape(-1,1)

        if intercept:
            x1 = np.column_stack([np.ones((len(xtrain),1)),xtrain]) # add a column of 1s to the matrix if intercept is desired
            x2 = np.column_stack([np.ones((len(xtest),1)),xtest])
        else:
            x1 = xtrain
            x2 = xtest

        h = [np.sort(np.sqrt(np.sum((xtrain-xtrain[i])**2,axis=1)))[r] for i in range(n)] # get distances from one point to all others 
                                                                                          # sort them, get the rth entry
                                                                                          # this distance defines the max distance for said point

        w = np.clip(dist(xtrain,xtrain) / h, 0.0, 1.0) # divide distances between all points by h (max distance) for the associated point
                                                       # any value above 1 is above max distance, clipped to one
                                                       # any value below 1 is within max distance, value stays
                                                       # any 0 (only point to itself) stays 0

        w = (1 - w ** 3) ** 3 # apply kernel to weight the points. This kernel can be changed if desired

        delta = np.ones(n)

        for iteration in range(iter):
            for i in range(n):
                W = np.diag(w[:,i])
                b = np.transpose(x1).dot(W).dot(ytrain) # Create matricies
                A = np.transpose(x1).dot(W).dot(x1) # X^T(W)(y) = X^T(X)(W)(Beta)
                ##
                A = A + 0.0001*np.eye(x1.shape[1]) # if we want L2 regularization
                beta = linalg.solve(A, b) # solve for Beta
                #beta, res, rnk, s = linalg.lstsq(A, b)
                yest[i] = np.dot(x1[i],beta) # Apply function to get y estimates

        residuals = ytrain - yest # calculate residuals 
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1) # get weights for nearest residuals/farthest. Defines "robustness"
        delta = (1 - delta ** 2) ** 2

    # Now that we have the final model, we can apply it to our test data and check the MSE
    ytestest = np.dot(x2, beta) 
    mse_list.append(mse(ytest,ytestest))

    print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_list)))
    return mse_list

```

## Test on real data...


```python
data = pd.read_csv('/Users/rebeccawagner/Documents/GitHub/Data 441/Data/cars.csv')

x = data.loc[:,'CYL':'WGT'].values
y = data['MPG'].values
```


```python
lw_ag_md_kfold(x, y, f=2/3, ter=3, intercept=True, n_splits=5, shuffle=True, random_state=411)
```

    The Cross-validated Mean Squared Error for Locally Weighted Regression is : 24.20828163247101





    [24.20828163247101]




```python

```
