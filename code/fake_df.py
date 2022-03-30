#!/usr/bin/env python
#######################

import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
import matplotlib.pyplot as plt

#synthetic data
df = pd.DataFrame(np.random.randint(-1000,1000,size=(5000, 50000))) 

#scale and pca
X = df.values 
std_scale = preprocessing.StandardScaler().fit(X)      
X_scaled = std_scale.transform(X)         
pca = decomposition.PCA(n_components= 0.9)
coordinates = pca.fit_transform(X_scaled)      
 
#cumsum of explained variance ratio
df = pd.DataFrame(coordinates)
var=np.cumsum(pca.explained_variance_ratio_)*100
df2 = pd.DataFrame(var, columns = ['cumulative_explained_variance (%)'])
df2['comp_index'] = range(1,len(df2.index)+1)

#plot
df2.plot(x ='comp_index', y='cumulative_explained_variance (%)', kind = 'scatter')	
plt.show()