

```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```


```python
restaurant_data= pd.read_csv("Resources/zomato.csv",encoding="ISO-8859-1")

```


```python
x_data = restaurant_data[["Country Code","Cuisines"]]

```


```python
from kmodes import kmodes

df_dummy = pd.get_dummies(x_data)

#transform into numpy array
X = df_dummy.reset_index().values

km = kmodes.KModes(n_clusters=12, init='Huang', n_init=5, verbose=0)
clusters = km.fit_predict(x_data)
df_dummy['clusters'] = clusters


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(2)

# Turn the dummified df into two columns with PCA
plot_columns = pca.fit_transform(df_dummy.ix[:,0:12])

# Plot based on the two dimensions, and shade by cluster label
plt.scatter(x=plot_columns[:,1], y=plot_columns[:,0], c=df_dummy["clusters"], s=30)
plt.show()
```

    C:\Users\mitra\Anaconda3\envs\PythonData\lib\site-packages\ipykernel_launcher.py:18: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
    


![png](output_3_1.png)



```python
km.cluster_centroids_
```




    array([['1', 'Chinese'],
           ['1', 'Continental'],
           ['1', 'Italian'],
           ['1', 'Tibetan'],
           ['208', 'Turkish'],
           ['1', 'Lebanese'],
           ['1', 'Indian'],
           ['1', 'Asian'],
           ['216', 'American'],
           ['30', 'Brazillian'],
           ['1', 'Japanese'],
           ['1', 'American']], dtype='<U11')


