    # -*- coding: utf-8 -*-
'''
DocumentaciÃ³n sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''

import matplotlib.pyplot as plt
import pandas as pd

from sklearn import cluster
import seaborn as sns

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

censo = pd.read_csv('mujeres_fecundidad_INE_2018.csv')

#Se pueden reemplazar los valores desconocidos por un nÃºmero
#censo = censo.replace(np.NaN,0)

# Sustituimos valores perdidos con la media      
for col in censo:
   censo[col].fillna(censo[col].mean(), inplace=True)
      
#seleccionar casos
subset = censo.loc[(censo['EDAD']>20) & (censo['EDAD']<=40)]

#seleccionar variables de interÃ©s para clustering
usadas = ['RELIGION', 'NHOGAR', 'NTRABA', 'TEMPRELA', 'NDESEOHIJO']
X = subset[usadas]

X_normal = X.apply(norm_to_zero_one)

# Heat Map
alg = cluster.KMeans(init='k-means++', n_clusters=5, n_init=5)
prediction = alg.fit_predict(X_normal) 
centers = pd.DataFrame(alg.cluster_centers_, columns=list(X))
centers_desnormal = centers.copy()
# Convertimos los centros a los rangos originales antes de normalizar
for var in list(centers):
    centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

heatmap = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
