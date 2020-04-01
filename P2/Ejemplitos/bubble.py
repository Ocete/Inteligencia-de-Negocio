    # -*- coding: utf-8 -*-
'''
DocumentaciÃ³n sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn import cluster
from math import floor
import seaborn as sns

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

censo = pd.read_csv('mujeres_fecundidad_INE_2018.csv')

'''
for col in censo:
   missing_count = sum(pd.isnull(censo[col]))
   if missing_count > 0:
      print(col,missing_count)
#'''

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

cluster_predict = {}
calinski = []
silh = []
buble_sizes = []
colors = []
param = []

all_colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', '#ffb347']
buble_sizes_template =[i*50 for i in range(1,10)]

# DBScan
rad_values = [r/10 for r in range(1, 5)]
for rad in rad_values:
    alg = cluster.DBSCAN(eps=rad, min_samples=20)
    cluster_predict = alg.fit_predict(X_normal)
    silh.append( float("{0:.2f}".format(
            metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(len(X)), random_state=123456))))
    calinski.append( float("{0:.2f}".format(
            metrics.calinski_harabasz_score(X_normal, cluster_predict))))

buble_sizes = buble_sizes + buble_sizes_template[::2]
colors = ['lightskyblue' for i in range(1,5)]
param = rad_values

# Resto de algoritmos
def get_alg(i_alg, k):
    algoritmos = (('KMeans', cluster.KMeans(init='k-means++', n_clusters=k, n_init=5)),
              ('Ward', cluster.AgglomerativeClustering(n_clusters=k, linkage='ward')),
              ('Birch', cluster.Birch(threshold=0.1, n_clusters=k)))
    return algoritmos[i_alg]

k_values = [k for k in range(2,10)]
for i_alg in range(3):
    for k in k_values:
        _, alg = get_alg(i_alg, k)
        cluster_predict = alg.fit_predict(X_normal) 
        silh.append( float("{0:.2f}".format(
                metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(len(X)), random_state=123456))))
        calinski.append( float("{0:.2f}".format(
                metrics.calinski_harabasz_score(X_normal, cluster_predict))))
    buble_sizes = buble_sizes + buble_sizes_template[:len(k_values)]
    colors = colors + [all_colors[i_alg] for i in range(len(k_values))]
    param = param + k_values


from prettytable import PrettyTable
table = PrettyTable()
table.add_column('Algoritmo', ['DBScan' for i in range(1,5)] + 
                              ['KMeans' for k in k_values] + 
                              ['Ward' for k in k_values] +
                              ['Birch' for k in k_values])
table.add_column('Silh', silh)
table.add_column('Calinski-Harabasz', calinski)
table.add_column('Color', colors)
print(table)

plt.scatter(x=calinski, y=silh, s=buble_sizes, c=colors, alpha=0.5)

plt.xlabel('Calinski-Harabasz')
plt.ylabel('Silh')
plt.title('Comparativa entre algoritmos')
plt.figure(figsize=(10,10))

plt.show()

