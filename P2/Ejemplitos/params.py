    # -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics as SKmetrics
from sklearn import cluster

from math import floor

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

censo = pd.read_csv('mujeres_fecundidad_INE_2018.csv')

# Sustituimos valores perdidos con la media      
for col in censo:
   censo[col].fillna(censo[col].mean(), inplace=True)
      
#seleccionar casos
subset = censo.loc[(censo['EDAD']>20) & (censo['EDAD']<=40)]

#seleccionar variables de enterés para clustering
usadas = ['NHOGAR', 'NTRABA', 'TEMPRELA', 'NDESEOHIJO']
X = subset[usadas]

X_normal = X.apply(norm_to_zero_one)
 
'''
# Estudio del Silh para Kmeans
silh = []
calinkski = []
k_values = [k for k in range(2,20)]
for k in k_values:
    alg = cluster.KMeans(init='k-means++', n_clusters=k, n_init=5)
    cluster_predict = alg.fit_predict(X_normal) 
    silh.append( "{0:.2f}".format(
            SKmetrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(len(X)), random_state=123456)))
    calinkski.append( "{0:.2f}".format(
            SKmetrics.calinski_harabasz_score(X_normal, cluster_predict)))

from prettytable import PrettyTable
table = PrettyTable()
table.add_column('Número de clusters', k_values)
table.add_column('Silh', silh)
table.add_column('Calinski-Harabasz', calinkski)
print(table)

plt.gca().invert_yaxis()
plt.plot(k_values, silh)
#if (plt.gca().yaxis_inverted()):
plt.show()
'''

# Estudio del radio para DBScan
plt.clf()
silh = []
calinkski = []
n_clusters = []
rad_values = [r/1000.0 for r in range (20, 400, 20)]

for rad in rad_values:
    alg = cluster.DBSCAN(eps=rad, min_samples=20)
    cluster_predict = alg.fit_predict(X_normal)
    silh.append( "{0:.2f}".format(
            SKmetrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(len(X)), random_state=123456)))
    calinkski.append( "{0:.2f}".format(
            SKmetrics.calinski_harabasz_score(X_normal, cluster_predict)))
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
    n_clusters.append( len(clusters['cluster'].value_counts()) )    

from prettytable import PrettyTable
table = PrettyTable()
table.add_column('Radio de cluster', rad_values)
table.add_column('Silh', silh)
table.add_column('Calinski-Harabasz', calinkski)
table.add_column('N clusters', n_clusters)
print(table)

plt.plot(rad_values, silh)

plt.ylim(-1,1)
plt.show()