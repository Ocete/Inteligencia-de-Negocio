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

algoritmos = (('KMeans', cluster.KMeans(init='k-means++', n_clusters=5, n_init=5)),
              ('MeanShift', cluster.MeanShift(cluster_all=False)),
              ('Ward', cluster.AgglomerativeClustering(n_clusters=5, linkage='ward')),
              ('DBScan', cluster.DBSCAN(eps=0.5, min_samples=10)),
              ('Birch', cluster.Birch(n_clusters=5)))

cluster_predict = {}
calinski = {}
silh = {}
times = {}
n_clusters = {}

clusters_fig, clusters_axis = plt.subplots(3, 2, figsize=(10,10))
clusters_colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', '#ffb347']

ijs = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

for i_alg, par in enumerate(algoritmos):
    name, alg = par
    print('----- Ejecutando ' + name,)
    t = time.time()
    cluster_predict[name] = alg.fit_predict(X_normal) 
    tiempo = time.time() - t
    times[name] = tiempo
    
    metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict[name])
    calinski[name] = metric_CH
    
    # El cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, 
    # digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    if len(X) > 10000:
       muestra_silhoutte = 0.2
    else:
       muestra_silhoutte = 1.0
       
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict[name], metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
    silh[name] = metric_SC

    # Asignamos de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict[name],index=X.index,columns=['cluster'])
    
    print("Tamaño de cada cluster:")
    size = clusters['cluster'].value_counts()
    cluster_fractions = []
    for num,i in size.iteritems():
       print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
       cluster_fractions.append( 100*i/len(clusters) )
    n_clusters[name] = len(size)
    
    # Bar charts
    
    if ( len(cluster_fractions) > 7 ):
        cluster_fractions = cluster_fractions[0:6]
    
    i, j = ijs[i_alg]
    y_pos = np.arange(len(cluster_fractions))
    labels = [ "Cluster " + str(i) for i in range(len(cluster_fractions)) ]
    
    clusters_axis[i, j].bar(y_pos, cluster_fractions, tick_label=labels, color=clusters_colors)
    clusters_axis[i, j].set_ylim(0, 100)
    clusters_axis[i, j].set_title(name)
    if (j == 0):        
        clusters_axis[i, j].set_ylabel("Cluster size (%)")
    
    
clusters_axis[2,1].remove()
clusters_fig.savefig("clusters.png")

# Heat Map
'''
_, alg = algoritmos[0]
cluster_predict[name] = alg.fit_predict(X_normal) 
centers = pd.DataFrame(alg.cluster_centers_, columns=list(X))
centers_desnormal = centers.copy()
# Convertimos los centros a los rangos originales antes de normalizar
for var in list(centers):
    centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

heatmap = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')

'''
    
from prettytable import PrettyTable
tabla = PrettyTable(['Algoritmo', 'CH', 'Silh', 'Tiempo', 'Número de clusters'])
for name, alg in algoritmos:
    tabla.add_row([name, "{0:.2f}".format(calinski[name]), "{0:.2f}".format(silh[name]), "{0:.2f}".format(times[name]), n_clusters[name]])
print(tabla)

'''
print("---------- Preparando el scatter matrix...")
# Se añade la asignación de clusters como columna a X
X_kmeans = pd.concat([X, clusters], axis=1)
sns.set()
variables = list(X_kmeans)
variables.remove('cluster')
sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
sns_plot.savefig("kmeans.png")
print("")
'''