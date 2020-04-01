    # -*- coding: utf-8 -*-
'''
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''

import time
import csv

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn import cluster
from math import floor
import seaborn as sns

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

censo = pd.read_csv('../mujeres_fecundidad_INE_2018.csv')

'''
for col in censo:
   missing_count = sum(pd.isnull(censo[col]))
   if missing_count > 0:
      print(col,missing_count)
#'''

#Se pueden reemplazar los valores desconocidos por un número
#censo = censo.replace(np.NaN,0)

# Sustituimos valores perdidos con la media      
for col in censo:
   censo[col].fillna(censo[col].mean(), inplace=True)
      
#seleccionar casos
subset = censo.loc[(censo['TRABAJAACT']==1) & (censo['NDESEOHIJO']<=10)
                    & (censo['NHOGAR']<=7)] 

# Seleccionar variables 
usadas = ['NHBIOADOP', 'EDAD', 'NTRABA', 'TEMPRELA', 'NHOGAR']
X = subset[usadas]

X_normal = X.apply(norm_to_zero_one)

print('Tamaño de la población tras filtrado: ',len(X_normal.index))

for col in X:
   missing_count = sum(pd.isnull(censo[col]))
   if missing_count > 0:
      print(col,missing_count, ' AFTER')
    
algoritmos = (('KMeans', cluster.KMeans(init='k-means++', n_clusters=5, n_init=5)),
              ('MeanShift', cluster.MeanShift(cluster_all=False, min_bin_freq=3)),
              ('Ward', cluster.AgglomerativeClustering(n_clusters=5, linkage='ward')),
              ('DBScan', cluster.DBSCAN(eps=0.35, min_samples=5)),
              ('Birch', cluster.Birch(threshold=0.1,n_clusters=5)))

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
    #if len(X) > 10000:
    #   muestra_silhoutte = 0.2
    #else:
    muestra_silhoutte = 1.0
       
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict[name], metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
    silh[name] = metric_SC

    # Asignamos de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict[name],index=X.index,columns=['cluster'])
    
    if (name == 'KMeans'):
        clusters_kmeans = clusters
        alg_kmeans = alg
    
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
# clusters_fig.savefig("clusters.png")
plt.show()
    
from prettytable import PrettyTable
header = ['Algoritmo', 'CH', 'Silh', 'Tiempo', 'Número de clusters']
tabla = PrettyTable(header)
for name, alg in algoritmos:
    tabla.add_row([name, 
                   "{0:.2f}".format(calinski[name]), 
                   "{0:.2f}".format(silh[name]), 
                   "{0:.2f}".format(times[name]), 
                   n_clusters[name]])
print(tabla)

# Escribir los datos en un general.csv
'''
with open('general.csv', mode='w+', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    for name, _ in algoritmos:
        writer.writerow({'Algoritmo': name, 
                         'CH': "{0:.2f}".format(calinski[name]), 
                         'Silh': "{0:.2f}".format(silh[name]), 
                         'Tiempo': "{0:.2f}".format(times[name]), 
                         'Número de clusters': n_clusters[name]})
#'''
    
# ------------------- BUBBLES ---------------------------
plt.clf()

all_colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', '#ffb347']
buble_sizes_template =[i*50 for i in range(1,20)]
cluster_predict = {}
calinski = []
silh = []
buble_sizes = []
param = []
k_clusters = []

# DBScan
rad_values = [r/20 for r in range(1, 10)]
for rad in rad_values:
    alg = cluster.DBSCAN(eps=rad, min_samples=20)
    cluster_predict = alg.fit_predict(X_normal)
    silh.append( float("{0:.2f}".format(
            metrics.silhouette_score(X_normal, cluster_predict, 
                metric='euclidean', sample_size=floor(len(X)), random_state=123456))))
    calinski.append( float("{0:.2f}".format(
            metrics.calinski_harabasz_score(X_normal, cluster_predict))))    
    Bclusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
    k_clusters.append(len(Bclusters['cluster'].value_counts()))

buble_sizes = buble_sizes_template[:len(rad_values)]
colors = [all_colors[0] for i in range(len(rad_values))]
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
                metrics.silhouette_score(X_normal, cluster_predict, 
                    metric='euclidean', sample_size=floor(len(X)), random_state=123456))))
        calinski.append( float("{0:.2f}".format(
                metrics.calinski_harabasz_score(X_normal, cluster_predict))))    
        Bclusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
        k_clusters.append(len(Bclusters['cluster'].value_counts()))
    buble_sizes = buble_sizes + buble_sizes_template[:len(k_values)]
    colors = colors + [all_colors[i_alg+1] for i in range(len(k_values))]
    param = param + k_values

from prettytable import PrettyTable
table = PrettyTable()
duplicated_names = ['DBScan' for i in range(len(rad_values))] + ['KMeans' for k in k_values] + ['Ward' for k in k_values] + ['Birch' for k in k_values]

table.add_column('Algoritmo', duplicated_names)
table.add_column('Silh', silh)
table.add_column('Calinski-Harabasz', calinski)
table.add_column('Color', colors)
table.add_column('Param', param)
table.add_column('Clusters', k_clusters)
print(table)

'''
header = ['Algoritmo', 'Silh', 'Calinski-Harabasz', 'Parámetro', 'N. clusters']
with open('comparativa.csv', mode='w+') as file:
    file.truncate()
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    for i, name in enumerate(duplicated_names):
        writer.writerow({'Algoritmo': name, 
                         'Silh': "{0:.2f}".format(silh[i]), 
                         'Calinski-Harabasz': "{0:.2f}".format(calinski[i]), 
                         'Parámetro': "{0:.2f}".format(param[i]), 
                         'N. clusters': k_clusters[i]})
#'''

plt.scatter(x=calinski, y=silh, s=buble_sizes, c=colors, alpha=0.5)

names = ['DBScan', 'KMeans', 'Ward', 'Birch']
patches = []
for i, name in enumerate(names):
    patches.append( mpatches.Patch(color=all_colors[i], label=name) )

plt.legend(handles=patches, title="Algoritmos")

plt.xlabel('Calinski-Harabasz')
plt.ylabel('Silh')
plt.title('Comparativa entre algoritmos')
plt.figure(figsize=(10,10))

# Por alguna razón esto no funciona
# .savefig("bubbles.png")
plt.show()

# -------------------------------------------------------
# ---------------- Scatter Matrix -----------------------

#'''
print("---------- Preparando el scatter matrix...")
# Se añade la asignación de clusters como columna a X
X_kmeans = pd.concat([X, clusters_kmeans], axis=1)
sns.set()
variables = list(X_kmeans)
variables.remove('cluster')
sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist")
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
# sns_plot.savefig("scatter_matrix.png")
print("")

plt.show()
#'''

# -------------------------------------------------------
# ---------------- Heat Map -----------------------------

#'''
plt.clf()
centers = pd.DataFrame(alg_kmeans.cluster_centers_, columns=list(X))
centers_desnormal = centers.copy()
# Convertimos los centros a los rangos originales antes de normalizar
for var in list(centers):
    centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())
import matplotlib.pyplot as plt
heatmap_fig, ax = plt.subplots(figsize=(10,10))
heatmap = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
# heatmap_fig.savefig("heatmap.png")
# Para evitar que los bloques de arriba y abajo se corten por la mitad
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#'''

Q1, Q3 = X.quantile(0.25), X.quantile(0.75)
IQR = Q3-Q1 
X = X[~((X < (Q1 - 1.5*IQR)) | (X > (Q3 + 1.5*IQR))).any(axis=1)]