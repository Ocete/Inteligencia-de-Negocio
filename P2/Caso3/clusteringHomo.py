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
import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn import metrics
from sklearn import cluster
from math import floor
import seaborn as sns

# Cosas bonitas por defecto
sns.set()

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
    if (col == 'SEXO' or col == 'SEXOPAR'):
        censo[col].fillna(-1, inplace=True)
    else:
        censo[col].fillna(censo[col].mean(), inplace=True)

# Computamos el Coeficiente de Tareas
tareas = ['VESTIR', 'BANAR', 'ACOSTAR', 'COMIDAS', 'ENFERMOS', 'JUGAR', 'DEBERES', 'COLEGIO', 'ROPA', 'ELIGEEXTRAESC']
def ComputarCoeficienteTareas(row):
    def unaTarea(encargado_tarea):
        if (encargado_tarea != 1 and encargado_tarea != 2):
            return 0
        if (encargado_tarea == 1):
            return -1
        return 1
    
    coef = 0
    for tarea in tareas:
        coef = coef + unaTarea(row[tarea])
    return abs( coef / 10 )

coef = censo.apply(ComputarCoeficienteTareas, axis=1)
coef = pd.DataFrame({'COEFTAREAS': np.array(coef)})
censo = pd.concat([censo, coef], axis=1)

#seleccionar casos
subset = censo.loc[(censo['CONVIVEH14']==1) & (censo['SEXO'] == censo['SEXOPAR']) & (censo['SEXO'] != -1)] 

# Seleccionar variables 
usadas = ['EDAD', 'NHIJOSCONV', 'ESTUDIOSA', 'SATISFACENINOS', 'COEFTAREAS']

tareas.append('COEFTAREAS')
tareas.append('SEXO')
tareas.append('SEXOPAR')
migente = subset[tareas]
print(migente)

X = subset[usadas] 
X_normal = X.apply(norm_to_zero_one)

print('Tamaño de la población tras filtrado: ', len(X_normal.index))

for col in X:
   missing_count = sum(pd.isnull(censo[col]))
   if missing_count > 0:
      print(col, missing_count, ' AFTER')
    
algoritmos = (('KMeans', cluster.KMeans(init='k-means++', n_clusters=4, n_init=5)),
              ('MeanShift', cluster.MeanShift(cluster_all=False, min_bin_freq=3)),
              ('Ward', cluster.AgglomerativeClustering(n_clusters=4, linkage='ward')),
              ('DBScan', cluster.DBSCAN(eps=0.35, min_samples=5)),
              ('Birch', cluster.Birch(threshold=0.1, n_clusters=5)))

cluster_predict = {}
calinski = {}
silh = {}
times = {}
n_clusters = {}

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
       
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict[name], metric='euclidean', sample_size=floor(len(X)), random_state=123456)
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
    
    if (name == 'KMeans'):
        bar_fig, bar_axis = plt.subplots() 
        bar_axis.bar(y_pos, cluster_fractions, tick_label=labels, color=clusters_colors)
        bar_axis.set_ylim(0, 100)
        bar_axis.set_title(name)
        bar_axis.set_ylabel("Cluster size (%)")
    
bar_fig.savefig("homo/clusters.png")
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
#'''
with open('homo/general.csv', mode='w+', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    for name, _ in algoritmos:
        writer.writerow({'Algoritmo': name, 
                         'CH': "{0:.2f}".format(calinski[name]), 
                         'Silh': "{0:.2f}".format(silh[name]), 
                         'Tiempo': "{0:.2f}".format(times[name]), 
                         'Número de clusters': n_clusters[name]})
#'''
    
# ----------------------- FUNCIONES DE DISTRIBUCIÓN ---------
    
print("---------- Preparando funciones de distribución...")
    
n_clusters_kmeans = n_clusters['KMeans']
n_var = len(usadas)
X_kmeans = pd.concat([X, clusters_kmeans], axis=1)

fig, axes = plt.subplots(n_clusters_kmeans, n_var, sharey=True, figsize=(15,15))
fig.subplots_adjust(wspace=0, hspace=0)

colors = sns.color_palette(palette=None, n_colors=n_clusters_kmeans, desat=None)

rango = []
for j in range(n_var):
   rango.append([X_kmeans[usadas[j]].min(), X_kmeans[usadas[j]].max()])

for i in range(n_clusters_kmeans):
    dat_filt = X_kmeans.loc[X_kmeans['cluster']==i]
    for j in range(n_var):
        #ax = sns.kdeplot(dat_filt[usadas[j]], label="", shade=True, color=colors[i], ax=axes[i,j])
        ax = sns.boxplot(dat_filt[usadas[j]], color=colors[i], flierprops={'marker':'o','markersize':4}, ax=axes[i,j])

        if (i==n_clusters_kmeans-1):
            axes[i,j].set_xlabel(usadas[j])
        else:
            axes[i,j].set_xlabel("")
       
        if (j==0):
           axes[i,j].set_ylabel("Cluster "+str(i))
        else:
            axes[i,j].set_ylabel("")
       
        axes[i,j].set_yticks([])
        axes[i,j].grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
        axes[i,j].grid(axis='y', b=False)
       
        ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]),rango[j][1]+0.05*(rango[j][1]-rango[j][0]))
    
plt.show()
fig.savefig("homo/boxes.png")
    
# ---------------- SCATTER MATRIX -----------------------

#'''
plt.clf()
print("---------- Preparando el scatter matrix...")
# Se añade la asignación de clusters como columna a X
variables = list(X_kmeans)
variables.remove('cluster')
sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25, 'alpha': 0.7}, diag_kind="hist")
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
sns_plot.savefig("homo/scatter_matrix.png")
plt.show()
#'''

# ----------------------- HEATMAPS -----------------------

#'''
plt.figure(1)
centers = pd.DataFrame(alg_kmeans.cluster_centers_, columns=list(X))
centers_desnormal = centers.copy()
#centers_desnormal = centers_desnormal.drop([4])

# Calculamos los centroides
X = pd.concat([X, clusters_kmeans], axis=1)
for variable in list(centers):
    for k_cluster in range(n_clusters_kmeans):
        centroide = X.loc[(clusters_kmeans['cluster']==k_cluster)][variable].mean() 
        centers_desnormal.loc[k_cluster, variable] = centroide
        
# Normalizamos
centers_normal2 = centers_desnormal.copy()
centers_normal2 = (centers_normal2 - centers_normal2.min() ) / (centers_normal2.max() - centers_normal2.min())

import matplotlib.pyplot as plt
heatmap_fig, ax = plt.subplots(figsize=(10,10))
heatmap = sns.heatmap(centers_normal2, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')


# Para evitar que los bloques de arriba y abajo se corten por la mitad
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
heatmap_fig.savefig("homo/heatmap.png")
#'''
