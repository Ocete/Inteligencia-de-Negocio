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
   censo[col].fillna(censo[col].mean(), inplace=True)
      
#seleccionar casos
subset = censo.loc[(censo['TRAREPRO']==1) & (censo['NEMBTRAREPRO']<=6)]
    
# Seleccionar variables 
usadas = ['NHIJOS', 'TIPOTRAREPRO', 'NMESESTRAREPRO', 'NEMBTRAREPRO']

X = subset[usadas]
    
X_normal = X.apply(norm_to_zero_one)

print('Tamaño de la población tras filtrado: ',len(X_normal.index))

for col in X:
   missing_count = sum(pd.isnull(censo[col]))
   if missing_count > 0:
      print(col,missing_count, ' AFTER')
    
algoritmos = (('KMeans', cluster.KMeans(init='k-means++', n_clusters=5, n_init=5)),
              ('MeanShift', cluster.MeanShift(cluster_all=False, min_bin_freq=3)),
              ('Ward', cluster.AgglomerativeClustering(n_clusters=4, linkage='ward')),
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
       
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict[name], metric='euclidean', sample_size=floor(len(X)), random_state=123456)
    silh[name] = metric_SC

    # Asignamos de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict[name],index=X.index,columns=['cluster'])
    
    if (name == 'KMeans'):
        clusters_kmeans = clusters
        alg_kmeans = alg
    elif (name == 'Ward'):
        clusters_ward = clusters
    
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
#clusters_fig.savefig("clusters.png")
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
    
# ----------------------- FUNCIONES DE DISTRIBUCIÓN ---------
    
print("---------- Preparando funciones de distribución...")
    
n_clusters_ward = n_clusters['Ward']
n_var = len(usadas)
X_ward = pd.concat([X, clusters_ward], axis=1)

fig, axes = plt.subplots(n_clusters_ward, n_var, sharey=True, figsize=(15,15))
fig.subplots_adjust(wspace=0, hspace=0)

colors = sns.color_palette(palette=None, n_colors=n_clusters_ward, desat=None)

rango = []
for j in range(n_var):
   rango.append([X_ward[usadas[j]].min(), X_ward[usadas[j]].max()])

for i in range(n_clusters_ward):
    dat_filt = X_ward.loc[X_ward['cluster']==i]
    for j in range(n_var):
        #ax = sns.kdeplot(dat_filt[usadas[j]], label="", shade=True, color=colors[i], ax=axes[i,j])
        ax = sns.boxplot(dat_filt[usadas[j]], color=colors[i], flierprops={'marker':'o','markersize':4}, ax=axes[i,j])

        if (i==n_clusters_ward-1):
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
#fig.savefig("boxes.png")
    
# ---------------- SCATTER MATRIX -----------------------

'''
plt.clf()
print("---------- Preparando el scatter matrix...")
# Se añade la asignación de clusters como columna a X
variables = list(X_ward)
variables.remove('cluster')
sns_plot = sns.pairplot(X_ward, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist")
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
# sns_plot.savefig("scatter_matrix.png")
plt.show()
#'''

# ----------------------- DENDOGRAMAS -----------------------

#En clustering hay que normalizar para las métricas de distancia
# X_normal = preprocessing.normalize(X, norm='l2')
X_normal = (X - X.min() ) / (X.max() - X.min())


#Vamos a usar este jerÃ¡rquico y nos quedamos con 100 clusters, es decir, cien ramificaciones del dendrograma
ward = cluster.AgglomerativeClustering(n_clusters=20, linkage='ward')
name, algorithm = ('Ward', ward)

cluster_predict = {}
k = {}

t = time.time()
cluster_predict[name] = algorithm.fit_predict(X_normal) 
tiempo = time.time() - t
k[name] = len(set(cluster_predict[name]))

# Se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict['Ward'],index=X.index,columns=['cluster'])
# Y se añade como columna a X
X_cluster = pd.concat([X, clusters], axis=1)

# Filtro quitando los elementos (outliers) que caen en clusters muy pequeÃ±os en el jerÃ¡rquico
min_size = 3
X_filtrado = X
'''
X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
k_filtrado = len(set(X_filtrado['cluster']))
print("De los {:.0f} clusters hay {:.0f} con mÃ¡s de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k['Ward'],k_filtrado,min_size,len(X),len(X_filtrado)))
X_filtrado = X_filtrado.drop('cluster', 1)
X_filtrado = X
#'''
#Normalizo el conjunto filtrado
X_filtrado_normal = preprocessing.normalize(X_filtrado, norm='l2')

# Obtengo el dendrograma usando scipy, que realmente vuelve a ejecutar el clustering jerárquico
from scipy.cluster import hierarchy
linkage_array = hierarchy.ward(X_filtrado_normal)
plt.clf()
dendro = hierarchy.dendrogram(linkage_array,orientation='left', p=10, truncate_mode='lastp') #lo pongo en horizontal para compararlo con el generado por seaborn
# puedo usar "p=10,truncate_mode='lastp'" para cortar el dendrograma en 10 hojas

# Dendograma usando seaborn (que a su vez usa scipy) para incluir un heatmap
X_filtrado_normal_DF = pd.DataFrame(X_filtrado_normal, index=X_filtrado.index, columns=usadas)

# Añadimos una columna de label para indicar el cluster al que pertenece cada objeto
labels = X_ward['cluster']
lut = dict(zip(set(labels), sns.color_palette(palette="Blues_d", n_colors=n_clusters_ward)))
row_colors = pd.DataFrame(labels)['cluster'].map(lut)
clustergrid = sns.clustermap(X_filtrado_normal_DF, method='ward', row_colors=row_colors, col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)

# Para añadir los labels reordenados. Ahora mismo no salen los colores en la 
# columna donde deberian. Intuyo que esto se debe a que los ids no encajan.

#'''
ordering = clustergrid.dendrogram_row.reordered_ind
labels_list = [x for _, x in sorted(zip(ordering,labels), key=lambda pair: pair[0])]
labels = pd.Series(labels_list, index=X_filtrado_normal_DF.index, name='cluster') 
lut = dict(zip(set(labels), sns.color_palette(palette="Blues_d", n_colors=n_clusters_ward)))
row_colors = pd.DataFrame(labels)['cluster'].map(lut)
clustergrid = sns.clustermap(X_filtrado_normal_DF, method='ward', row_colors=row_colors, col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)
#'''

#plt.savefig("dendograma.png")

# ----------------------- HEATMAPS -----------------------

#'''
plt.figure(1)
centers = pd.DataFrame(alg_kmeans.cluster_centers_, columns=list(X))
centers_desnormal = centers.copy()
centers_desnormal = centers.drop([4])

# Calculamos los centroides
X = pd.concat([X, clusters_ward], axis=1)
for variable in list(centers):
    for k_cluster in range(n_clusters_ward):
        centroide = X.loc[(clusters_ward['cluster']==k_cluster)][variable].mean() 
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
#heatmap_fig.savefig("heatmap.png")
#'''
