# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
Fecha:
    Noviembre/2019
Contenido:
    Ejemplo de clustering jerÃ¡rquico en Python
    Inteligencia de Negocio
    Grado en IngenierÃ­a InformÃ¡tica
    Universidad de Granada
"""

'''
DocumentaciÃ³n sobre clustering jerÃ¡rquico en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://seaborn.pydata.org/generated/seaborn.clustermap.html
'''

import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn import preprocessing

datos = pd.read_csv('mujeres_fecundidad_INE_2018.csv')

#Se pueden reemplazar los valores desconocidos por un nÃºmero
#datos = datos.replace(np.NaN,0)

#O imputar, por ejemplo con la media      
for col in datos:
   datos[col].fillna(datos[col].mean(), inplace=True)

#seleccionar casos
subset = datos.loc[(datos['EDAD']>20) & (datos['EDAD']<=40) & (datos['COD_CCAA']==1)]

#seleccionar variables de interÃ©s para clustering
usadas = ['RELIGION', 'NHOGAR', 'NTRABA', 'TEMPRELA', 'NDESEOHIJO']
X = subset[usadas]

#Para sacar el dendrograma en el jerÃ¡rquico, no puedo tener muchos elementos.
#Hago un muestreo aleatorio para quedarme solo con 1000, aunque lo ideal es elegir un caso de estudio que ya dÃ© un tamaÃ±o asÃ­
if len(X)>1000:
   X = X.sample(1000, random_state=123456)

#En clustering hay que normalizar para las mÃ©tricas de distancia
X_normal = preprocessing.normalize(X, norm='l2')

#Vamos a usar este jerÃ¡rquico y nos quedamos con 100 clusters, es decir, cien ramificaciones del dendrograma
ward = cluster.AgglomerativeClustering(n_clusters=100, linkage='ward')
name, algorithm = ('Ward', ward)

cluster_predict = {}
k = {}

print(name,end='')
t = time.time()
cluster_predict[name] = algorithm.fit_predict(X_normal) 
tiempo = time.time() - t
k[name] = len(set(cluster_predict[name]))
print(": k: {:3.0f}, ".format(k[name]),end='')
print("{:6.2f} segundos".format(tiempo))

#se convierte la asignaciÃ³n de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict['Ward'],index=X.index,columns=['cluster'])
#y se aÃ±ade como columna a X
X_cluster = pd.concat([X, clusters], axis=1)

#Filtro quitando los elementos (outliers) que caen en clusters muy pequeÃ±os en el jerÃ¡rquico

min_size = 3
X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
k_filtrado = len(set(X_filtrado['cluster']))
print("De los {:.0f} clusters hay {:.0f} con mÃ¡s de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k['Ward'],k_filtrado,min_size,len(X),len(X_filtrado)))
X_filtrado = X_filtrado.drop('cluster', 1)

X_filtrado = X

#Normalizo el conjunto filtrado
X_filtrado_normal = preprocessing.normalize(X_filtrado, norm='l2')

#Saco el dendrograma usando scipy, que realmente vuelve a ejecutar el clustering jerÃ¡rquico
from scipy.cluster import hierarchy
linkage_array = hierarchy.ward(X_filtrado_normal)
plt.figure(1)
plt.clf()
dendro = hierarchy.dendrogram(linkage_array,orientation='left', p=10,truncate_mode='lastp') #lo pongo en horizontal para compararlo con el generado por seaborn
#puedo usar, por ejemplo, "p=10,truncate_mode='lastp'" para cortar el dendrograma en 10 hojas

#Ahora lo saco usando seaborn (que a su vez usa scipy) para incluir un heatmap
import seaborn as sns
X_filtrado_normal_DF = pd.DataFrame(X_filtrado_normal,index=X_filtrado.index,columns=usadas)
sns.clustermap(X_filtrado_normal_DF, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)