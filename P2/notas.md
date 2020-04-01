Tipos de variables:
- Categoricas: NO USAR PARA PREDECIR - esto serán los casos de estudio
- Utilizar variables numericas (continuas o discretas)
- Ordinales (numéricas que representan cosas pero existen
	un orden entre los valores. Ejemplo: estudios

Caso estudio minimo/maximo: [400, varios miles]. Puede ser comparativa interesante estudiar un subconjunto de datos frente al total. Numero de variables a usar por comparativa: maximo 8/10, aunque 4 o 5 pueden estar bien. No menos de 4!

Podemos sacar los numeros (clases) de los clusteres y despues utilizar un arbol de decisión para poder interpretar mejor los datos.

# Primer Seminario: clustering.py:

La variable religion se utiliza para distinguir entre creyentes (valores cercanos a 1) y catolicos (cercanos a 6). Podriamos haberlo usado unicamente con dos clusters. De todas formas esto no es un buen caso de estudio: las variables no son interesantes.

Cuidado al interpretar los centroidos (la primera gráfica). Pueden ser más o menos relevantes a nivel lógico.

En el XLSX podemos utilizar la variable TIPO (numericos o categoricos) para buscar variables que estudiar. Mirando la *scatter matrix* podemos estudiar que características tiene cada cluster.

En cuanto a los filtros, hemos usado la edad para filtrar, pero es una variable super buena para añadir en nuestro clustering. Por ejemplo para coger a la gente de Andalucia, miramos en el XLSX que la variable es *COD_CCAA*, en la tabla 1 vemos que el valor =1 indica andalucia. Considerar también el tamaño de la población tras haber filtrado.

También se pueden utilizar análisis comparados entre varios tipos de clustering.

Tras filtrar y tomar las variables que queremos estudiar, realizamos normalización lineal entre 0 y 1. En esta normalización podemos dar "pesos" a algunas variables si son más importantes.

# Ejecución:

Hacer una lista de algoritmos y ejecutar para todos ellos. KMeans fija el número de clusteres a K. Podemos comparar algunas métricas (dos) y el tiempo de ejecución. Para calcular el Silhouette podemos hacer un muestreo de los datos. Para la ejecución final usar todos los datos. Para interpretar los centroides, desnormalizamos. Utilizar también Mean Shift, que es parecido a Kmeans.

### Nota: en caso de ejecución lenta
Si en algún momento tenemos un algoritmo muy lento, tomar una muestra de los datos y probar el algoritmo para si merece la pena:
X = X.sample(100)

# Segundo Seminario - clustering_jerarquico.py:

**DBSCAN** tiene dos parametros (radio y numero de vecinos), y se usa para detectar bien los clusteres. **BIRCH** también se usa. Es eficiente pues van observando los datos poco a poco. Como es mas antiguo no tiene otras capacidades como adaptarse a cambios en los datos (concert drifts).

Silhoutte: es el ratio de la distancia entre elementos del clusters frente a otros clusters. Se calcula punto a punto. Buscamos:
- Que la distancia intra cluster sea reducida (los elementos de un mismo cluster este agrupados).
- Que la distancia entre clusters sea máxima.
Que valga 0 significa que el punto podría estar en dos clusters. Eso ya es de por si bastante malo. Que sea negativo significa que estaría mejor que estuviese en otro cluster. El coeficiente es la media del valor Silhouette para cada punto.

Clustering jerárquico (aglomerativo con enlace simple). Calculamos afinidad (que a veces es distancia, pero no siempre) entre elementos. Vamos agrupando los de mayor afinidad. Los outlayer son los que tienen poca afinidad o están lejos del resto. En este algoritmo los outlayers se agrupan en el nivel alto de la jerarquía (al final).

Enlace simple minimiza la distancia mínima, el otro minimiza la distancia máxima.

El algoritmo **WARD** agrupa elementos temporalmente y calculando su varianza, minimizandola. Asi aumentamos la compactación de los clusters.

Para usar el jerárquico, lo más interesante es el orden en el que se han ido agrupando. Para ver a que nivel corto en el árbol utilizo *n_clusters*.

En el script, el proceso de filtrado podemos reiterarlo varias veces quitando nuevos outlayers. En las prácticas, el jerárquico puede ser útil con pocos elementos. Para el heat map, el clustering marcará el orden y de ahí podemos ver las caracteríticas de cada clustering.

Para pintar cositas en 2 dimensiones a partir de N dimensiones, utilizar el algoritmo **MDS**.

## Cosas que hacer:
Poner los tamaños de los clusters en algun modo de visualización bonito.

Podemos pintar añadiendo la clasificación (C1, C2, C3) a cada fila. A partir de ello construir un arbol de decisión.

Para el análisis, no utilizar siempre el mismo algoritmo. Utilizar los centroides (que puede ser más o menos representativo), asi como un estudio con los jerárquicos. Para uno de ellos podemos hacer lo del árbol de decisión.

Dentro de un mismo caso de estudio podemos comparar distintos trozos del mismo clúster, o estudiando distintas subpoblaciones y ver como son segmentadas por distintas variables.

## Dudas

- Como saber el numero de clusters en el WARD
- Se veo feo la gráfica clusters.
