# Práctica 3

## Indicaciones generales

- F1-score con average='micro' es la métrica que se usa en la competición para

- La línea 94 no hace falta si usamos las lineas 84-86. Usamos del 84 al 86 si estamos entrando (validación cruzada). Para la submision final utilizaremos todos los datos como entrenamiento.

- Las últimas tres líneas contienen como hacer la submision, rellenando submission.csv.

- Obtener la media de los F1-score que obtenemos en validación cruzada y añadir una línea con dicho valor a la tablita.

## Preprocesado

Las variables categóricas hacerlas binarias. Esto está hecho en **ejemplo_de_avanzado.py**. Con Boruta se seleccionan variables. También se utiliza Greed Search y Random Search.

Buscar tutoriales en Kernel.

## Algoritmos y ataque


- LGBM es más rápido que XGM pero no es mas eficaz. Es casi el mismo algoritmo. Puedes ejecutar random search o greedy search (probando distintos valores del algorimto) con este algorimto justo por eso. Cambiar también el número de hebras para que vaya rapidito jeje.

- Estudiar 1vs1 o 1vsAll (= OVA). Esto es en la configuración de LGBM.

- Estudiar preprocesado

- Estudiar algoritmos de desbalanceo

pip install --upgrade lightgbm
pip install --upgrade lgbm

# Ideas propias

Coger el tema 4 sobre preprocesado y aplicar todas las cosas que se dan. Te cada a las funciones objetivo a aplicar en la reducción de datos, utilizar **Filter**, así tendremos resultados independientes del algoritmos y podemos ejecutarlos después más rápido. Mirar la diapo 175/181 para estudiar el posible orden de aplicación de las técnicas de preprocesamiento.

# Cuaderno de Bitácora

# T1: Ejemplo de Jorge

- lgb.LGBMClassifier
- Category to number

# T2: Borramos category to number

Primer dia

- lgb.LGBMClassifier
- Sin preprocesado

Solo he borrado el category to number. Mejoramos a 0.7181

# T3: Noise reduction, feature selection y XGBoosting

Primer dia: Intento reducir ruido utilizando los algoritmos vistos en clase. Solo encuentro implementaciones en R. Me dispongo a usar R pero los algoritmos me dan problemas que no se resolver. Deshecho la idea de limpiar ruido. Con algoritmo conocidos.

Segundo dia: Intento aplicar selección de características. Utilizo los algoritmos de Mlxtend. Los algoritmos SFS, SBS, SFFS, SBFS tienen problemas de dependencias (les falta un import!) y no puedo ejecutarlos. Pruebo con ExhaustiveForwardSearch (EFS) pero no acaba, como es natural. Tambien pruebo a estudiar una matriz de correlaciones. Como en principio no acaba de ejecutar utilizo undersampling para ejecutarlo en un subgrupo de las instancias. Tampoco acaba.

Utilizamos el algoritmo SFS de Mlxtend.

Tercer dia: empiezo a usar XGBoosting y configurar sus parametros mientras dejo ejecutando un estudio sobre los parametros del SelectKBest. Configuro los parametros del XGBoosting y me da buenos resultados asi que lo subo.

F1-score: 0.7469 - sin preprocesado.

# T4: Feature selection

Tras completar el estudio de Feature selection subo el mejor valor de SelectKBest (k=33) con lgbm. Obtengo 0.7169, peor que el resultado original (T2) sin preprocesado.

0.602438213868688,
0.6116707107964416,
0.6493374976696515,
0.6579790612236598,
0.6593451418604825,
0.6590765269869187,
0.7116012685842023,
0.7122229120523216,
0.7133050286950359,
0.7127064072512184,
0.7118084796132996,
0.7123111708758809,
0.7122843047061089,
0.7130133859809624,
0.7148821451436069,
0.7150126162389444,
0.7149972654986965,
0.7154232057840849,
0.7159412346084174,
0.7161254233187216,
0.7154769242825262,
0.71592204561253,
0.7160678588342378,
0.7158299495640517,
0.7170847407391353,
0.7171845085870276,
0.7178675515558629,
0.7172382312819734,
0.7176948724588746,
0.7184393030015184,
0.7175874282469494,
0.7185544209270527,
0.7186925649261474,
0.7180786020921887,
0.7181208134068692,
0.7181860451261478,
0.7183318626179831,
0.7183625596811056,
0.7180210407734892,
0.7182666281010347,
0.7186733807157479,
0.7178790655865521,
0.718028715885933,
0.7184930307028204,
0.7178330139547914,
0.7182320932976328,
0.7176910403139352,
0.7174953350697634,
0.7179749951788051,
0.7185314011850605,
0.7178828971425087,
0.7178406901715786,
0.718485355295885,
0.7179135984021359,
0.7178023169652916,
0.7181246500428051,
0.7176066166538535,
0.7178330174150671,
0.7179519736698639,
0.7183932608671101,
0.7186925644107872,
0.7186503576607258,
0.7186503576607258,
0.7185314033201242,
0.7184316331162994,
0.7184316331162994,
0.7182397681892081,
0.7182397681892081

# T5: Subida errónea

En este intento subi de forma equivocada los resultados del intento 4.

# T6: Feature selection con XGBoosting

Unimos a continuación los resultados de los dos intentos anteriores, combinando la mejor configuración obtenida de XGBoosting con el preprocesado de SelectKBest con k=33. Obtenemos un f1-score de 0.7440, ligeramente peor que el 0.7469 del intento con XGBoosting sin preprocesado pero en la mitad de tiempo.

# T7: Selección de instancias y salto a stacking
