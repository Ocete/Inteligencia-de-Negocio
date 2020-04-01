import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

#importing the necessary libraries
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LinearRegression
from sklearn.utils import column_or_1d

from sklearn.metrics import accuracy_score

# lectura de datos
data_x = pd.read_csv('../nepal_earthquake_tra.csv')
data_y = pd.read_csv('../nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('../nepal_earthquake_tst.csv')

data_x.drop(labels=['building_id'], axis=1,inplace = True)
data_x_tst.drop(labels=['building_id'], axis=1,inplace = True)
data_y.drop(labels=['building_id'], axis=1,inplace = True)

data_x = pd.get_dummies(data_x)
data_x_tst = pd.get_dummies(data_x_tst)

X = data_x.values
X_tst = data_x_tst.values
Y = np.ravel(data_y.values)

# Undersampling ---------------------------------------------------------
#'''
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, Y_resampled = rus.fit_resample(X, Y)
#X_resampled, Y_resampled = rus.fit_resample(X_resampled, Y_resampled)
#'''

# Validación cruzada ---------------------------------------------------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
def validacion_cruzada(modelo, X, Y, cv):
    Y_test_all = []
    f1_total = 0
    #print("Resultados de validacion cruzada:")
    for train, test in cv.split(X, Y):
        t = time.time()
        modelo = modelo.fit(X[train],Y[train])
        tiempo = time.time() - t
        Y_pred = modelo.predict(X[test])
        f1 = f1_score(Y[test],Y_pred,average='micro')
        f1_total += f1
        #print("F1-score: {:.4f}, tiempo: {:6.2f} segundos".format(f1 , tiempo))
        Y_test_all = np.concatenate([Y_test_all,Y[test]])

    f1_total /= 5
    print(f1_total)

    #return modelo, Y_test_all
    return f1_total
#------------------------------------------------------------------------

# --------- Selección de características con SFS Sequential Forward Selection(sfs)

lgbm = lgb.LGBMClassifier(n_estimators=200, n_jobs=4)

print("iniciando preprocesado")

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
'''
selector = SelectKBest(chi2, k=10).fit(X, Y)
X = selector.transform(X)
X_tst = selector.transform(X_tst) 
'''

'''
#clf = ExtraTreesClassifier(n_estimators=50)
clf = lgbm
clf = clf.fit(X, Y)
model = SelectFromModel(clf, prefit=True)
X = model.transform(X)
X_tst = model.transform(X_tst)
#'''
print("preprocesado completado")

'''
clf = lgbm
k1_array = [i for i in range(1,69)]
f1_array = []
for k1 in k1_array:
    selector = SelectKBest(chi2, k=k1).fit(X, Y)
    X_new = selector.transform(X)
    print("Caracteristicas: " + str(k1))
    f1 = validacion_cruzada(clf, X_new, Y, skf)
    f1_array.append(f1)
'''

k1_array = [i for i in range(1,69)]
f1_array = [0.602438213868688, 0.6116707107964416, 0.6493374976696515,0.6579790612236598,
 0.6593451418604825, 0.6590765269869187, 0.7116012685842023, 0.7122229120523216,
 0.7133050286950359,0.7127064072512184, 0.7118084796132996, 0.7123111708758809,
 0.7122843047061089, 0.7130133859809624, 0.7148821451436069, 0.7150126162389444,
 0.7149972654986965, 0.7154232057840849, 0.7159412346084174, 0.7161254233187216,
 0.7154769242825262, 0.71592204561253, 0.7160678588342378, 0.7158299495640517,
 0.7170847407391353, 0.7171845085870276, 0.7178675515558629, 0.7172382312819734,
 0.7176948724588746, 0.7184393030015184, 0.7175874282469494, 0.7185544209270527,
 0.7186925649261474, 0.7180786020921887, 0.7181208134068692, 0.7181860451261478,
 0.7183318626179831, 0.7183625596811056, 0.7180210407734892, 0.7182666281010347,
 0.7186733807157479, 0.7178790655865521, 0.718028715885933, 0.7184930307028204,
 0.7178330139547914, 0.7182320932976328, 0.7176910403139352, 0.7174953350697634,
 0.7179749951788051, 0.7185314011850605, 0.7178828971425087, 0.7178406901715786,
 0.718485355295885, 0.7179135984021359, 0.7178023169652916, 0.7181246500428051,
 0.7176066166538535, 0.7178330174150671, 0.7179519736698639, 0.7183932608671101,
 0.7186925644107872, 0.7186503576607258, 0.7186503576607258, 0.7185314033201242,
 0.7184316331162994, 0.7184316331162994, 0.7182397681892081, 0.7182397681892081]

max_f1 = max(f1_array)
max_k1 = 0
for i in k1_array:
    if f1_array[i-1] == max_f1:
        max_k1 = i

print("Max k1: ", max_k1)

selector = SelectKBest(chi2, k=max_k1).fit(X, Y)
X = selector.transform(X)
X_tst = selector.transform(X_tst)
# -----------------------------------------------------------------------

#'''
clf = lgbm
clf = clf.fit(X, Y)
Y_pred_tra = clf.predict(X)
print("F1 score (training): {:.4f}".format( f1_score(Y, Y_pred_tra, average='micro') ))
y_pred_tst = clf.predict(X_tst)

validacion_cruzada(clf, X, Y, skf)

df_submission = pd.read_csv('../nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission.csv", index=False)
#'''