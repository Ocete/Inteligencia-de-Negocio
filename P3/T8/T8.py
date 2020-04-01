import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

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
'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

print("iniciando preprocesado")

max_k1 = 33
selector = SelectKBest(chi2, k=max_k1).fit(X, Y)
X = selector.transform(X)
X_tst = selector.transform(X_tst)

print("preprocesado completado")
'''
# -----------------------------------------------------------------------

estimators = [
    ('rf',
     RandomForestClassifier(
         n_estimators=200, n_jobs=-1, max_depth=50, warm_start="True")),
    ('xgb',
     xgb.XGBClassifier(
         n_estimators=600, seed=323232, eta=0.1, max_depth=10)),
]

clf = StackingClassifier(estimators=estimators,
                         final_estimator=LogisticRegression())

print("iniciando entrenamiento")
clf = clf.fit(X, Y)
print("iniciando predicción")
y_pred_tst = clf.predict(X_tst)

#Y_pred_tra = clf.predict(X)
#print("F1 score (training): {:.4f}".format( f1_score(Y, Y_pred_tra, average='micro') ))
#validacion_cruzada(clf, X, Y, skf)

df_submission = pd.read_csv('../nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission.csv", index=False)
