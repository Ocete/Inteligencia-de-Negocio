import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

le = preprocessing.LabelEncoder()

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
y = np.ravel(data_y.values)

#------------------------------------------------------------------------
'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
le = preprocessing.LabelEncoder()

from sklearn.metrics import f1_score

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(f1_score(y[test],y_pred,average='micro') , tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all
#------------------------------------------------------------------------

'''
print("------ XGB...")
clf = xgb.XGBClassifier(n_estimators = 200)
#clf, y_test_clf = validacion_cruzada(clf,X,y,skf)
#'''

#'''

clf = xgb.XGBClassifier(n_estimators=600,
                        reg_alpha=0.3,
                        seed=123456,
                        eta=0.1,
                        max_depth=10)

# lgbm, y_test_lgbm = validacion_cruzada(lgbm, X, y, skf)

clf = lgbm
clf = clf.fit(X, y)
y_pred_tra = clf.predict(X)
print("F1 score: {:.4f}".format( f1_score(y, y_pred_tra, average='micro') ))
y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('../nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("submission.csv", index=False)
