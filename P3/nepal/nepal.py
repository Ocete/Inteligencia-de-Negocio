import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import f1_score

def validacion_cruzada(modelo, X, Y, cv):
    Y_test_all = []
    print("Puntuaciones de la validacion cruzada: ")

    for train, test in cv.split(X, Y):
        t = time.time()
        modelo = modelo.fit(X[train],Y[train])
        tiempo = time.time() - t
        Y_pred = modelo.predict(X[test])
        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(f1_score(Y[test],Y_pred,average='micro') , tiempo))
        Y_test_all = np.concatenate([Y_test_all,Y[test]])

    print("")

    return modelo, Y_test_all

data_x = pd.read_csv('../nepal_earthquake_tra.csv')
data_y = pd.read_csv('../nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('../nepal_earthquake_tst.csv')

for l in ['building_id']:
    del data_x[l]
    del data_x_tst[l]
    del data_y[l]

data_x = pd.get_dummies(data_x)
data_x_tst = pd.get_dummies(data_x_tst)

X = data_x.values
X_tst = data_x_tst.values
y = np.ravel(data_y.values)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)

clf = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=600, num_leaves=90)
# clf, y_test_clf = validacion_cruzada(clf, X, y, skf)

t = time.time()
clf = clf.fit(X, y)
y_pred_tra = clf.predict(X)
tiempo = time.time() - t
print("F1 score (total): {:.4f}, tiempo: {:6.2f} segundos".format(
        f1_score(y, y_pred_tra,average='micro') , tiempo))

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('../nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission['damage_grade'] = [int(i) for i in df_submission['damage_grade']]
df_submission.to_csv("submission.csv", index=False)
