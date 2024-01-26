import tensorflow as tf
import os
import tensorflow.keras.models as keras_models
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score



models = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
m = models[3]



for i in range(0, 300):
    model = keras_models.load_model(f"./Model-IMR90/{m}Model{i}.tf")

    names = ['IMR90']
    for name in names:
        Data_dir = './data/%s/' % name
        test = np.load(Data_dir + '%s_test.npz' % name)
        X_en_tes, X_pr_tes, y_tes = test['X_en_tes'], test['X_pr_tes'], test['y_tes']
      
        print("****************Testing %s cell line specific model on %s cell line****************" % (m, name))
        y_pred = model.predict([X_en_tes, X_pr_tes])
        auc = roc_auc_score(y_tes, y_pred)
        aupr = average_precision_score(y_tes, y_pred)
        print(i)
        print("AUC : ", auc)
        print("AUPR : ", aupr)