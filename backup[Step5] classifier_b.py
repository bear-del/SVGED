import numpy as np
from util import LoadData_pickle
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from BaseSVDD import BaseSVDD
from sklearn import metrics

# dir = './model/dcae-C0ab/'
dir = './data/AUV/'
# dcae_only：仅使用DCAE对原始数据进行处理后的128维数据
# glow_enc：使用glow的encoder对原始数据进行处理后的3072维数据
# glow_enc_dcae：对glow_enc再使用DCAE进行处理后的128维数据

X_train = LoadData_pickle(path=dir, name='C0_train', type='rb')
X_train = np.squeeze(X_train)
# X_train = X_train[:, :192]

# des = X_train.std(axis=0)
# media = X_train.mean(axis=0)
# X_train = (X_train - media) / des
total_accum_precent = []
for i in range(5):
    # clf = IsolationForest(n_estimators=500)
    clf = OneClassSVM(nu=0.9, kernel='rbf', gamma='auto')
    # clf = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on')
    clf.fit(X_train)

    faults = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    accum_percent = []

    for fault in faults:
        X_test = LoadData_pickle(path=dir + fault, name='_test', type='rb')
        X_test = np.squeeze(X_test)
        # X_test = X_test[:, :192]
        # X_test = (X_test - media) / des
        y_pred_test = clf.predict(X_test)

        if fault == 'C0':
            n_error_test = y_pred_test[y_pred_test == -1].size
        else:
            n_error_test = y_pred_test[y_pred_test == 1].size
        percent = n_error_test / len(X_test)
        print('% test errors condition', fault, ':', percent)
        accum_percent.append(1 - percent)
    avg_accuracy = np.array(accum_percent).mean()
    print('Avg. Accuracy:', avg_accuracy)
    total_accum_precent.append(avg_accuracy)

print('Avg. Accuracy:', np.array(total_accum_precent).mean())

