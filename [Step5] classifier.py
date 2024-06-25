import numpy as np
from util import LoadData_pickle
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from BaseSVDD import BaseSVDD
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics

# dir = './model/dcae-C0fb/'
# dir = './model/glow-da/'
# dir = './data/AUV/STFT/'
dir = './final/5/'
type = ''
type1 = 'SVDD'
clf = None

X_train = LoadData_pickle(path=dir, name='C0_train'+ type, type='rb')
X_train = X_train.reshape((840, -1))

# des = X_train.std(axis=0)
# media = X_train.mean(axis=0)
# X_train = (X_train - media) / des
total_accum_percent = []
for i in range(2):
    if type1 == 'IF':
        clf = IsolationForest(n_estimators=100)
    elif type1 == 'OCSVM':
        clf = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
    elif type1 == 'SVDD':
        clf = BaseSVDD(C=0.8, gamma='auto', kernel='rbf', display='on')
    elif type1 == 'LOF':
        clf = LocalOutlierFactor(n_neighbors=20, novelty=True)
    clf.fit(X_train)

    faults = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    accum_percent = []

    for fault in faults:
        X_test = LoadData_pickle(path=dir + fault, name='_test' + type, type='rb')
        X_test = X_test.reshape((360, -1))
        # if fault == 'C0':
        #     X_test = X_test
        # else:
        #     X_test = X_test[:70, :]
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
    print('第'+ str(i+1) + '次Avg. Accuracy:', avg_accuracy)
    total_accum_percent.append(avg_accuracy)

# 输出
output_txt = './results/Accuracy/accuracy_'+type+'_'+type1+'.txt'
r1=np.array(total_accum_percent)*100
np.savetxt(output_txt,r1,fmt='%.3f')

r2=(np.array(total_accum_percent).mean())*100
print('Total Avg. Accuracy:', r2)
total = np.array(['Total Acc: {:.3f}'.format(r2)])
content_to_write = '\n'.join(total)
with open(output_txt, 'a') as file:
    file.write(content_to_write)