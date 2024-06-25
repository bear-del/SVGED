import numpy as np
from util import LoadData_pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from BaseSVDD import BaseSVDD
from sklearn import metrics

# dir = './model/dcae-C0aa/'
dir = './data/AUV/type7/'

X_train = LoadData_pickle(path=dir, name='C0_G_train', type='rb')
X_train = np.squeeze(X_train)

total_accum_precent = []
# clf = IsolationForest(n_estimators=500)
clf = OneClassSVM(nu=0.9, kernel='rbf', gamma='auto')
# clf = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on')
clf.fit(X_train)

faults = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
accum_percent = []
x_test = []

for fault in faults:
    X_test_1 = LoadData_pickle(path=dir + fault, name='_G_test', type='rb')
    x_test.extend(X_test_1)

x_test = np.squeeze(x_test)
y_test = np.concatenate([np.ones(360),-1*np.ones(2320),np.ones(200)])
y_score = clf.decision_function(x_test)
y_pred = clf.predict(x_test)


fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
# avg_accuracy = np.array(accum_percent).mean()
# print('Avg. Accuracy:', avg_accuracy)
# total_accum_precent.append(avg_accuracy)

# print('Avg. Accuracy:', np.array(total_accum_precent).mean())

