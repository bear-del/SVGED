from util import LoadData_pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle
import numpy as np

# data_1 = LoadData_pickle(path='./data/AUV/type8/', name='C0_train')
# data_2 = LoadData_pickle(path='./data/AUV/type8_label/', name='C0_train')

# data_T = LoadData_pickle(path='./data/AUV/',name='C0_train')
# scaler = preprocessing.StandardScaler()
# data_T = scaler.fit_transform(data_T)
#
# for class_num in range(8):
#     data_TN = LoadData_pickle(path='./data/AUV/', name=f'C{class_num}_train')
#     data_TN = scaler.transform(data_TN)
#     file_TN = open('./data/AUV/type8/C' + str(class_num) + '_train.pkl', 'wb')
#     pickle.dump(data_TN, file_TN)
#     file_TN.close()
#
#     data_tn = LoadData_pickle(path='./data/AUV/', name=f'C{class_num}_test')
#     data_tn = scaler.transform(data_tn)
#     file_tn = open('./data/AUV/type8/C' + str(class_num) + '_test.pkl', 'wb')
#     pickle.dump(data_tn, file_tn)
#     file_tn.close()
#
# data_2 = data_2.T
# plt.plot(data_2, color="orange")
#
# data_t = data_t.T
# plt.plot(data_t, color="steelblue")

# data_T = LoadData_pickle(path='./data/AUV/',name='C3_train')
# data_t = LoadData_pickle(path='./data/AUV/label/',name='C3_train')
# data_T = np.squeeze(data_T)
# data_t = np.squeeze(data_t)

np.random.seed(42)
data = 0.2 * np.random.randn(360, 2)
data = np.r_[data + 3, data]