import numpy as np
import pickle
from util import LoadData_pickle,save_data

path='./model/glow-ca/'
for type_a in range(8):
    print(f'{type_a}')
    data_train = LoadData_pickle(path = path, name='C' + f'{type_a}_trainca')
    data_test = LoadData_pickle(path = path, name='C' + f'{type_a}_testca')
    len_train = len(data_train)
    len_test = len(data_test)

    label_train = np.ones((len_train, 1)) * type_a
    label_test = np.ones((len_test, 1)) * type_a

    result_train = (data_train, label_train)
    file_train = open('./model/glow-ca-label/C' + f'{type_a}' + '_trainca.pkl', 'wb')
    pickle.dump(result_train, file_train)
    file_train.close()

    result_test = (data_test, label_test)
    file_test = open('./model/glow-ca-label/C' + f'{type_a}' + '_testca.pkl', 'wb')
    pickle.dump(result_test, file_test)
    file_test.close()

    # file_train = open('./data/AUV/C' + f'{type_a}' + '_train.pkl', 'wb')
    # pickle.dump(X_train, file_train)
    # file_train.close()
    #
    # file_test = open('./data/AUV/C' + f'{type_a}' + '_test.pkl', 'wb')
    # pickle.dump(X_test, file_test)
    # file_test.close()




