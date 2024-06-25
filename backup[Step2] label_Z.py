import numpy as np
import pickle
from util import LoadData_pickle,save_data

path='./data/AUV/type9/'
label = 'aa'
for type_a in range(1):
    print(f'{type_a}')
    data_train = LoadData_pickle(path = path, name='C' + f'{type_a}_G_train'+label)
    data_test = LoadData_pickle(path = path, name='C' + f'{type_a}_G_test'+label)
    len_train = len(data_train)
    len_test = len(data_test)

    label_train = np.ones((len_train, 1)) * type_a
    label_test = np.ones((len_test, 1)) * type_a

    result_train = (data_train, label_train)
    file_train = open('./data/AUV/type9_label/C' + f'{type_a}' + '_train'+label+'.pkl', 'wb')
    pickle.dump(result_train, file_train)
    file_train.close()

    result_test = (data_test, label_test)
    file_test = open('./data/AUV/type9_label/C' + f'{type_a}' + '_test'+label+'.pkl', 'wb')
    pickle.dump(result_test, file_test)
    file_test.close()

    # file_train = open('./data/AUV/C' + f'{type_a}' + '_train.pkl', 'wb')
    # pickle.dump(X_train, file_train)
    # file_train.close()
    #
    # file_test = open('./data/AUV/C' + f'{type_a}' + '_test.pkl', 'wb')
    # pickle.dump(X_test, file_test)
    # file_test.close()




