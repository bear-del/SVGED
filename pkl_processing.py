import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from util import LoadData_pickle

for type_a in range(8):
    print(f'{type_a}')
    file1 = LoadData_pickle(path='./data/AUV/type1/',name='C' + f'{type_a}_1')
    file2 = LoadData_pickle(path='./data/AUV/type1/',name='C' + f'{type_a}_2')
    file3 = LoadData_pickle(path='./data/AUV/type1/',name='C' + f'{type_a}_3')

    file = np.concatenate((file1, file2, file3), axis=0)
    file = file[:1200]
    file32 = file.astype(np.float32)


    X_train, X_test = train_test_split(file32, test_size=0.3, random_state=42)

    file_train = open('./data/AUV/C' + f'{type_a}' + '_train.pkl', 'wb')
    pickle.dump(X_train, file_train)
    file_train.close()

    file_test = open('./data/AUV/C' + f'{type_a}' + '_test.pkl', 'wb')
    pickle.dump(X_test, file_test)
    file_test.close()
