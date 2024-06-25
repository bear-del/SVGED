import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
from sklearn.decomposition import KernelPCA,PCA
import numpy as np
from util import LoadData_pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from BaseSVDD import BaseSVDD
from sklearn import metrics
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile

# 假设你的振动信号存储在一个形状为 (360, 3072) 的二维数组中
# 这里使用随机数据作为示例
np.random.seed(42)
vibration_data = np.random.rand(360, 3072)

# 设置STFT的参数
nperseg = 256  # 窗口长度
noverlap = 128  # 重叠部分
fs = 1000  # 采样率，这里仅作为示例，根据实际数据设置

# 循环处理每个振动信号


# 将结果转换为NumPy数组


for class_num in range(8):
    data_train = LoadData_pickle(path='./data/AUV/', name=f'C{class_num}_train')
    stft_results_train = []
    for i in range(840):
        frequencies, times, Zxx = stft(data_train[i])
        stft_results_train.append(np.abs(Zxx))
    stft_results_train1 = np.array(stft_results_train)
    file_train = open('./data/AUV/STFT/C' + str(class_num) + '_train_STFT.pkl', 'wb')
    pickle.dump(stft_results_train1, file_train)
    file_train.close()

    data_test = LoadData_pickle(path='./data/AUV/', name=f'C{class_num}_test')
    stft_results_test = []
    for i in range(360):
        frequencies, times, Zxx_t = stft(data_test[i])
        stft_results_test.append(np.abs(Zxx_t))
    stft_results_test1 = np.array(stft_results_test)
    file_test = open('./data/AUV/STFT/C' + str(class_num) + '_test_STFT.pkl', 'wb')
    pickle.dump(stft_results_test1, file_test)
    file_test.close()

    #
    # data_test = LoadData_pickle(path='./data/AUV/', name=f'C{class_num}_test')
    # fft_results_test = np.fft.fft(data_test, axis=1)
    # file_test = open('./data/AUV/FFT/C' + str(class_num) + '_test_FFT.pkl', 'wb')
    # pickle.dump(fft_results_test, file_test)
    # file_test.close()

# data_train = LoadData_pickle(path='./data/AUV/', name='C1_train')
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 对每个振动信号进行FFT
# fft_results = np.fft.fft(data_train, axis=1)

# # 计算频率轴
# fs = 10000  # 采样率，这里仅作为示例，根据实际数据设置
# freqs = np.fft.fftfreq(3072, 1/fs)
#
# # 绘制第一个振动信号的FFT结果
# plt.plot(freqs, np.abs(fft_results[0]))
# plt.title('FFT Magnitude')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()