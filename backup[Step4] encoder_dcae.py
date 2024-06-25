import tensorflow as tf
import numpy as np
import os
from data_provider import read_data_sets
import pickle
import math
import nextbatch

def kl_divergence(p, p_hat):
    return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(150)
np.random.seed(150)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 全局设置参数
learning_rate = 1e-4
# epochs = 1000
# batch_size = 2

root_model = './model/dcae-C0_nopre'

faults = ['C0','C1','C2','C3','C4','C5','C6','C7']
# faults=['x0']

for fault in faults:

    path_out = './results/'+fault+'_train.pkl'
    path_1_out = './results/' + fault + '_test.pkl'
    train_encoded = []
    test_encoded = []

    root = './data/AUV/label/'
    best_log_dir = root_model

    # root_data = '../results/dcae/' + fault + '_train.pkl'
    # root_test = '../results/dcae/' + fault + '_test.pkl'

    data_dir = root + fault + '_train.pkl'
    with open(data_dir, 'rb') as f:
        datasets, labels = pickle.load(f)
    dataset_train = nextbatch.DataProvider(datasets, labels)  # 绑定

    test_dir = root + fault + '_test.pkl'
    with open(test_dir, 'rb') as f_1:
        datasets_2, labels_2 = pickle.load(f_1)
    dataset_test = nextbatch.DataProvider(datasets_2, labels_2)  # 绑定

    # Build graph
    tf.reset_default_graph()
    # tensorflow graph
    # 1,占位符
    inputs_ = tf.placeholder(tf.float32, [None, 3072, 1], name='inputs')
    # 2，神经网络
    layers = tf.keras.layers
    # ### Encoder
    conv1 = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)(inputs_)
    maxpool1 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv1)
    conv2 = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool1)
    maxpool2 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv2)
    conv3 = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool2)
    maxpool3 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv3)
    conv4 = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool3)
    maxpool4 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv4)
    conv5 = layers.Conv1D(filters=2, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool4)
    maxpool5 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv5)
    re = tf.reshape(maxpool5, [-1, 192])
    # -----------#
    latent = layers.Dense(units=128, activation=tf.nn.relu)(re)
    # -----------#
    # ---Decoder---#
    x = layers.Dense(units=192, activation=tf.nn.relu)(re)
    x = tf.reshape(x, [-1, 96, 2])
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    rx = layers.Conv1D(filters=1, kernel_size=5, padding='same', activation=tf.nn.relu)(x)

    # 3， 损失loss
    diff = rx - inputs_
    loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))

    # 4， 优化及minimize
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(
        loss)

    # Evaluation
    var_dict = {x.op.name: x for x in
                tf.contrib.framework.get_variables('encoder/')
                if 'Adam' not in x.name}

    l_cp = tf.train.latest_checkpoint(best_log_dir)
    print('latest checkpoint:',l_cp)
#
    tf.contrib.framework.init_from_checkpoint(
        l_cp, var_dict)
# #
#
        # 映射数据
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_encoded.append(sess.run(latent,
                                      feed_dict={inputs_: dataset_train.images.reshape(-1,3072,1)}))
        test_encoded.append(sess.run(latent,
                                     feed_dict={inputs_: dataset_test.images.reshape(-1,3072,1)}))

    with open(path_out, 'wb') as f:
        pickle.dump(np.array(train_encoded), f, pickle.HIGHEST_PROTOCOL)
    with open(path_1_out, 'wb') as f:
        pickle.dump(np.array(test_encoded), f, pickle.HIGHEST_PROTOCOL)