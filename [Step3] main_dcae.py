import tensorflow as tf
from data_provider import *
import matplotlib.pyplot as plt
import nextbatch
import os
import math
import numpy as np
import pickle
#from sklearn import preprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "0"#指定在第0块GPU上跑

# 定义KL散度
def kl_divergence(p, p_hat):
    return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)

# root = './data/AUV/label/'
root = './model/glow-ca-label/'
label_in = 'ca'
label_out = 'fc'
faults= ['C0']

for fault in faults :
    data_dir = root+fault+'_train'+label_in+'.pkl'
    with open(data_dir, 'rb') as f:
        datasets,labels= pickle.load(f)
    dataset_train= nextbatch.DataProvider(datasets, labels) #绑定

    test_dir = root+fault+'_test'+label_in+'.pkl'
    with open(test_dir, 'rb') as f_1:
        datasets_2,labels_2= pickle.load(f_1)
    dataset_test= nextbatch.DataProvider(datasets_2, labels_2) #绑定

# 全局设置参数
learning_rate = 1e-5
epochs = 2001
batch_size = 16

# tensorflow graph
# 1,占位符
inputs_ = tf.placeholder(tf.float32, [None, 3072, 1], name='inputs')

# 2，神经网络
layers=tf.keras.layers
### Encoder
# conv1 = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(inputs_)
# maxpool1 = layers.MaxPooling1D(pool_size=2, strides=4, padding='same')(conv1)
# conv2 = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool1)
# maxpool2 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv2)
# conv3 = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool2)
# maxpool3 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv3)
# conv4 = layers.Conv1D(filters=2, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool3)
# maxpool4 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv4)
#
# re=tf.reshape(maxpool4,[-1,192])
#
# latent=layers.Dense(units=128,activation=tf.nn.relu)(re)
# # ---Decoder---#
# x=layers.Dense(units=192,activation=tf.nn.relu)(re)
# x=tf.reshape(x,[-1,96,2])
# x=layers.UpSampling1D(2)(x)
# x=layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
# x=layers.UpSampling1D(2)(x)
# x=layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
# x=layers.UpSampling1D(2)(x)
# x=layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
# x=layers.UpSampling1D(4)(x)
# rx=layers.Conv1D(filters=1, kernel_size=5, padding='same', activation=tf.nn.relu)(x)

conv1 = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(inputs_)
maxpool1 = layers.MaxPooling1D(pool_size=2, strides=4, padding='same')(conv1)
conv2 = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool1)
maxpool2 = layers.MaxPooling1D(pool_size=2, strides=4, padding='same')(conv2)
conv3 = layers.Conv1D(filters=2, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool2)
maxpool3 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv3)

re=tf.reshape(maxpool3,[-1,192])

latent=layers.Dense(units=128,activation=tf.nn.relu)(re)

x=layers.Dense(units=192,activation=tf.nn.relu)(re)
x=tf.reshape(x,[-1,96,2])
x=layers.UpSampling1D(2)(x)
x=layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
x=layers.UpSampling1D(4)(x)
x=layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
x=layers.UpSampling1D(4)(x)
rx=layers.Conv1D(filters=1, kernel_size=5, padding='same', activation=tf.nn.relu)(x)

# 3， 损失loss
diff = rx - inputs_
loss= tf.reduce_mean(tf.reduce_sum(diff**2,axis=1))

# 4， 优化及minimize
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate ,beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)

# 5，  summary
tf.summary.scalar("loss", loss)
logs_path = './logs'

# 6, Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
#
# 储存模型
saver = tf.train.Saver(max_to_keep=30)
ckpt_path = './model/dcae-' + fault + label_out + '/model.ckpt'

# session 初始化
init_op = tf.global_variables_initializer()

# 运行session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

   af=[]
   total_batch = int(len(dataset_train.images) / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = dataset_train.next_batch(batch_size=batch_size)
            batch_x = batch_x.reshape(batch_size,3072,1)
            # x = np.reshape(None, 3072)
            _, c = sess.run([optimiser, loss],feed_dict={inputs_: batch_x})

            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        af.append(avg_cost)
        if epoch % 20 == 0:
            c = loss.eval(feed_dict={inputs_: batch_x})
            print('epoch {}, training loss {}'.format(epoch, c))

        if epoch % 100 == 0:
            # write log
            save_path = saver.save(sess, ckpt_path, global_step=epoch)
            print('checkpoint saved in %s' % save_path)
   print('Optimization Finished')
   print('Cost:', loss.eval({inputs_ : dataset_test.images.reshape(-1,3072,1) }))

np.savetxt('./model/dcae-' + fault + label_out +'/dcae_loss_'+fault+'.txt',af)
plt.plot(af, label='train loss')
plt.savefig('./model/dcae-' + fault + label_out +'/loss_dcae.png')
plt.show()
