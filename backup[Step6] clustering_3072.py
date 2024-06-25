import tensorflow as tf
import numpy as np
import pickle
import nextbatch

from dataset_provider import provide_data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#指定在第0块GPU上跑
#visiualize
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sn.color_palette("hls", 8))

    # We create a scatter plot.
    f = plt.figure(figsize=(10, 10))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    # plt.grid(c='r')
    ax.axis('on')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(8):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

# def get_one_hot(labels, nb_classes):
#     res = np.eye(nb_classes)[np.array(labels).reshape(-1)]
#     return np.squeeze(res.reshape(list(labels.shape)+[nb_classes]))
# def sigmoid(x):
#     return 1/(1 + np.exp(-x))
#
# tf.reset_default_graph()
# best_model=False

# dir = './DATABACKUP/dcae_only/'
dir = './data/AUV/'
# dcae_only：仅使用DCAE对原始数据进行处理后的128维数据
# glow_enc：使用glow的encoder对原始数据进行处理后的3072维数据
# glow_enc_dcae：对glow_enc再使用DCAE进行处理后的128维数据

# 读取正常数据
path_train_0 = dir+'C0_train.pkl'
with open(path_train_0,'rb') as f0:
        x_train_0 = pickle.load(f0)
x_train_0 = x_train_0.reshape(700,128)
# print(x_train_0)
y0=np.ones([700])*0
x_train_0 ,y0= (x_train_0,y0)
# print(x_train_0,y0)

path_train_1 = dir+'C1_train.pkl'
with open(path_train_1,'rb') as f1:
        x_train_1 = pickle.load(f1)
x_train_1 = x_train_1.reshape(700,128)
y1=np.ones([700])*1
x_train_1, y1= (x_train_1, y1)
# print(x_train_1, y1)

path_train_2 = dir+'C2_train.pkl'
with open(path_train_2,'rb') as f2:
        x_train_2 = pickle.load(f2)
x_train_2 = x_train_2.reshape(700,128)
y2=np.ones([700])*2
x_train_2, y2= (x_train_2, y2)
# print(x_train_2, y2)

path_train_3 = dir+'C3_train.pkl'
with open(path_train_3,'rb') as f3:
        x_train_3 = pickle.load(f3)
x_train_3 = x_train_3.reshape(700,128)
y3=np.ones([700])*3
x_train_3, y3= (x_train_3, y3)
# print(x_train_3, y3)

path_train_4 = dir+'C4_train.pkl'
with open(path_train_4,'rb') as f4:
        x_train_4 = pickle.load(f4)
x_train_4 = x_train_4.reshape(700,128)
y4=np.ones([700])*4
x_train_4, y4= (x_train_4, y4)
# print(x_train_4, y4)

path_train_5 = dir+'C5_train.pkl'
with open(path_train_5,'rb') as f5:
        x_train_5 = pickle.load(f5)
x_train_5 = x_train_5.reshape(700,128)
y5=np.ones([700])*5
x_train_5, y5= (x_train_5, y5)
# print(x_train_5, y5)

path_train_6 = dir+'C6_train.pkl'
with open(path_train_6,'rb') as f6:
        x_train_6 = pickle.load(f6)
x_train_6 = x_train_6.reshape(700,128)
y6=np.ones([700])*6
x_train_6, y6= (x_train_6, y6)
print(x_train_6, y6)

path_train_7 = dir+'C7_train.pkl'
with open(path_train_7,'rb') as f7:
        x_train_7 = pickle.load(f7)
x_train_7 = x_train_7.reshape(700,128)
y6=np.ones([700])*6
x_train_7, y7= (x_train_7, y7)
print(x_train_7, y7)

# path_train_0 = dir+'x0_test.pkl'
# with open(path_train_0,'rb') as f0:
#         x_train_0 = pickle.load(f0)
# x_train_0 = x_train_0.reshape(300,128)
# # print(x_train_0)
# y0=np.ones([300])*0
# x_train_0 ,y0= (x_train_0,y0)
# # print(x_train_0,y0)
#
# path_train_1 = dir+'x1_test.pkl'
# with open(path_train_1,'rb') as f1:
#         x_train_1 = pickle.load(f1)
# x_train_1 = x_train_1.reshape(300,128)
# y1=np.ones([300])*1
# x_train_1, y1= (x_train_1, y1)
# # print(x_train_1, y1)
#
# path_train_2 = dir+'x2_test.pkl'
# with open(path_train_2,'rb') as f2:
#         x_train_2 = pickle.load(f2)
# x_train_2 = x_train_2.reshape(300,128)
# y2=np.ones([300])*2
# x_train_2, y2= (x_train_2, y2)
# # print(x_train_2, y2)
#
# path_train_3 = dir+'x3_test.pkl'
# with open(path_train_3,'rb') as f3:
#         x_train_3 = pickle.load(f3)
# x_train_3 = x_train_3.reshape(300,128)
# y3=np.ones([300])*3
# x_train_3, y3= (x_train_3, y3)
# # print(x_train_3, y3)
#
# path_train_4 = dir+'x4_test.pkl'
# with open(path_train_4,'rb') as f4:
#         x_train_4 = pickle.load(f4)
# x_train_4 = x_train_4.reshape(300,128)
# y4=np.ones([300])*4
# x_train_4, y4= (x_train_4, y4)
# # print(x_train_4, y4)
#
# path_train_5 = dir+'x5_test.pkl'
# with open(path_train_5,'rb') as f5:
#         x_train_5 = pickle.load(f5)
# x_train_5 = x_train_5.reshape(300,128)
# y5=np.ones([300])*5
# x_train_5, y5= (x_train_5, y5)
# # print(x_train_5, y5)
#
# path_train_6 = dir+'x6_test.pkl'
# with open(path_train_6,'rb') as f6:
#         x_train_6 = pickle.load(f6)
# x_train_6 = x_train_6.reshape(300,128)
# y6=np.ones([300])*6
# x_train_6, y6= (x_train_6, y6)
# # print(x_train_6, y6)


x_train = np.vstack([x_train_0, x_train_1, x_train_2, x_train_3, x_train_4, x_train_5, x_train_6, x_train_7])
y = np.hstack([y0, y1, y2, y3, y4, y5, y6, y7]).astype(np.int64)
# x_train = np.hstack([x_train_0,x_train_1])
# y = np.hstack([y0,y1])
#
print(x_train.shape, y.shape)

# That's an impressive list of imports.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sn
sn.set_style('whitegrid')
sn.set_palette('muted')
sn.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from sklearn.manifold import TSNE
digits_proj = TSNE().fit_transform(x_train)
scatter(digits_proj, y)
# plt.savefig('./fig/cluster_dcae.png')
plt.show()