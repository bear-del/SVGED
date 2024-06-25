import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import adam
from keras.callbacks import Callback, TensorBoard
from keras.callbacks.callbacks import ModelCheckpoint
from flow_layers import *
from datetime import datetime
#from sklearn import preprocessing
from util import LoadData_pickle
from NN_architecture import build_basic_model

# 加载训练数据
class_num = [0,1,2,3,4,5,6]
data_T, _ = LoadData_pickle(path='./data/data/',name='x0_train',type='rb')
data_t, _ = LoadData_pickle(path='./data/data/',name='x0_test',type='rb')
# data_T = np.squeeze(data_T)
# data_t = np.squeeze(data_t)

# 加载生成时用的数据
# 这里当时改得比较急写得不好，这个循环好像也没啥意义，但是反正功能是正常的
loaded_files_train = {}
for i in range(7):
    file_name = './data/data/'+f"x{i}_train.pkl"
    variable_name_train = f"x{i}"
    with open(file_name, "rb") as file:
        file1, _ = pickle.load(file)
        # file1 = np.squeeze(file1)
        loaded_files_train[variable_name_train] = file1

data_T0 = loaded_files_train["x0"]
data_T1 = loaded_files_train["x1"]
data_T2 = loaded_files_train["x2"]
data_T3 = loaded_files_train["x3"]
data_T4 = loaded_files_train["x4"]
data_T5 = loaded_files_train["x5"]
data_T6 = loaded_files_train["x6"]

loaded_files_test = {}
for i in range(7):
    file_name = './data/data6/'+f"x{i}_test.pkl"
    variable_name_test = f"x{i}"
    with open(file_name, "rb") as file:
        file1, _ = pickle.load(file)
        # file1 = np.squeeze(file1)
        loaded_files_test[variable_name_test] = file1

data_T0t = loaded_files_test["x0"]
data_T1t = loaded_files_test["x1"]
data_T2t = loaded_files_test["x2"]
data_T3t = loaded_files_test["x3"]
data_T4t = loaded_files_test["x4"]
data_T5t = loaded_files_test["x5"]
data_T6t = loaded_files_test["x6"]

# # for i in class_num:
# #     num_now = f"data_T{i}"
# #     data_Ti = globals()[num_now]
# #     x_encoded = encoder.predict(data_Ti, verbose=1)
# #     generated_examples = np.array(x_encoded)
# Preprocessing
# scaler = preprocessing.StandardScaler()
# data_T_scaled = scaler.fit_transform(data_T)
# data_t_scaled = scaler.transform(data_t)
#
# transformer = preprocessing.Normalizer()
# data_T = transformer.fit_transform(data_T_scaled)
# data_t = transformer.transform(data_t_scaled)

original_dim = 3072
depth = 8  # orginal paper use depth=32
level = 3  # orginal paper use level=6 for 256*256 CelebA HQ
batch_size = 32
epochs = 1000

x_in = Input(shape=(original_dim,))
# 给输入加入噪声（add noise into inputs for stability.）
x_noise = Lambda(lambda s: K.in_train_phase(s + 0.01 * K.random_uniform(K.shape(s)), s))(x_in)
x = Reshape((-1,1, original_dim, 1))(x_noise)
x_outs = []

squeeze = Squeeze()
inner_layers = []
outer_layers = []
for i in range(5):
    inner_layers.append([])
for i in range(3):
    outer_layers.append([])

for i in range(level):  #level = 3
    x = squeeze(x)    # shape=[h, w, c] ==> shape=[h/n, w/n, n*n*c] n=2  ori->[32,32,12] now->[768,1,4]
    for j in range(depth):  #depth = 10
        actnorm = Actnorm()
        permute = Permute(mode='random')
        # permute = Permute(mode='reverse')
        # permute = InvDense()
        split = Split()
        couple = CoupleWrapper(build_basic_model(2**(i+1)))
        # couple = CoupleWrapper(build_basic_model(3*2**(i+1)))
        concat = Concat()
        inner_layers[0].append(actnorm)
        inner_layers[1].append(permute)
        inner_layers[2].append(split)
        inner_layers[3].append(couple)
        inner_layers[4].append(concat)
        x = actnorm(x)   #缩放平移变换层（Scale and shift）
        x = permute(x)
        x1, x2 = split(x)  #将输入分区沿着最后一个轴切分为2部分 -> [(768,1,2), (768,1,2)]
        x1, x2 = couple([x1, x2])   #仿射耦合层 return[(768,1,2),(768,1,2)]
        x = concat([x1, x2])
    if i < level - 1:
        split = Split()
        condactnorm = CondActnorm()
        reshape = Reshape()
        outer_layers[0].append(split)
        outer_layers[1].append(condactnorm)
        outer_layers[2].append(reshape)
        x1, x2 = split(x)
        x_out = condactnorm([x2, x1])
        x_out = reshape(x_out)   #压平，不影响第一个维度
        x_outs.append(x_out)
        x = x1
    else:
        for _ in outer_layers:
            _.append(None)

final_actnorm = Actnorm()
final_concat = Concat()
final_reshape = Reshape()

x = final_actnorm(x)
x = final_reshape(x)
x = final_concat(x_outs+[x])

# 使用 Functional API 构建模型
# inputs = XXX; outputs = XXX; model = keras.Model(inputs=inputs, outputs=outputs)
encoder = Model(x_in, x)
for l in encoder.layers:
    if hasattr(l, 'logdet'):
        encoder.add_loss(l.logdet)

encoder.summary()
# 编译模型：模型构建完成后，调用 compile() 方法来配置模型优化器、损失函数和评估指标等。
# metrics: 在训练和测试期间的模型评估标准。 通常会使用 metrics = ['accuracy']
encoder.compile(loss=lambda y_true, y_pred: 0.5 * K.sum(y_pred ** 2, 1) + 0.5 * np.log(2 * np.pi) * K.int_shape(y_pred)[1],
            optimizer=adam(1e-4))

# 搭建逆模型（生成模型），将所有操作倒过来执行
# 这里不使用decoder
# x_in = Input(shape=K.int_shape(encoder.outputs[0])[1:])
# # x_in = Input(shape=(original_dim,))
# x = x_in
#
# x = final_concat.inverse()(x)
# outputs = x[:-1]
# x = x[-1]
# x = final_reshape.inverse()(x)
# x = final_actnorm.inverse()(x)
# x1 = x
#
# for i,(split,condactnorm,reshape) in enumerate(list(zip(*outer_layers))[::-1]):
#     if i > 0:
#         x1 = x
#         x_out = outputs[-i]
#         x_out = reshape.inverse()(x_out)
#         x2 = condactnorm.inverse()([x_out, x1])
#         x = split.inverse()([x1, x2])
#     for j,(actnorm,permute,split,couple,concat) in enumerate(list(zip(*inner_layers))[::-1][i*depth: (i+1)*depth]):
#         x1, x2 = concat.inverse()(x)
#         x1, x2 = couple.inverse()([x1, x2])
#         x = split.inverse()([x1, x2])
#         x = permute.inverse()(x)
#         x = actnorm.inverse()(x)
#     x = squeeze.inverse()(x)
# x = Reshape(shape=(-1,original_dim))(x)
#
# decoder = Model(x_in, x)
# decoder.summary()

class generate_callback(Callback):
    def __init__(self):
        super().__init__()
        # self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 499 or epoch == epochs - 1:
            for i in class_num:
                num_now = f"data_T{i}"
                data_Ti = globals()[num_now]
                x_encoded = encoder.predict(data_Ti, verbose=1)
                generated_examples = np.array(x_encoded)
                gentime = datetime.strftime(datetime.now(), '%m%d %H%M%S')
                print(generated_examples.shape)
                file = open('./data/generated/C' + str(i) + '_G_' + gentime + '_train.pkl', 'wb')
                pickle.dump(generated_examples, file)
                file.close()
                num_now_test = f"data_T{i}t"
                data_Ti_test = globals()[num_now_test]
                x_encoded_test = encoder.predict(data_Ti_test, verbose=1)
                generated_examples_test = np.array(x_encoded_test)
                print(generated_examples_test.shape)
                file_test = open('./data/generated/C' + str(i) + '_G_' + gentime + '_test.pkl', 'wb')
                pickle.dump(generated_examples_test, file_test)
                file.close()

            # for
            # # count_examples = 0
            # # num_examples_to_eval = 1000
            # gen_list = []
            # # x1=laod(path)# train :700*128/test:300*128
            # x1 = LoadData_pickle(path='./data/DCAE/', name='C1_train', type='rb')
            # x1_generate=decoder.predict(x1_train)# 700*128/300*128
            # x1_generated=decoder.predict(x1_test)
            # # while count_examples < num_examples_to_eval:
            #
            #     # x_decoded = decoder.predict(np.random.randn(batch_size, 128), verbose=1)
            #     # gen_list.extend(x_decoded)
            #     # count_examples += batch_size
            # # generated_examples = np.array(gen_list)
            # # gentime = datetime.strftime(datetime.now(), '%m%d %H%M%S')
            # # print(generated_examples.shape)
            # file = open('./data/generated/C' + class_num + '_G_' + gentime + '.pkl', 'wb')
            # pickle.dump(generated_examples, file)
            # file.close()

generate_callback = generate_callback()

checkpoint_path = './data/checkpoint/weights.{epoch:05d}-{loss:.2f}.h5'
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='loss', verbose=1,
                                      save_weights_only=True, save_best_only=True, period=10)

# tensorboard_callback = TensorBoard(log_dir='logs', update_freq='epoch', histogram_freq=1,
#                                    write_graph=True)

# history = encoder.fit(data_T, data_T, batch_size=batch_size, epochs=epochs, validation_data=(data_t, data_t), verbose=1,
#                       callbacks=[generate_callback, tensorboard_callback, checkpoint_callback])

history = encoder.fit(data_T, data_T, batch_size=batch_size, epochs=epochs, validation_data=(data_t, data_t), verbose=1,
                      callbacks=[generate_callback, checkpoint_callback])

# summarize history for loss
gentime1 = datetime.strftime(datetime.now(), '%m%d %H%M%S')
plt.figure(2)
plt.plot()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.savefig('./data/loss_img/C'+class_num+'_Generated_'+gentime1+'.png')
plt.savefig('./data/loss_img/C0_Generated_'+gentime1+'.png')
plt.show()
