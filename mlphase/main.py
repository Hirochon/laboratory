# coding: utf-8
# import subprocess

import sys
import os
import numpy as np
import json
import pickle
from datetime import datetime
# import itertools
import matplotlib.pyplot as plt

from making_2D_image_20191028 import load_data
# from making_2D_image_20191028 import surf_figure, reduce_pkl_data

# from common.simple_convnet import SimpleConvNet
# from common.trainer import Trainer
# from dataset.mnist import load_mnist
# import matplotlib as mpl

from keras.utils import np_utils, plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
# from keras.datasets import mnist
# from keras.layers import Activation, Dropout, Reshape, SeparableConv2D
# from keras.callbacks import EarlyStopping

from file_operation import MAKE_DIR, COPY_FILES


sys.path.append(os.pardir)


def make_model_deeplearning(input_dim, units=[50], last_unit=5):
    # units = [unit,unit,unit]
    
    model = Sequential()
    for i, num_unit in enumerate(units):
        if i == 0:
            model.add(Dense(num_unit, input_shape=(input_dim,), activation="relu"))
        else:
            model.add(Dense(num_unit, activation="relu"))
    
    model.add(Dense(units=last_unit, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def make_model_CNN(input_dim, units=[50], last_unit=5):
    # units = [unit,unit,unit]
    
    model = Sequential()
    for i, num_unit in enumerate(units):
        if i == 0:
            model.add(Dense(num_unit, input_shape=(input_dim,), activation="relu"))
        else:
            model.add(Dense(num_unit, activation="relu"))
    
    model.add(Dense(units=last_unit, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def make_figs(x, t, infile, sfolder):

    temp1 = np.random.randint(0, np.shape(x)[0], size=16)

    print("ch", np.shape(x), np.shape(x)[1])
    for j in range(np.shape(x)[1]):
        fig = plt.figure()
        for i, dat in enumerate(x[temp1]):
            ax = fig.add_subplot(4, 4, i + 1)
            ax.set_title(t[temp1[i]])
            ax.set_xticks([], minor=False)
            ax.set_yticks([], minor=False)
            ax.imshow(dat[j, :, :])
            # print(dat[j, :, :])

        file = sfolder + "/" + os.path.basename(infile).split(".")[0] + "_" + str(j) + "_figs.png"
        print(file)
        plt.savefig(file)
        plt.close()


def make_figs_every_label(x, t, infile, sfolder):
    num_classes = np.max(t) + 1
    # temp1 = np.random.randint(0,np.shape(x)[0],size=16)
    temp1 = []
    for j in range(num_classes):
        target_mode = j
        ind = np.where(t == target_mode)
        for i in range(5):
            temp1.append(ind[0][i])
    temp1 = np.array(temp1)

    # print(ind)
    # print(t[temp1])
    # for i in temp1:
    #     fig = plt.figure()
    #     for j in [0,1]:
    #         ax = fig.add_subplot(1,2,j+1)
    #         dat = x[i]
    #         ax.imshow(dat[j,:,:])
    #         ax.set_title(str(t[i]))
    #     plt.show()

    for j in range(np.shape(x)[1]):
        fig = plt.figure()
        for i, dat in enumerate(x[temp1]):
            ax = fig.add_subplot(num_classes, 5, i + 1)
            ax.set_title(t[temp1[i]])
            ax.set_xticks([], minor=False)
            ax.set_yticks([], minor=False)
            ax.imshow(dat[j, :, :])
            # print(dat[j, :, :])

        file = sfolder + "/" + os.path.basename(infile).split(".")[0] + "_" + str(j) + "_figs.png"
        print(file)
        plt.savefig(file)
        plt.close()


def make_figs_every_label_every_ch(x, t, infile, sfolder):
    num_classes = np.max(t) + 1
    # temp1 = np.random.randint(0,np.shape(x)[0],size=16)
    temp1 = []
    for j in range(num_classes):
        target_mode = j
        ind = np.where(t == target_mode)
        temp1.append(ind[0][0])
    temp1 = np.array(temp1)
    fig = plt.figure()
    for i, dat in enumerate(x[temp1]):
        ax = fig.add_subplot(num_classes, 2, 2 * (i + 1))
        # ax.set_title(t[temp1[i]])
        ax.set_xticks([], minor=False)
        ax.set_yticks([], minor=False)
        ax.imshow(dat[0, :, :])
        ax = fig.add_subplot(num_classes, 2, 2 * (i + 2))
        # ax.set_title(t[temp1[i]])
        ax.set_xticks([], minor=False)
        ax.set_yticks([], minor=False)
        ax.imshow(dat[1, :, :])
    file = sfolder + "/" + os.path.basename(infile).split(".")[0] + "_every_mode_figs.png"
    print(file)
    plt.savefig(file)
    plt.close()


def make_fig_one_pair_abs(x, sfolder, text=""):
    temp1 = np.random.randint(0, np.shape(x[0])[0], size=4)

    n = len(x)
    print(np.shape(x[0]))
    fig = plt.figure()
    for i, j in enumerate(temp1):
        for k in range(n):
            ax = fig.add_subplot(4, n, (n * i) + k + 1)
            # ax.set_title("in")
            ax.set_xticks([], minor=False)
            ax.set_yticks([], minor=False)
            ax.imshow(np.abs(x[k][j, 0, :, :]))

    file = sfolder + "/" + "in_out_figs_" + text + "_".join(str(temp1)) + ".png"
    print(file)
    plt.savefig(file)
    plt.close()


def main_learn(file_train, file_test, sfolder, num_data):
    (x_train, t_train) = load_data(file_train, num_data=num_data)
    (x_test, t_test) = load_data(file_test, num_data=100)
    t_test_label = t_test
    # x_train = np.abs(x_train)
    # x_test  = np.abs(x_test)

    print(x_train[0, 0, 0, 0])
    print("num_data = ", len(x_train), np.shape(x_train))

    # x_train = reshape_complex_to_real_imag(x_train ,axis = 1)
    # x_test  = reshape_complex_to_real_imag(x_test  ,axis = 1)

    # print(np.angle(x_train))

    x_train = reshape_complex_to_abs_phase(x_train, axis=1)
    x_test = reshape_complex_to_abs_phase(x_test, axis=1)
    print(x_train[0, 0, 0, 0])
    print(x_train[0, 1, 0, 0])

    print(t_test)
    print(np.where(t_test == 2))

    # input_dim = np.shape(x_train)[2] * np.shape(x_train)[3]
    # input_dim = ( np.shape(x_train)[2], np.shape(x_train)[3])
    # print(input_dim, np.shape(x_train) )
    # exit()

    # # mnist test
    # (x_train, t_train), (x_test, t_test) = mnist.load_data()
    # x_train = x_train.astype('float32')   # int型をfloat32型に変換
    # x_test = x_test.astype('float32')
    # x_train /= 255                        # [0-255]の値を[0.0-1.0]に変換
    # x_test /= 255
    # x_train  = x_train.reshape(x_train.shape[0],-1, x_train.shape[1],x_train.shape[2])
    # x_test   = x_test.reshape(x_test.shape[0],-1,x_test.shape[1],x_test.shape[2])
    # t_test_label = t_test
    # print( np.shape(x_test), np.shape(t_test) )

    # make_figs(x_train,t_train,file_train,sfolder)
    make_figs_every_label(x_train, t_train, file_train, sfolder)
    make_figs_every_label_every_ch(x_train, t_train, file_train, sfolder)
    make_figs_every_label_every_ch(x_test, t_test, file_test, sfolder)

    exit()
    # for 1D input
    # x_train  = x_train.reshape(-1, input_dim)
    # x_test = x_test.reshape(-1, input_dim)

    num_classes = np.max(t_train) + 1

    t_train = np_utils.to_categorical(t_train, num_classes)
    t_test = np_utils.to_categorical(t_test, num_classes)
    
    units = [50, 50]
  
    model = Sequential()
    
    model.add(Conv2D(50, (6, 6), activation='relu', padding="same", data_format="channels_first"))
    # model.add(SeparableConv2D(50, (6, 6), activation='relu', padding="same", data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(50, (4, 4), activation='relu', padding="same", data_format="channels_first"))
    # model.add(SeparableConv2D(50, (4, 4), activation='relu', padding="same", data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    csv_logger = CSVLogger(sfolder + "/" + os.path.basename(file_train).split(".")[0] + "_training.csv", separator=',')
    epochs = 100
    history = model.fit(x=x_train,
                        y=t_train,
                        epochs=epochs,
                        batch_size=100,
                        validation_split=0.2,
                        callbacks=[csv_logger]
                        )

    histroy_fig(history, file_train, sfolder)
    save_history(history, file_train, sfolder)
    save_model(model, file_train, sfolder)

    score = model.evaluate(x=x_test, y=t_test)
    print('test_loss:', score[0])
    print('test_acc:', score[1])

    conv_layers = []
    for i, layer in enumerate(model.get_config()["layers"]):
        if "Conv" in layer["class_name"] or "Pooling" in layer["class_name"]:
            print(i, layer["class_name"])
            conv_layers.append(model.layers[i].output)
    # conv_layers = [l.output for l in model.layers[:4]]
    conv_model = Model(inputs=model.inputs, outputs=conv_layers)
    conv_outputs = conv_model.predict(x_test)
    
    for j in range(num_classes):
        target_mode = j
        ind = np.where(t_test_label == target_mode)
        n = ind[0][0]
        for i in range(len(conv_outputs)):
            plot_conv_outputs(conv_outputs[i][n],
                              file_test, sfolder,
                              "mode" + str(j) + "filter" + str(i))

    # save memo
    file = sfolder + "/" + os.path.basename(file_train).split(".")[0] + "_result_memo.json"
    print(file)
    with open(file, "w") as f:
        memo = {"sfolder": str(sfolder),
                "file_train": str(file_train),
                "num_data": str(num_data),
                "num_classes": str(num_classes),
                "num_unit": str(units),
                "epochs": str(epochs),
                "test_loss": str(score[0]),
                "test_acc": str(score[1])
                }
        json.dump(memo, f, indent=4)

        
def save_model(model, file_train, sfolder):
    file = sfolder + "/" + os.path.basename(file_train).split(".")[0] + "_model.png"
    print("save_file:", file)
    plot_model(model, to_file=file, show_shapes=True)
    # model.summary()


def save_history(history, file_train, sfolder):
    file = sfolder + "/" + os.path.basename(file_train).split(".")[0] + "_history.pkl"
    print("save_file:", file)
    with open(file, 'wb') as f:
        pickle.dump(history, f)


def histroy_fig(history, file_train, sfolder):
    epochs = range(len(history.history['acc']))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['acc'], label='training')
    plt.plot(epochs, history.history['val_acc'], label='validation')
    plt.title('acc')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='training')
    plt.plot(epochs, history.history['val_loss'], label='validation')
    plt.title('loss')
    plt.legend()
    plt.savefig(sfolder + "/" + os.path.basename(file_train).split(".")[0] + '.png')


def save_components_fit(yt, y_pred, sfolder):
    last = min([len(yt), 50])
    ind = (np.random.permutation(len(yt))[:last])
    plt.figure(figsize=(10, 5))
    plt.plot(yt[0][ind], label="y_test")
    plt.plot(y_pred[0][ind], "o-", label="y_predicted")
    plt.xlabel("components")
    plt.ylabel("value")
    plt.savefig(sfolder + "/components_fit.png")
    plt.legend()
    plt.cla()


def save_components_fit_9(yt, y_pred, sfolder):
    print("fit_9", np.shape(yt), len(yt))

    # last = min([len(yt),50])
    last = 50
    ind = (np.random.permutation(np.shape(yt)[1])[:last])
    fig = plt.figure(figsize=(10, 5))
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.set_ylim(-1, 1)
        ax.plot(yt[i][ind], label="y_test")
        ax.plot(y_pred[i][ind], "o-", label="y_predicted")
        if i + 1 in [1, 4, 7]:
            ax.set_xlabel("components")
        if i + 1 in [7, 8, 9]:
            ax.set_ylabel("value")
    plt.savefig(sfolder + "/components_fit_9.png")
    plt.legend()
    plt.cla()


def plot_conv_outputs(outputs, file, sfolder, text=""):
    filters = outputs.shape[0]
    # print(outputs.shape)
    for i in range(filters):
        plt.subplot(filters / 10 + 1, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        # plt.xlabel(f'filter {i}')
        plt.imshow(outputs[i, :, :])
    file = sfolder + "/" + os.path.basename(file).split(".")[0] + "_filter_output_" + str(text) + '.png'
    print(file)
    plt.savefig(file)


def plot_conv_weights(filters):
    filter_num = filters.shape[3]
    
    for i in range(filter_num):
        plt.subplot(filter_num / 10 + 1, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        # plt.xlabel(f'filter {i}')
        plt.imshow(filters[:, :, 0, i])


def main_regression_learn(file_x, file_y, file_test_x, file_test_y, sfolder, num_data=0):
    x, y = load_data(file_x, file_y, num_data=num_data)
    xt, yt = load_data(file_test_x, file_test_y, num_data=100)
    print(len(x), len(yt), np.shape(x), np.shape(yt))
    # print(type( x[0,0,0,0] ) )
    
    for i in range(10):
        make_fig_one_pair_abs((x, y), sfolder)
    
    # if ("complex" in str(type(x[0,0,0,0]))):
    # x = reshape_complex_to_abs_phase(x, axis=1)
    x = np.abs(x)
    dim = np.shape(x)[2] * np.shape(x)[3]
    x = x.reshape(-1, dim)

    y = np.abs(y)
    dim = np.shape(y)[2] * np.shape(y)[3]
    y = y.reshape(-1, dim)

    xt = np.abs(xt)
    xdim = np.shape(xt)[2]
    dim = np.shape(xt)[2] * np.shape(xt)[3]
    xt = xt.reshape(-1, dim)

    yt = np.abs(yt)
    ydim = np.shape(yt)[2]
    dim = np.shape(yt)[2] * np.shape(yt)[3]
    yt = yt.reshape(-1, dim)

    units = [50, 50, 50]
    epochs = 100
    err, score, y_pred = regressin_set(x, y, xt, yt, num_data, units, sfolder, epochs=epochs)

    xt = xt.reshape(-1, 1, xdim, xdim)
    y_pred_2d = y_pred.reshape(-1, 1, ydim, ydim)
    yt = yt.reshape(-1, 1, ydim, ydim)
    for i in range(10):
        make_fig_one_pair_abs((xt, yt, y_pred_2d), sfolder, "pred")
    
    file = sfolder + "/output.json"
    with open(file, "w") as f:
        memo = {"sfolder": str(sfolder),
                "file_train_x": str(file_x),
                "file_train_y": str(file_y),
                "file_test_x": str(file_test_x),
                "file_test_y": str(file_test_y),
                "num_data": str(num_data),
                "num_unit": str(units),
                "epochs": str(epochs),
                "err": str(err),
                "test_loss": str(score[0]),
                "test_acc": str(score[1])
                }
        json.dump(memo, f, indent=4)


def main_regression_CNN_learn(file_x, file_y, file_test_x, file_test_y, sfolder, num_data=0,
                              active_function="relu", reshape_type="abs_phase", num_conv_node=50):
    x, y, _ = load_data(file_x, file_y, num_data=num_data)
    xt, yt, _ = load_data(file_test_x, file_test_y, num_data=100)

    for s in [x, y, xt, yt]:
        print(np.shape(s))

    x = reshape_complex(x, mode=reshape_type, normalize=True)
    y = reshape_complex(y, mode=reshape_type, normalize=True)
    xt = reshape_complex(xt, mode=reshape_type, normalize=True)
    yt = reshape_complex(yt, mode=reshape_type, normalize=True)

    for s in [x, y, xt, yt]:
        print(np.shape(s))

    units = [50, 50, 50]
    epochs = 100
    err, score, y_pred = regressin_CNN(x, y, xt, yt, num_data, units, sfolder,
                                       epochs=epochs,
                                       active_function=active_function,
                                       num_conv_node=num_conv_node)

    y_pred_2d = y_pred.reshape(-1, 1, np.shape(yt)[2], np.shape(yt)[3])

    for i in range(10):
        make_fig_one_pair_abs((xt, yt, y_pred_2d), sfolder, "pred")
    
    file = sfolder + "/output.json"
    with open(file, "w") as f:
        memo = {"sfolder": str(sfolder),
                "file_train_x": str(file_x),
                "file_train_y": str(file_y),
                "file_test_x": str(file_test_x),
                "file_test_y": str(file_test_y),
                "num_data": str(num_data),
                "num_unit": str(units),
                "epochs": str(epochs),
                "err": str(err),
                "test_loss": str(score[0]),
                "test_acc": str(score[1]),
                "active_func": str(active_function),
                "reshape_type": str(reshape_type),
                "num_conv_node": str(num_conv_node)
                }
        json.dump(memo, f, indent=4)


def reshape_complex(x, mode="abs_phase", normalize=True):

    if normalize:
        x = x / np.max(np.abs(x))

    if ("complex" in str(type(x[0, 0, 0, 0]))):
        if mode == "abs_phase":
            x = reshape_complex_to_abs_phase(x, axis=1)
            if normalize:
                x[:, 1, :, :] = (x[:, 1, :, :] + np.pi) / (2. * np.pi)
        if mode == "abs":
            x = np.abs(x)
        if mode == "real_imag":
            x = reshape_complex_to_real_imag(x, axis=1)
    return x


def reshape_complex_to_real_imag(x, axis=1):
    return np.append(np.real(x), np.imag(x), axis=axis)


def reshape_complex_to_abs_phase(x, axis=1):
    return np.append(np.abs(x), np.angle(x), axis=axis)


def regressin_CNN(x, y, xt, yt, num_data, units, sfolder, epochs=100,
                  active_function="relu", num_conv_node="num_conv_node"):
    x = np.array(x)
    y = np.array(y)
    xt = np.array(xt)
    yt = np.array(yt)

    for s in [x, y, xt, yt]:
        print(np.shape(s))

    ydim = np.shape(y)
    out_dim = ydim[1] * ydim[2] * ydim[3]
    y = y.reshape(-1, out_dim)
    yt = yt.reshape(-1, out_dim)

    print("")
    print("check point 1")
    for s in [x, y, xt, yt]:
        print(np.shape(s))
 
    active_func = active_function
    # "relu" # "tanh" # "linear" # "sigmoid" # # "relu", "sigmoid", "tanh",

    ncn = num_conv_node

    model = Sequential()
    model.add(Conv2D(ncn, (6, 6), activation=active_func, padding="same", data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_first"))
    model.add(Flatten())

    for unit in units:
        model.add(Dense(unit, activation=active_func))
        # model.add(Dropout(0.2))
    # try:
    #     for unit in units:
    #         model.add(Dense(unit, activation=active_func))
    #         # model.add(Dropout(0.2))
    # except:
    #     pass

    model.add(Dense(out_dim))

    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    history = model.fit(x=x,
                        y=y,
                        epochs=epochs,
                        validation_data=(xt, yt),
                        verbose=1)
    y_pred = model.predict(xt)

    file = "learning"
    histroy_fig(history, file, sfolder)
    save_history(history, file, sfolder)
    save_model(model, file, sfolder)
    save_components_fit(yt, y_pred, sfolder)
    save_components_fit_9(yt, y_pred, sfolder)

    score = model.evaluate(x=xt, y=yt)
    err = np.sqrt(np.mean((yt - y_pred) ** 2))
    err2 = np.mean(np.sqrt((yt - y_pred) ** 2))
    print("  err :", err)
    print("  err2:", err2)
    print("scoore:", score)

    return (err, err2), score, y_pred


def regressin_set(x, y, xt, yt, num_data, units, sfolder, epochs=100):
    x = np.array(x)
    y = np.array(y)
    xt = np.array(xt)
    yt = np.array(yt)

    in_dim = np.shape(x)[1]
    out_dim = np.shape(yt)[1]

    for s in [x, y, xt, yt]:
        print(np.shape(s))
    
    model = Sequential()
    model.add(Dense(units[0], activation='relu', input_shape=(in_dim,)))

    for unit in units[1:]:
        model.add(Dense(unit, activation='relu'))
    # try:
    #     for unit in units[1:]:
    #         model.add(Dense(unit, activation='relu'))
    # except:
    #     pass

    model.add(Dense(out_dim))
    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    history = model.fit(x=x,
                        y=y,
                        epochs=epochs,
                        validation_data=(xt, yt),
                        verbose=2)
    y_pred = model.predict(xt)

    file = "learning"
    histroy_fig(history, file, sfolder)
    save_history(history, file, sfolder)
    save_model(model, file, sfolder)
    save_components_fit(yt, y_pred, sfolder)

    score = model.evaluate(x=xt, y=yt)
    err = np.sqrt(np.mean((yt - y_pred) ** 2))
    print("   err:", err)
    print("scoore:", score)

    return err, score, y_pred


def main_make_figs(file_x, file_y, sfolder, num_data=0):
    (x, y, t) = load_data(file_x, file_y, num_data=num_data)


def main(data_folder, num_data, active_function="relu", sfolder="", reshape_type="abs_phase", num_conv_node=50):
    strtime = datetime.now().strftime("%Y%m%d%H%M" + "-" + "%S")
    print("")
    print("time=", strtime)

    # 保存フォルダの指定,時刻がフォルダの名前になる
    if not sfolder:
        sfolder = strtime

    MAKE_DIR(sfolder)   # 保存フォルダの作成
    COPY_FILES(sfolder, ["py", "json"])   # 実行プログラムソースコードのバックアップ

    file_train_y = folder + "/" + "mirror_train_data.pkl"
    file_train_x = folder + "/" + "detec_train_data.pkl"

    file_test_y = folder + "/" + "mirror_test_data.pkl"
    file_test_x = folder + "/" + "detec_test_data.pkl"

    # main_make_figs(file_train_x,file_train_y,sfolder,num_data=num_data)

    # main_learn(file_train_x,file_test,sfolder,num_data)
    
    # main_regression_learn(file_train_x,file_train_y,
    #                       file_test_x, file_test_y,
    #                       sfolder,num_data=num_data)

    main_regression_CNN_learn(file_train_x, file_train_y,
                              file_test_x, file_test_y, sfolder,
                              num_data=num_data,
                              active_function=active_function,
                              reshape_type=reshape_type,
                              num_conv_node=num_conv_node)


if __name__ == '__main__':
    folder = "result/2020_1008_100022_add_noise_paramator"
    num_data = 4
    
    # for af, mode, ncn in itertools.product(["relu", "linear", "tanh", "sigmoid"], ["abs_phase", "abs", "real_imag"], [100]):
    #     # if af=="tanh" and mode =="abs_phase":
    #     #     continue
    #     main(folder, num_data, active_function=af, reshape_type=mode, num_conv_node=ncn)

    main(folder, num_data)

# ["tanh","linear","sigmoid","relu"] : #",10000]:
