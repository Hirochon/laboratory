#!/usr/bin/env python

# coding: utf-8
import sys,os,itertools
sys.path.append(os.pardir)
import numpy as np
import json
import pickle
from datetime import datetime
import subprocess
import matplotlib.pyplot as plt
from file_operation import *
from making_2D_image_20191022 import load_data,surf_figure,reduce_pkl_data

#from common.simple_convnet import SimpleConvNet
#from common.trainer import Trainer
#from dataset.mnist import load_mnist
#import matplotlib as mpl

from keras.datasets import mnist
from keras.utils import np_utils,plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout,                          Conv2D, MaxPooling2D, Reshape,                          SeparableConv2D
from keras.callbacks import EarlyStopping, CSVLogger


def main(x_len,y_len,num_data,units,sfolder,epochs = 100):
    x, y = [],[]
    xt,yt = [],[]
    x_len = x_len
    y_len = y_len
    H = np.array( [ np.random.random(x_len)-0.5 for i in range(y_len) ] )
    noise = 0.1

    num_data = num_data
    for i in range(num_data):
        xx = np.random.random(x_len)
        yy = np.dot(H,xx)
        x.append(xx)
        y.append(yy+noise*np.random.randn(y_len))

    num_test_data = 100
    for i in range(num_test_data):
        xx = np.random.random(x_len)
        yy = np.dot(H,xx)
        xt.append(xx)
        yt.append(yy)

        


    # In[4]:


    x = np.array(x)
    y = np.array(y)
    xt = np.array(xt)
    yt = np.array(yt)
    #np.shape(x),np.shape(y)


    # In[5]:


    model = Sequential()
    model.add(Dense(units[0], activation='relu', input_shape = (x_len, )))
    try:
        for unit in units[1:]:
            model.add(Dense(unit, activation='relu'))
    except:
        pass
    model.add(Dense(y_len))
    model.compile(optimizer='rmsprop',
                loss='mean_squared_error',
                metrics=['accuracy'])

    epochs = epochs
    model.fit(x = x, y = y, epochs = epochs,validation_data=(xt,yt),verbose=2)

    y_pred = model.predict(xt)

    last = min( [y_len,50] )
    print(last,len(y_pred))
    ind = ( np.random.permutation(y_len)[:last] )

    plt.plot(yt[0][ind])
    plt.plot(y_pred[0][ind],"o-")
    plt.savefig(sfolder+"/pred_xlen"+str(x_len)+
                          "_ylen"+str(y_len)+"_"+
                          "_".join(["Desns"+str(i) for i in units])+
                          ".png")
    #plt.show()
    plt.cla()

    score = model.evaluate(x=xt, y=yt)
    err = np.sqrt( np.mean( (yt-y_pred)**2 ) )
    print(err)
    print(score)

    return ( err, score )


if __name__ == '__main__':
    
    list_x_len = [2,10,100,1000,10000]
    list_y_len = [2,10,100,1000,10000]
    list_num_data = [1000]
    list_units    = [[10],[10,10],[50],[50,50]]
    sfolder  = "taju-kaiki-201910241456-00"

    MAKE_DIR(sfolder)

    output = {}
    i =0
    for x_len,y_len,num_data,units in itertools.product( list_x_len,
                                                               list_y_len,
                                                               list_num_data,
                                                               list_units ):

        print(i,x_len,y_len,num_data,units)
        err,score = main(x_len,y_len,num_data,units,sfolder,epochs = 100)

        output[i]= {"x_len":x_len,"y_len":y_len,"num_data":num_data,
                        "units":units,"err":err,"score":score}
        i += 1 
    
    file = sfolder +"/output.json"
    with open(file, "w") as f :
        json.dump(output,f,indent=4)
