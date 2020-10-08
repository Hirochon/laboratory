"""
verinfo = グラフ表示
ver1.1 2018.11.18 check for ITC27
ver1.0 2018.11.** Developing codes
by H.Tsuchiya (NIFS)
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

#公開する関数を指定する
__all__ = ['SAVE_FIGURE_1',\
           'SAVE_FIGURE_2',\
           'SAVE_FIGURE_4',\
           'SAVE_FIGURE_4_ITC',\
           'CHECK_3D_PROFILE',\
           'SAVE_2D_FIGURE'
           ];

def SAVE_FIGURE_1(data,text,text1):
    x = np.arange(data[0].shape[0])
    fig = plt.figure()
    for g in data:
        plt.plot(x,g)
    plt.suptitle(text+"_"+text1)
    plt.savefig(text+".png")
    plt.close(fig)
    #plt.show()

def SAVE_FIGURE_2(dat1,dat2,text,text1):
    xa = np.arange(dat1[0].shape[0])
    xb = np.arange(dat2[0].shape[0])
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    for y in dat1:
        ax1.plot(xa,y)
    ax2 = plt.subplot(2,1,2)
    for y in dat2:
        ax2.plot(xb,y)
    fig.suptitle(text+"_"+text1)
    fig.savefig(text+".png")
    plt.close(fig)
    #plt.show()

def SAVE_FIGURE_4(dat1,dat2,dat3,dat4,text,text1):
    x1 = np.arange(dat1[0].shape[0])
    x2 = np.arange(dat2[0].shape[0])
    x3 = np.arange(dat3[0].shape[0])
    x4 = np.arange(dat4[0].shape[0])
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    for y in dat1:
        ax1.plot(x1,y)
    ax2 = plt.subplot(2,2,2)
    for y in dat2:
        ax2.plot(x2,y)
    ax3 = plt.subplot(2,2,3)
    for y in dat3:
        ax3.plot(x3,y)
    ax4 = plt.subplot(2,2,4)
    for y in dat4:
        ax4.plot(x4,y)
    fig.suptitle(text+"_"+text1)
    fig.savefig(text+".png")
    plt.close(fig)
    #plt.show()

def SAVE_FIGURE_4_ITC(dat1,dat2,dat3,dat4,text,text1):
    x1 = np.arange(dat1[0].shape[0])
    x2 = np.arange(dat2[0].shape[0])
    x3 = np.arange(dat3[0].shape[0])
    x4 = np.arange(dat4[0].shape[0])
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(x1,dat1[0],linewidth=3.0,label="reconstructed f",color="c")
    ax1.plot(x1,dat1[1],linewidth=1.5,label="assumed f",linestyle="--",color="r")
    ax1.legend(loc='uppper right',
               bbox_to_anchor=(-1.0, 0.5, 0.5, .100), )
    ax2 = plt.subplot(2,2,2)
    ax2.plot(x2,dat2[0],linewidth=3.0,label="reconstructed f",color="c")
    ax2.plot(x2,dat2[1],linewidth=1.5,label="assumed f",linestyle="--",color="r")
    ax2.set_ylim([-0.15,0.15])
    ax3 = plt.subplot(2,2,3)
    ax3.plot(x3,dat3[0],marker="o",linewidth=3.0,label="reconstructed g",color="c")
    ax3.plot(x3,dat3[1],label="assumed g",linestyle="--",color="r")
    ax4 = plt.subplot(2,2,4)
    ax4.plot(x4,dat4[0],marker="o",linewidth=3.0,label="reconstructed g",color="c")
    ax4.plot(x4,dat4[1],label="assumed g",linestyle="--",color="r")
    fig.suptitle(text+"_"+text1)
    fig.savefig(text+".png")
    plt.close(fig)
    #plt.show()

def CHECK_3D_PROFILE(xyz,f,text,text1):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    p = ax.scatter(x,y,z,c=f,cmap='Blues')
    fig.colorbar(p)
    fig.savefig(text+".png")
    plt.close(fig)

def SAVE_2D_FIGURE(dat,text,text1):
    fig = plt.figure()
    plt.imshow(dat,cmap="GnBu_r")
    #plt.xlabel(L1)
    #plt.ylabel(L0)
    plt.colorbar()
    plt.savefig(text+".png")
    plt.close(fig)

if __name__ == "__main__":
    x  = np.arange(-1.0,1.0,0.1)
    y0 = np.sin(np.pi * x)
    y1 = np.cos(np.pi * x)
    y2 = y0*y0
    y3 = y1*y1
    SAVE_FIGURE_1([y0,y1],"fig1","comment1")
    SAVE_FIGURE_2([y0,y1],[y2,y3],"fig2","comment2")
    SAVE_FIGURE_4([y0,y1],[y2,y3],[y0,y2],[y1,y3],"fig4","comment4")

    z  = np.reshape(y0,(4,5))
    SAVE_2D_FIGURE(z,"fig_2d","comment_2d")

    xyz = np.array(
        [(i,j,k)
         for i in range(10)
         for j in range(10)
         for k in range(10)
        ])
    f = np.empty((xyz.shape[0]))
    for i in range(xyz.shape[0]):
        f[i]= np.exp(-np.linalg.norm(xyz[i])/10)
    CHECK_3D_PROFILE(xyz,f,"fig_3d","comment_3d")
