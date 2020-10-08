# coding: utf-8
# import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
# import csv
import pickle
import json
# from numpy.fft import fftn


def make_teacher_data(input):

    nx, ny = input["target"]["num_x"], input["target"]["num_y"]
    lx, ly = input["target"]["lx"], input["target"]["ly"]
    amp = input["amp"]
    noise = input["noise"]
    max_m = input["target"]["max_m"]
    max_n = input["target"]["max_n"]
    list_m = np.arange(max_m)
    list_n = np.arange(max_n)
    ndat = input["learning"]["ndat"]
    # ntest = input["learning"]["ntest"]
    pkl_file = input["pkl_surface_teacher"]
    
    data = []
    print("make_teacher_data")
    print("out=", pkl_file)
    for m, n in itertools.product(list_m, list_n):
        print(m, n)
        for i in range(ndat):  # 400
            x, y, z = make_surface(
                m=m, n=n, nx=nx, ny=ny, lx=lx, ly=ly, amp=amp, noise=noise)
            row = {"m": m, "n": n, "z": z, "x": x, "y": y}
            data.append(row)

    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)


def make_test_data(input):
    nx, ny = input["target"]["num_x"], input["target"]["num_y"]
    lx, ly = input["target"]["lx"], input["target"]["ly"]
    amp = input["amp"]
    noise = input["noise"]
    max_m = input["target"]["max_m"]
    max_n = input["target"]["max_n"]
    list_m = np.arange(max_m)
    list_n = np.arange(max_n)
    # ndat = input["learning"]["ndat"]
    ntest = input["learning"]["ntest"]
    pkl_file = input["pkl_surface_test"]

    data = []
    print("make_test_data")
    print("out=", pkl_file)
    for m, n in itertools.product(list_m, list_n):
        print(m, n)
        for i in range(ntest):
            x, y, z = make_surface(
                m=m, n=n, nx=nx, ny=ny, lx=lx, ly=ly, amp=amp, noise=noise)
            row = {"m": m, "n": n, "z": z, "x": x, "y": y}
            data.append(row)

    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)


def load_data(*files, num_data=0):
    if len(files) == 1:
        return load_data_1(files[0], num_data=num_data)
    elif len(files) == 2:
        return load_data_2(files[0], files[1], num_data=num_data)


def load_data_2(file1, file2, num_data=0):
    text = str(num_data)
    if num_data == 0:
        text = ""

    new_file = os.path.dirname(file1) + "/" \
        + os.path.basename(file1).split(".")[0] + "_" \
        + os.path.basename(file2).split(".")[0] + "_" \
        + text + ".pkl"

    if not os.path.isfile(new_file):
        print("make combined file: ", new_file)
        new_file = combine_pkl_data(file1, file2,
                                    new_file, num_data=num_data)

    print("load_data of input", file1, file2)
    print("load_data:", new_file)

    with open(new_file, "rb") as f:
        data = pickle.load(f)

    # z1, z2, t = [], [], []
    # ind = (np.random.permutation(len(data))[:num_data])

    # for i in ind:  # 2019.10.07 reducing data for memory flow
    #     row = data[i]
    #     z1.append([row["z1"]])
    #     z2.append([row["z2"]])
    #     t.append([row["m"], row["n"]])

    z1, z2, t = [], [], []

    for d in data:
        z1.append([d["z1"]])
        z2.append([d["z2"]])
        t.append([d["info"]])

    z1 = np.array(z1)
    z2 = np.array(z2)
    t = np.array(t)
    return (z1, z2, t)


def combine_pkl_data(file1, file2, new_file, num_data=0):

    with open(file1, "rb") as f:
        detec_data = pickle.load(f)
    with open(file2, "rb") as f:
        mirror_data = pickle.load(f)

    # if len(data1) != len(data2):
    #     print("error @ combine_pkl_data in maiking_2D_image_***")

    # if num_data == 0:
    #     num_data = len(data1)

    # ind = (np.random.permutation(len(data1))[:num_data])

    # data = []
    # for i in ind:  # 2019.10.07 reducing data for memory flow
    #     row1 = data1[i]
    #     row2 = data2[i]
    #     # print( i, row1["m"],row1["n"],row2["m"],row2["n"])
    #     row = {
    #         "m": row1["m"],
    #         "n": row1["n"],
    #         "z1": row1["z"],
    #         "z2": row2["z"]}
    #     data.append(row)

    i = 0
    data = []
    for e_data in mirror_data["elip"]:
        row = {
            "info": detec_data[i]["info"],
            "z1": detec_data[i]["z"],
            "z2": e_data["z"]
        }
        i += 1
        data.append(row)

    for m_data in mirror_data["mode"]:
        row = {
            "info": detec_data[i]["info"],
            "z1": detec_data[i]["z"],
            "z2": m_data["z"]
        }
        i += 1
        data.append(row)

    with open(new_file, 'wb') as f:
        pickle.dump(data, f)

    return new_file


def load_data_1(file, num_data=0):
    file = reduce_pkl_data(file, num_data=num_data)

    x = []
    t = []
    with open(file, "rb") as f:
        data = pickle.load(f)

    if num_data == 0:
        num_data = len(data)

    ind = (np.random.permutation(len(data))[:num_data])

    for i in ind:  # 2019.10.07 reducing data for memory flow
        row = data[i]
        x.append([row["z"]])
        t.append(row["n"])

    x = np.array(x)
    t = np.array(t)
    return(x, t)


def reduce_pkl_data(file, num_data=0):
    # print(file)
    if not os.path.isfile(file):
        print("no such file", file)
        print("exit()")
        exit()

    if num_data == 0:
        return file

    file_out = os.path.splitext(file)[0] + "_" + str(num_data) + ".pkl"
    # print(file_out)
    if os.path.isfile(file_out):
        return file_out

    with open(file, "rb") as f:
        data = pickle.load(f)

    data = np.array(data)
    ind = (np.random.permutation(len(data))[:num_data])
    data = data[ind]

    print("save new_file:", file_out)
    with open(file_out, 'wb') as f:
        pickle.dump(data, f)

    return file_out


def load_surf_data(file):
    x = []
    y = []
    z = []
    t = []
    with open(file, "rb") as f:
        data = pickle.load(f)

    for row in data:
        x.append([row["x"]])
        y.append([row["y"]])
        z.append([row["z"]])
        t.append((row["m"], row["n"]))

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    t = np.array(t)

    return (x, y, z, t)


def make_surface(nx=16, ny=16, m=1, n=1, lx=1., ly=1., amp=1., noise=0):
    xx = np.linspace(-0.5, 0.5, nx) * lx
    yy = np.linspace(-0.5, 0.5, ny) * ly
    x, y = np.meshgrid(xx, yy, indexing="ij")
    phi = np.random.rand() * 2. * np.pi
    oz = np.zeros((nx, ny), dtype=np.complex)
    oz[m, n] = np.sin(phi) + 1j * np.cos(phi)
    oz[-m, -n] = np.sin(phi) - 1j * np.cos(phi)
    oz = oz.transpose() * (nx * ny / 2) * amp / 2.
    oi = np.fft.ifftn(oz)
    """
    os = np.fft.fftshift(oz)
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(np.abs(os))
    ax[1].imshow(np.real(oi))
    ax[2].imshow(np.imag(oi))
    plt.show()
    """
    oi = oi + noise * np.random.randn(nx, ny)
    return x, y, np.real(oi)


def make_test_image(nx=16, ny=16, m=1, n=1, lx=1., ly=1.):
    # xx  = np.linspace(-nx/2,nx/2,nx)
    # yy  = np.linspace(-ny/2,ny/2,ny)

    xx = np.linspace(-0.5, 0.5, nx) * lx
    yy = np.linspace(-0.5, 0.5, ny) * ly
    x, y = np.meshgrid(xx, yy, indexing="ij")
    kx, ky = 2. * np.pi * m / lx, 2. * np.pi * n / ly
    # zz  = np.sin(x*2.*np.pi*m + y*2.*np.pi*n).transpose()
    zz = np.sin(kx * x + ky * y).transpose()
    # zz = np.array([np.sin(xx*2.*np.pi/m + y*2.*np.pi/n) for y in yy])

    fz = np.fft.fftn(zz)
    print(fz[m, n])
    print(fz[-m, -n])
    fs = np.fft.fftshift(fz)
    iz = np.fft.ifftn(fz)

    print(np.shape(fz))
    oz = np.zeros((nx, ny))
    oz[m, n] = 1.
    oz[-m, -n] = 1.
    oz = oz.transpose() * (nx * ny / 2)
    oi = np.fft.ifftn(oz)
    os = np.fft.fftshift(oz)

    fig, ax = plt.subplots(2, 3)
    ax[0][0].imshow(zz)
    ax[0][1].imshow(np.abs(fs))
    ax[0][2].imshow(np.abs(os))
    ax[1][0].imshow(np.real(iz))
    ax[1][1].imshow(np.imag(iz))
    ax[1][2].imshow(np.real(oi))
    # ax[1][2].imshow(np.abs(iz))
    plt.show()

    print(np.allclose(zz, iz))
    print(np.real(fz[n, :]))
    print(np.abs(oz[n, :]))


def surf_figure(file):
    (x, t) = load_data(file)
    fig = plt.figure()
    temp1 = np.random.randint(0, np.shape(x)[0], size=16)
    for i, dat in enumerate(x[temp1]):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.set_title(t[temp1[i]])
        ax.set_xticks([], minor=False)
        ax.set_yticks([], minor=False)
        ax.imshow(dat[0, :, :])

    plt.savefig(file.split(".")[0] + ".png")
    plt.close()
    # plt.show()


if __name__ == '__main__':
    # make_test_image(m=1,n=2,nx=16,ny=32,lx=10.,ly=10.)
    # x,y,z = make_surface(m=1,n=2,nx=16,ny=32,lx=10.,ly=10.,amp=12.)
    # print(np.max(z),np.min(z))
    # main()
    # test()

    # surf_figure("plk_surf_teacher.pkl")
    # exit()

    with open("input0.json", "r") as f:
        input = json.load(f)
        input["pkl_file_teacher"] = 'train_data.pkl'
        input["pkl_file_test"] = 'test_data.pkl'

    pkl_file_teacher = input["pkl_file_teacher"]
    pkl_file_test = input["pkl_file_test"]

    make_teacher_data(input)
    make_test_data(input)
    (x_train, t_train) = load_data(pkl_file_teacher)
    (x_test, t_test) = load_data(pkl_file_test)
    print(np.shape(x_train))

    print(np.shape(t_train))

    temp1 = np.random.randint(0, np.shape(x_train)[0], size=10)
    temp2 = np.random.randint(0, np.shape(x_test)[0], size=10)

    print(t_train[temp1])
    print(t_test[temp2])
    fig, ax = plt.subplots(2, 10)
    for i, dat in enumerate(x_train[temp1]):
        ax[0][i].imshow(dat[0, :, :])
    for i, dat in enumerate(x_test[temp2]):
        ax[1][i].imshow(dat[0, :, :])
    plt.show()
