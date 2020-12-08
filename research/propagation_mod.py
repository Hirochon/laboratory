"""
verinfo = 座標系やターゲットの反射率の設定など幾何学に関わる関数
ver1.1 2018.11.18 check for ITC27
ver1.0 2018.11.13 Developing codes
by H.Tsuchiya (NIFS)
"""
import numpy as np
from numpy import sin, cos, pi, exp
# from numpy import sin, cos, tan, pi, exp
# import pandas as pd
# import json
# import collections as cl
import itertools
# from file_operation import *
# from figure_mod import *


def SELECT_MATRIX_TYPE(H_complex, nd, input_data):
    if input_data["matrix_type"]["type"] == "real":
        H = np.append(np.real(H_complex),
                      np.imag(H_complex), axis=0)  # Hを積み上げる
        nd = nd * 2
    elif input_data["matrix_type"]["type"] == "complex":
        H = H_complex    # Hは複素数のまま
        nd = nd * 1
    else:
        print("err: not type in selection of matrix type:",
              input_data["matrix_type"]["type"])
        exit()
    # print(nd,input_data["matrix_type"]["type"])
    return H, nd


def SELECT_ADD_NOISE(g0, noise_level, input_data):  # g = g0 + noise
    if input_data["matrix_type"]["type"] == "real":
        # print(input_data["matrix_type"]["type"] )
        g = ADD_NOISE(g0, noise_level)
    elif input_data["matrix_type"]["type"] == "complex":
        g = ADD_NOISE(np.real(g0), noise_level) + \
            ADD_NOISE(np.imag(g0), noise_level) * 1j
    else:
        print("err: not type in selection of matrix type:",
              input_data["matrix_type"]["type"])
        exit()
    return g

# ===============================================================================
# def MAKE_TARGET_CORD(input_data):

#     #基準点（枠の一番端）を計算しておく
#     cx = input_data["target"]["center"][0] \
#             - (input_data["target"]["num_x"]-1)/2 * input_data["target"]["dx"]
#     cy = input_data["target"]["center"][1] \
#             - (input_data["target"]["num_y"]-1)/2 * input_data["target"]["dy"]
#     cz = input_data["target"]["center"][2] \
#             - (input_data["target"]["num_z"]-1)/2 * input_data["target"]["dz"]

#     target_cords = np.array(
#         [(cx+i*input_data["target"]["dx"],
#           cy+j*input_data["target"]["dy"],
#           cz+k*input_data["target"]["dz"])
#          for i in range(input_data["target"]["num_x"])
#          for j in range(input_data["target"]["num_y"])
#          for k in range(input_data["target"]["num_z"])
#         ])

#     return target_cords

# ===============================================================================
# def MAKE_TARGET_VALS(target_cords,input_data):
#     #target_val = MAKE_TARGET_VALS_1(target_cords)
#     #target_val = MAKE_TARGET_VALS_X0_PLANE(target_cords)
#     if input_data["target"]["type"] == "MAKE_TARGET_VALS_X_COS2" :
#         target_val = MAKE_TARGET_VALS_X_COS2(target_cords)
#     if input_data["target"]["type"] == "REAL_SIN_IMAG_ZERO" :
#         target_val = REAL_SIN_IMAG_ZERO(target_cords)
#     return target_val

# def MAKE_TARGET_VALS_1(target_cords):
#     target_val = np.empty((target_cords.shape[0]))
#     n = target_cords.shape[0]
#     for i in range(n):
#         target_val[i] = 0.1E0 * 1/np.sqrt(10.+DISTANCE(target_cords[i],[0,0,0]))
#     return target_val

# def MAKE_TARGET_VALS_X0_PLANE(target_cords):
#     target_val = np.empty((target_cords.shape[0]))
#     n = target_cords.shape[0]
#     for i in range(n):
#         print(i,target_cords[i][0],target_cords[i][1],target_cords[i][2])
#         if target_cords[i][0] == 0.0 :
#             target_val[i] = 0.1E0
#         else:
#             target_val[i] = 0.E0
#     return target_val

# def MAKE_TARGET_VALS_X_COS2(target_cords):
#     target_val = np.empty((target_cords.shape[0]))
#     n = target_cords.shape[0]
#     xmax = target_cords.min(axis=0)[0]
#     xmin = target_cords.max(axis=0)[0]
#     xl   = xmax-xmin
#     for i in range(n):
#         target_val[i] = 0.01*cos(np.pi*target_cords[i][0]/xl)**2
#         #print(target_cords[i])
#         #print(i,target_cords[i][0],target_cords[i][1],target_cords[i][2])
#     return target_val

# def REAL_SIN_IMAG_ZERO(target_cords):
#     target_val = np.empty((target_cords.shape[0]))
#     n = target_cords.shape[0]
#     for i in range(n):
#         target_val[i] = sin(2.*np.pi*i/n)
#     return target_val


# ===============================================================================
def MAKE_DETECT_CORD(input_data):

    # 基準点（枠の一番端）を計算しておく
    cx = input_data["detector"]["center"][0] - \
        (input_data["detector"]["num_x"] - 1) / 2 * input_data["detector"]["dx"]
    cy = input_data["detector"]["center"][1] - \
        (input_data["detector"]["num_y"] - 1) / 2 * input_data["detector"]["dy"]
    cz = input_data["detector"]["center"][2] - \
        (input_data["detector"]["num_z"] - 1) / 2 * input_data["detector"]["dz"]

    detector_cords = np.array(
        [(cx + i * input_data["detector"]["dx"],
          cy + j * input_data["detector"]["dy"],
          cz + k * input_data["detector"]["dz"])
         for i in range(input_data["detector"]["num_x"])
         for j in range(input_data["detector"]["num_y"])
         for k in range(input_data["detector"]["num_z"])
         ])

    # rotation matrix
    r = input_data["detector"]["x_theata"] / 180 * pi
    rx = np.matrix([(1., 0., 0.),
                    (0., cos(r), sin(r)),
                    (0., -sin(r), cos(r))])

    r = input_data["detector"]["y_theata"] / 180 * pi
    ry = np.matrix([(cos(r), 0., -sin(r)),
                    (0., 1., 0.),
                    (sin(r), 0., cos(r))])

    r = input_data["detector"]["z_theata"] / 180 * pi
    rz = np.matrix([(cos(r), 0., -sin(r)),
                    (0., 1., 0.),
                    (sin(r), 0., cos(r))])

    for i in range(detector_cords.shape[0]):
        detector_cords[i] = detector_cords[i] * rx * ry * rz

    return detector_cords

# ===============================================================================


def MAKE_SOURCE_CORD(input_data):
    source_cords = np.array([input_data["source"]["center"]])
    return source_cords

# ===============================================================================


def MAKE_NOISE_LEVEL(input_data):
    noise_level = input_data["noise"]
    return noise_level


# ===============================================================================
def MAKE_PROPAG_MTRX(
        input_data,
        target_cords,
        detect_cords,
        source_cords,
        cpu=0):

    s1 = input_data["propagation_mode"]["source_to_target"]
    s2 = input_data["propagation_mode"]["taret_to_detector"]

    if (s1 == "spherical" and s2 == "spherical"):
        H = SPHERICAL_SPHERICAL(input_data, target_cords,
                                detect_cords, source_cords,
                                cpu=cpu)
    else:
        print("err: in MAKE_PROPAG_MTRX", s1, s2)
        exit()

    return H


def SPHERICAL_SPHERICAL(
        input_data,
        target_cords,
        detect_cords,
        source_cords,
        cpu=None):
    import time
    import multiprocessing as mp
    freq = input_data["freq"]
    vc = 29979245
    kr = 2.E0 * pi * freq / vc / 1.E3  # 単位系はmm
    H = np.empty((detect_cords.shape[0], target_cords.shape[0]), dtype=complex)
    if cpu is None:
        t1 = time.time()
        for i in range(detect_cords.shape[0]):
            for j in range(target_cords.shape[0],):
                # r1 = DISTANCE(target_cords[j],source_cords[0])
                # r2 = DISTANCE(target_cords[j],detect_cords[i])
                # ss = SPHERICAL_WAVE(kr,r1) * SPHERICAL_WAVE(kr,r2)
                # H[i][j] = ss
                H[i][j] = WAVE_COMPONENT_CAL(detect_cords[i],
                                             target_cords[j],
                                             source_cords[0],
                                             kr)
        t2 = time.time()
        print("W/O  multiprocessing cpu=", cpu, " cputime=", t2 - t1)
    else:
        t1 = time.time()
        # H0 = np.empty(
        #     (detect_cords.shape[0],
        #      target_cords.shape[0]),
        #     dtype=complex)
        if cpu == 0:
            cpu = mp.cpu_count()
        p = mp.Pool(processes=cpu)
        arg_list = itertools.product(detect_cords[:],
                                     target_cords[:],
                                     [source_cords[0]],
                                     [kr])
        res = p.starmap(WAVE_COMPONENT_CAL, arg_list)
        p.close()
        H = np.array(res).reshape(
            (detect_cords.shape[0], target_cords.shape[0]))
        t2 = time.time()
        # print("With multiprocessing cpu=",cpu," cputime=",t2-t1)

    return H


def WAVE_COMPONENT_CAL(detect_cord, target_cord, source_cord, kr):
    r1 = DISTANCE(target_cord, source_cord)
    r2 = DISTANCE(target_cord, detect_cord)
    ss = SPHERICAL_WAVE(kr, r1) * SPHERICAL_WAVE(kr, r2)
    return ss

# ===============================================================================


def DISTANCE(p1, p2):
    r = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2
    r = np.sqrt(r)
    return r

# ===============================================================================


def SPHERICAL_WAVE(kr, r):
    s = 1.0E0 / r * exp(complex(0, kr * r))
    return s

# ===============================================================================


def WAVE_MAT_CAL(g, target_val, divs=None):
    # import time
    if divs is None:
        # t1 = time.time()
        s = np.empty((g.shape[1]))
        s = np.dot(g, target_val)
        # t2 = time.time()
        # print("W/O  DASK: d=",divs," cputime=",t2-t1)
    else:
        if divs == 0:
            divs = int(np.max(g.shape) / 500)
        # t1 = time.time()
        s = WAVE_MAT_CAL_by_DASK(g, target_val, d=divs)
        # t2 = time.time()
        # print("With DASK: d=",divs," cputime=",t2-t1)
    return s


def WAVE_MAT_CAL_by_DASK(x, y, d=4):
    import dask.array as da
    nx1, nx2 = int(np.shape(x)[0] / d), int(np.shape(x)[1] / d)
    ny1 = int(np.shape(y)[0] / d)
    dax = da.from_array(x, chunks=(nx1, nx2))
    day = da.from_array(y, chunks=(ny1))
    res = da.dot(dax, day)
    s = res.compute()
    return np.array(s)


def ADD_NOISE(g0, noise_level):
    # g = g0 + \
    #    np.random.randn(g0.shape[0]) * np.average(np.abs(g0)) * noise_level + \
    #    np.random.randn(g0.shape[0]) * np.average(np.abs(g0)) * noise_level * 1j
    g = g0 + np.random.randn(g0.shape[0]) * \
        np.average(np.abs(g0)) * noise_level
    return g


# ===============================================================================
# ===============================================================================
# if __name__ == "__main__":
#     input_data = READ_INPUT_DATA("input.json")
#     target_cords = MAKE_TARGET_CORD(input_data)
#     target_value = MAKE_TARGET_VALS(target_cords)
#     detect_cords = MAKE_DETECT_CORD(input_data)
#     source_cords = MAKE_SOURCE_CORD(input_data)
#     noise_level = MAKE_NOISE_LEVEL(input_data)
#     H = MAKE_PROPAG_MTRX(input_data,
#                          target_cords,
#                          detect_cords,
#                          source_cords)
#     g0 = WAVE_MAT_CAL(H, target_value)
#     g = ADD_NOISE(g0, noise_level)  # g = g0 + noise
#     CHECK_3D_PROFILE(target_cords, target_value,
#                      "_target_val")
#     SAVE_FIGURE_2([np.real(g), np.real(g0), np.real(g0)],
#                   [np.imag(g), np.imag(g0), np.imag(g0)],
#                   "_g_g_with_noise", "")
#     print("save figures", "_target_val", "_g_g_with_noise")
