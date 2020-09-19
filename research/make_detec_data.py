import numpy as np
import pickle
import json
from tqdm import tqdm

from propagation_mod import (
    MAKE_SOURCE_CORD,
    MAKE_DETECT_CORD,
    MAKE_NOISE_LEVEL,
    MAKE_PROPAG_MTRX,
    SELECT_MATRIX_TYPE,
    WAVE_MAT_CAL,
    SELECT_ADD_NOISE)


def load_mirror_data(mirror_data):
    elip_mirror_data = mirror_data["elip"]
    mode_mirror_data = mirror_data["mode"]

    x = []
    y = []
    z = []
    shape = []
    info = []

    if len(elip_mirror_data) > 0:
        for row in elip_mirror_data:
            x.append(row["x"])
            y.append(row["y"])
            z.append(row["z"])
            shape.append("elip")
            info.append(row["info"])

    if len(mode_mirror_data) > 0:
        for row in mode_mirror_data:
            x.append(row["x"])
            y.append(row["y"])
            z.append(row["z"])
            shape.append("mode")
            info.append(row["info"])
        
    np_x = np.array(x)
    np_y = np.array(y)
    np_z = np.array(z)
    np_shape = np.array(shape)
    np_info = np.array(info)
    
    return np_x, np_y, np_z, np_shape, np_info


def reflect_detector(xx, yy, zz, detec_params):
    target_cords = np.array([xx, yy, zz]).transpose()
    target_value = np.ones((target_cords.shape[0]))

    source_cords = MAKE_SOURCE_CORD(detec_params)
    detect_cords = MAKE_DETECT_CORD(detec_params)  # 検出器の座標のベクトル
    nd = detect_cords.shape[0]         # number of measurement data
    noise_level = MAKE_NOISE_LEVEL(detec_params)

    H_complex = MAKE_PROPAG_MTRX(detec_params, target_cords, detect_cords, source_cords, cpu=20)
    H, nd = SELECT_MATRIX_TYPE(H_complex, nd, detec_params)  # Hを実数に積み上げるかどうか

    f = target_value  # + 1j * 0.0 # fは虚部０の複素数であることを明示　
    g0 = WAVE_MAT_CAL(H, f, divs=None)  # g0 = Hf
    g = SELECT_ADD_NOISE(g0, noise_level, detec_params)  # g = g0 + noise

    #fig,ax = plt.subplots(1,1)
    # ax.imshow(np.imag(g).reshape((20,20)))
    # plt.show()

    # SAVE_FIGURE_2([np.real(g), np.real(g0),np.real(g0)], \
    #              [np.imag(g), np.imag(g0),np.imag(g0)], \
    #               "g_g_with_noise","")

    # CHECK_3D_PROFILE(detect_cords,np.real(g),"g_real","")
    # CHECK_3D_PROFILE(detect_cords,np.imag(g),"g_imag","")
    #CHECK_3D_PROFILE(detect_cords,np.abs(g), "g_abs","")

    return g


def _make_detec_data(detec_params, mirror_data):
    detec_data = []
    (x, y, z, shape, info) = load_mirror_data(mirror_data)

    for i in tqdm(range(x.shape[0])):
        xx = x[i, :, :].flatten()
        yy = y[i, :, :].flatten()
        zz = z[i, :, :].flatten()
        g = reflect_detector(xx, yy, zz, detec_params)
        g = np.reshape(g, (detec_params["detector"]["num_x"], detec_params["detector"]["num_y"]))
        row = {"z": g, "shape": shape[i], "info": info[i]}
        detec_data.append(row)

    return detec_data


def make_detec_data(detec_params, train_mirror_data, test_mirror_data):
    print("Make train data!\n")
    train_detec_data = _make_detec_data(detec_params, train_mirror_data)
    print("Complete train data!!")

    print("Make test data!\n")
    test_detec_data = _make_detec_data(detec_params, test_mirror_data)
    print("Complete test data!!")

    return train_detec_data, test_detec_data


def main_make_detec_data(result_folder):
    start_folder = "./run_instruments/"
    name_json_mirror_params = "params_making_mirror_data.json"
    name_json_detec_params = "params_making_detec_data.json"

    # detectorのjsonパラメータをロード
    with open(start_folder + name_json_detec_params, "r") as f:
        detec_params = json.load(f)

    # detectorのjsonパラメータを結果にdump
    with open(result_folder + name_json_detec_params, "w") as f:
        json.dump(detec_params, f, indent=2)

    # mirrorのパラメーターjsonからmirrorのデータ名をロード
    with open(start_folder + name_json_mirror_params, "r") as f:
        mirror_params = json.load(f)

    # mirrorのデータをロード
    with open(result_folder + mirror_params["pkl_mirror_train"], "rb") as f:
        train_mirror_data = pickle.load(f)
    with open(result_folder + mirror_params["pkl_mirror_test"], "rb") as f:
        test_mirror_data = pickle.load(f)

    train_detec_data, test_detec_data = make_detec_data(detec_params, train_mirror_data, test_mirror_data)

    with open(result_folder + detec_params["pkl_detec_train"], "wb") as f:
        pickle.dump(train_detec_data, f)
        print("output: ", detec_params["pkl_detec_train"])
    with open(result_folder + detec_params["pkl_detec_test"], "wb") as f:
        pickle.dump(test_detec_data, f)
        print("output: ", detec_params["pkl_detec_test"])

    print("Build {} folder\n".format(result_folder))


if __name__ == "__main__":
    start_folder = "./run_instruments/"
    name_json_mirror_params = "params_making_mirror_data.json"
    name_json_detec_params = "params_making_detec_data.json"

    # detectorのjsonパラメータをロード
    with open(start_folder + name_json_detec_params, "r") as f:
        detec_params = json.load(f)

    result_folder = "result/" + detec_params["result_folder"] + "/"

    # detectorのjsonパラメータを結果にdump
    with open(result_folder + name_json_detec_params, "w") as f:
        json.dump(detec_params, f, indent=2)

    # mirrorのパラメーターjsonからmirrorのデータ名をロード
    with open(start_folder + name_json_mirror_params, "r") as f:
        mirror_params = json.load(f)

    # mirrorのデータをロード
    with open(result_folder + mirror_params["pkl_mirror_train"], "rb") as f:
        train_mirror_data = pickle.load(f)
    with open(result_folder + mirror_params["pkl_mirror_test"], "rb") as f:
        test_mirror_data = pickle.load(f)

    train_detec_data, test_detec_data = make_detec_data(detec_params, train_mirror_data, test_mirror_data)

    with open(result_folder + detec_params["pkl_detec_train"], "wb") as f:
        pickle.dump(train_detec_data, f)
        print("output: ", detec_params["pkl_detec_train"])
    with open(result_folder + detec_params["pkl_detec_test"], "wb") as f:
        pickle.dump(test_detec_data, f)
        print("output: ", detec_params["pkl_detec_test"])

    print("Build {} folder\n".format(result_folder))
