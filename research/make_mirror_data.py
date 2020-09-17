import numpy as np
import pickle
import json
from tqdm import tqdm
import itertools

from file_operation import make_dir


def make_elip_param_list(elip_params, data_num):
    """ミラー毎の楕円のパラメータをランダム作成関数"""

    ellipse_nums = []
    param_list = []

    for data_i in range(data_num):

        ellipse_nums.append(np.random.randint(elip_params["elip_num_min"], elip_params["elip_num_max"]))    # 楕円の数[vector]

        init_elip_len_x_list = np.random.randint(elip_params["elip_len_x_min"], elip_params["elip_len_x_max"], size=ellipse_nums[data_i])    # 楕円の長さ(横)[matrix]
        init_elip_len_y_list = np.random.randint(elip_params["elip_len_y_min"], elip_params["elip_len_y_max"], size=ellipse_nums[data_i])    # 楕円の長さ(縦)[matrix]
        init_coord_x_list = np.random.randint(elip_params["coord_x_min"], elip_params["coord_x_max"], size=ellipse_nums[data_i])     # 楕円の中心からのズレ(横)[matrix]
        init_coord_y_list = np.random.randint(elip_params["coord_y_min"], elip_params["coord_y_max"], size=ellipse_nums[data_i])     # 楕円の中心からのズレ(縦)[matrix]
        init_theta_list = np.pi * np.random.rand(ellipse_nums[data_i]) * 2 / elip_params["theta_rate"]   # 楕円の回転角[matrix]

        param_list.append(np.array([init_elip_len_x_list, init_elip_len_y_list, init_coord_x_list, init_coord_y_list, init_theta_list]))
        
    return param_list, ellipse_nums, elip_params["axis_x"], elip_params["axis_y"], elip_params["nx"], elip_params["ny"]


def make_elip_spot_mirror(elip_len_x_list, elip_len_y_list, coord_x_list, coord_y_list, theta_list, axis_x, axis_y, ellipse_num, nx, ny):
    """楕円毎に与えられたパラメータに従って楕円を描いていく"""
    
    elip_spot_mirror = np.zeros([axis_x, axis_y])
    
    for k in range(ellipse_num):
        rotate = np.array([[np.cos(theta_list[k]), np.sin(theta_list[k])], [-np.sin(theta_list[k]), np.cos(theta_list[k])]])    # 回転行列の定義
        
        for j in range(axis_y):
            for i in range(axis_x):
                y = j + 1
                x = i + 1
                
                [X, Y] = np.dot([x - coord_x_list[k], y - coord_y_list[k]], rotate)     # 行列の積

                x_formula = X**2 / elip_len_x_list[k]**2    # 楕円の方程式のxとa部分
                y_formula = Y**2 / elip_len_y_list[k]**2    # 楕円の方程式のyとb部分

                if x_formula + y_formula <= 1:
                    elip_spot_mirror[j, i] += 1

    # 楕円作成後にnxとnyに合わせてスケールを変化させることを理想としている。
    # 現在は未実装なので、np.linspace(-0.5, 0.5, nx) が np.linspace(-0.5, 0.5, axis_x)となってる。
    # つまりnxとnyは引数として存在しているが、何もしていない。実装予定(記: 2020/9/11)。
    xx = np.linspace(-0.5, 0.5, axis_x) * axis_x    # -0.5〜0.5間でaxis_x個に分けて、axis_xでブロードキャスト
    yy = np.linspace(-0.5, 0.5, axis_y) * axis_y    # -0.5〜0.5間でaxis_y個に分けて、axis_yでブロードキャスト
    x, y = np.meshgrid(xx, yy, indexing="ij")   # 2次元配列としてxとyをそれぞれ用意
    
    third_dim_elip_spot_mirror = {"x": x, "y": y, "z": elip_spot_mirror}
    
    return third_dim_elip_spot_mirror


def make_mode_spot_mirror(nx=16, ny=16, m=1, n=1, axis_x=1., axis_y=1., amp=1., noise=0):
    xx = np.linspace(-0.5, 0.5, nx) * axis_x
    yy = np.linspace(-0.5, 0.5, ny) * axis_y
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


def _make_mirror_data(mirror_params, is_train):
    """ミラー作成関数(trainとtestで別々に実行)"""

    ##############
    # 共通の処理 #
    ##############
    mirror_params["elip"]["axis_x"] = mirror_params["mode"]["axis_x"] = mirror_params["common"]["axis_x"]
    mirror_params["elip"]["axis_y"] = mirror_params["mode"]["axis_y"] = mirror_params["common"]["axis_y"]
    mirror_params["elip"]["nx"] = mirror_params["mode"]["nx"] = mirror_params["common"]["nx"]
    mirror_params["elip"]["ny"] = mirror_params["mode"]["ny"] = mirror_params["common"]["ny"]
    
    if is_train:
        elip_data_num = mirror_params["elip"]["train_num"]
        mode_data_num = mirror_params["mode"]["train_num"]
    else:
        elip_data_num = mirror_params["elip"]["test_num"]
        mode_data_num = mirror_params["mode"]["test_num"]

    ##############
    # 楕円ゾーン #
    ##############
    elip_mirror_data = []
    if elip_data_num > 0:
        print("===Type Ellipse start===")

        # 楕円作成のパラメータをランダムに作成
        elip_param_list, ellipse_nums, axis_x, axis_y, nx, ny = make_elip_param_list(mirror_params["elip"], elip_data_num)

        # make_elip_param_listで作成したパラメータを元に楕円型ミラーデータを作成
        for [elip_len_x_list, elip_len_y_list, coord_x_list, coord_y_list, theta_list], ellipse_num in tqdm(zip(elip_param_list, ellipse_nums), total=elip_data_num):
            elip_spot_mirror = make_elip_spot_mirror(elip_len_x_list, elip_len_y_list, coord_x_list, coord_y_list, theta_list, axis_x, axis_y, ellipse_num, nx, ny)
            elip_spot_mirror_params = {"elip_len_x_list": elip_len_x_list, "elip_len_y_list": elip_len_y_list,
                                       "coord_x_list": coord_x_list, "coord_y_list": coord_y_list, "theta_list": theta_list,
                                       "axis_x": axis_x, "axis_y": axis_y, "ellipse_num": ellipse_num, "nx": nx, "ny": ny}
            elip_spot_mirror.update({"info": elip_spot_mirror_params})
            elip_mirror_data.append(elip_spot_mirror)
        print("===Type Ellipse end===\n")
    else:
        print("---Ignore Ellipse---\n")

    ################
    # モードゾーン #
    ################
    mode_mirror_data = []
    if mode_data_num > 0:
        print("===Type Mode start===")
        mode_params = mirror_params["mode"]
        m_list, n_list = np.arange(mode_params["max_m"]), np.arange(mode_params["max_n"])
        for m, n in itertools.product(m_list, n_list):
            print(m, n)
            for _ in tqdm(range(mode_data_num)):
                x, y, z = make_mode_spot_mirror(m=m, n=n, nx=mode_params["nx"], ny=mode_params["ny"], axis_x=mode_params["axis_x"],
                                                axis_y=mode_params["axis_y"], amp=mode_params["amp"], noise=mode_params["noise"])
                row = {"x": x, "y": y, "z": z}
                mode_spot_mirror_params = {"m": m, "n": n}
                row.update({"info": mode_spot_mirror_params})
                mode_mirror_data.append(row)
        print("===Type Mode end===\n")
    else:
        print("---Ignore Mode---\n")

    mirror_data = {"elip": elip_mirror_data, "mode": mode_mirror_data}

    return mirror_data


def make_mirror_data(mirror_params):
    """ミラー作成関数(訓練データ/テストデータ)"""

    # 訓練データの作成
    print("Make train data!\n")
    train_mirror_data = _make_mirror_data(mirror_params["shape"], is_train=True)

    # テストデータの作成
    print("Make test data!\n")
    test_mirror_data = _make_mirror_data(mirror_params["shape"], is_train=False)

    return train_mirror_data, test_mirror_data


if __name__ == "__main__":
    # 実験器具フォルダ/ファイルの指定
    start_folder = "./run_instruments/"
    name_json_mirror_params = "params_making_mirror_data.json"

    with open(start_folder + name_json_mirror_params, "r") as f:
        mirror_params = json.load(f)

    result_folder = make_dir(mirror_params["dir_name"], is_time=True, pre_folder="result/")

    with open(result_folder + "/" + name_json_mirror_params, "w") as f:
        json.dump(mirror_params, f, indent=2)

    train_mirror_data, test_mirror_data = make_mirror_data(mirror_params)

    with open(result_folder + "/" + mirror_params["pkl_mirror_train"], "wb") as f:
        pickle.dump(train_mirror_data, f)
        print("output: ", mirror_params["pkl_mirror_train"])

    with open(result_folder + "/" + mirror_params["pkl_mirror_test"], "wb") as f:
        pickle.dump(test_mirror_data, f)
        print("output: ", mirror_params["pkl_mirror_test"])
