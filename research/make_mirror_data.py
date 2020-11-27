import numpy as np
import pickle
import json
from tqdm import tqdm
import itertools
from PIL import Image

from file_operation import make_dir


def make_elip_param_list(elip_params, data_num):
    """与えられた範囲内でパラメータをランダムに作成するための関数
    
    ミラーの作成にて楕円の形を作り出すため
    `params_making_mirror_data.json`で決められたパラメータの範囲指定に従って、
    パラメータをランダムで作成致します。

    Args:
        elip_params (dict{...}): elip内。`params_making_mirror_data.json`により範囲を指定されたパラメータの辞書群。
        data_num (int): ミラーの数。
    
    Returns:
        param_list (list[list[int]]): init_elip_len_x_list, init_elip_len_y_list, init_coord_x_list, init_coord_y_list, init_theta_listが入ったリスト
        ellipse_nums (list[int]): ミラー毎の楕円の数
        elip_params["axis_x"] (int): xに関して範囲-0.5〜0.5に対してどれだけ範囲を広くするか
        elip_params["axis_y"] (int): yに関して範囲-0.5〜0.5に対してどれだけ範囲を広くするか
        elip_params["nx"] (int): (-0.5〜0.5)*axis_xをどれだけ細分化するか
        elip_params["ny"] (int): (-0.5〜0.5)*axis_yをどれだけ細分化するか
        elip_params["noise"] (float): noiseの最大値を決める

    Note:
        np.random.randint(0, 1) → 0しか出力しない
        np.random.randint(1, 3) → 1と2を出力

    """

    ellipse_nums = []
    param_list = []
    elip_params["amp_degree"] = elip_params["max_degree"] - elip_params["min_degree"]

    if elip_params["max_degree"] < elip_params["min_degree"]:
        print("Error!!!")
        print("You must set min_degree <= max_degree\n")
        exit()

    for data_i in range(data_num):

        ellipse_nums.append(np.random.randint(elip_params["elip_num_min"], elip_params["elip_num_max"]))    # 楕円の数[vector]

        init_elip_len_x_list = np.random.randint(elip_params["elip_len_x_min"], elip_params["elip_len_x_max"], size=ellipse_nums[data_i])   # 楕円の長さ(横)[matrix]
        init_elip_len_y_list = np.random.randint(elip_params["elip_len_y_min"], elip_params["elip_len_y_max"], size=ellipse_nums[data_i])   # 楕円の長さ(縦)[matrix]
        init_coord_x_list = np.random.randint(elip_params["coord_x_min"], elip_params["coord_x_max"], size=ellipse_nums[data_i])            # 楕円の中心からのズレ(横)[matrix]
        init_coord_y_list = np.random.randint(elip_params["coord_y_min"], elip_params["coord_y_max"], size=ellipse_nums[data_i])            # 楕円の中心からのズレ(縦)[matrix]
        init_theta_list = np.random.rand(ellipse_nums[data_i]) * elip_params["amp_degree"] / 180 * np.pi + (elip_params["min_degree"] / 180 * np.pi)    # 楕円の回転角[matrix]

        param_list.append(np.array([init_elip_len_x_list, init_elip_len_y_list, init_coord_x_list, init_coord_y_list, init_theta_list]))
        
    return param_list, ellipse_nums, elip_params["axis_x"], elip_params["axis_y"], elip_params["nx"], elip_params["ny"], elip_params["amp"], elip_params["noise_rate"]


def make_elip_spot_mirror(elip_len_x_list, elip_len_y_list, coord_x_list, coord_y_list, theta_list, axis_x, axis_y, ellipse_num, nx, ny, amp, noise_rate):
    """ミラー毎に楕円(z)とx,yの値を入れていく
    
    ①楕円毎に与えられたパラメータに従って楕円を描いていく
    ②xとyについて対応する値を代入
    ③楕円の値(z)の画素値を拡大/縮小
    
    Args:
        elip_len_x_list (list[int]): 楕円のx方向に関しての半径
        elip_len_y_list (list[int]): 楕円のy方向に関しての半径
        coord_x_list (list[int]): xに関しての中心点からのズレ
        coord_y_list (list[int]): yに関しての中心点からのズレ
        theta_list (list[float]): 回転角度(radian)
        axis_x: xの0.5に対しての大きさ
        axis_y: yの0.5に対しての大きさ
        nx: -axis_x〜axis_xまでをnx個に細分化
        ny: -axis_y〜axis_yまでをny個に細分化
        noise: zへ追加するnoiseの最大値

    Returns:
        third_dim_elip_spot_mirror (dict{x: list[[float]], y: list[[float]], z: list[[float]]}): xとyとミラー(楕円)の値(高さ)
    
    """
    
    elip_spot_mirror = np.zeros([axis_x, axis_y])
    
    for k in range(ellipse_num):
        rotate = np.array([[np.cos(theta_list[k]), np.sin(theta_list[k])], [-np.sin(theta_list[k]), np.cos(theta_list[k])]])    # 回転行列の定義
        plus_minus = [-1, 1]
        zero_one_randomer = np.random.randint(0, 2)
        rated_amp = np.random.rand() * amp
        
        for j in range(axis_y):
            for i in range(axis_x):
                y = j + 1
                x = i + 1
                
                [X, Y] = np.dot([x - coord_x_list[k], y - coord_y_list[k]], rotate)     # 行列の積

                x_formula = X**2 / elip_len_x_list[k]**2    # 楕円の方程式のxとa部分
                y_formula = Y**2 / elip_len_y_list[k]**2    # 楕円の方程式のyとb部分

                # # 楕円の条件を通す
                # if x_formula + y_formula <= 1:
                #     elip_spot_mirror[j, i] += amp * np.exp(- x_formula - y_formula)    # ガウス分布*amplitude(最大値)

                # 楕円の条件を無くす！ただの確率密度関数(ちょっと違うけど)となる。
                elip_spot_mirror[j, i] += plus_minus[zero_one_randomer] * rated_amp * np.exp(- x_formula - y_formula)

    xx = np.linspace(-0.5, 0.5, nx) * axis_x    # -0.5〜0.5間でnx個に分けて、axis_xでブロードキャスト
    yy = np.linspace(-0.5, 0.5, ny) * axis_y    # -0.5〜0.5間でnx個に分けて、axis_yでブロードキャスト
    x, y = np.meshgrid(xx, yy, indexing="ij")   # 2次元配列としてxとyをそれぞれ用意

    if (nx != axis_x) or (ny != axis_y):    # 想定した枠と分割する個数が異なる場合に、画素を拡大/縮小させることにより、対応。
        Imaged_elip_spot_mirror = Image.fromarray(elip_spot_mirror)
        resized_elip_spot_mirror = np.asarray(Imaged_elip_spot_mirror.resize((nx, ny)))
        third_dim_elip_spot_mirror = {"x": x, "y": y, "z": resized_elip_spot_mirror}
    else:
        third_dim_elip_spot_mirror = {"x": x, "y": y, "z": elip_spot_mirror}

    noise = amp * noise_rate
    third_dim_elip_spot_mirror["z"] += noise * np.random.randn(nx, ny)

    return third_dim_elip_spot_mirror


def make_mode_spot_mirror(nx=16, ny=16, m=1, n=1, axis_x=1., axis_y=1., amp=1., noise_rate=0):
    """ミラー毎に線上のシマシマ作成&代入
    """

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
    noise = amp * noise_rate
    oi += noise * np.random.randn(nx, ny)
    return x, y, np.real(oi)


def _make_mirror_data(mirror_params, is_train):
    """ミラー作成関数(elipとmodeのデータを作ることに集中する関数)
    
    楕円作成関数とmode作成関数を別々に実行。

    Args:
        mirror_params (common: {...}, elip: {...}, mode: {...}): shape内。詳しくは`params_making_mirror_data.json`を参照。
        is_train (boolean): 訓練データですか？

    Returns:
        mirror_data (dict{elip: list[dict{x:~,y:~,z:~,info:{...}]}, mode: list[dict{m:~,n:~,z~,x~,y~}]}): elipとmodeでそれぞれ作成したミラーのデータ
    
    """

    ##############
    # 共通の処理 #
    ##############
    mirror_params["elip"]["axis_x"] = mirror_params["mode"]["axis_x"] = mirror_params["common"]["axis_x"]
    mirror_params["elip"]["axis_y"] = mirror_params["mode"]["axis_y"] = mirror_params["common"]["axis_y"]
    mirror_params["elip"]["nx"] = mirror_params["mode"]["nx"] = mirror_params["common"]["nx"]
    mirror_params["elip"]["ny"] = mirror_params["mode"]["ny"] = mirror_params["common"]["ny"]
    mirror_params["elip"]["amp"] = mirror_params["mode"]["amp"] = mirror_params["common"]["amp"]
    mirror_params["elip"]["noise_rate"] = mirror_params["mode"]["noise_rate"] = mirror_params["common"]["noise_rate"]

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
        elip_param_list, ellipse_nums, axis_x, axis_y, nx, ny, amp, noise_rate = make_elip_param_list(mirror_params["elip"], elip_data_num)

        # make_elip_param_listで作成したパラメータを元に楕円型ミラーデータを作成
        for [elip_len_x_list, elip_len_y_list, coord_x_list, coord_y_list, theta_list], ellipse_num in tqdm(zip(elip_param_list, ellipse_nums), total=elip_data_num):
            elip_spot_mirror = make_elip_spot_mirror(elip_len_x_list, elip_len_y_list, coord_x_list, coord_y_list, theta_list, axis_x, axis_y, ellipse_num, nx, ny, amp, noise_rate)
            elip_spot_mirror_params = {"elip_len_x_list": elip_len_x_list, "elip_len_y_list": elip_len_y_list,
                                       "coord_x_list": coord_x_list, "coord_y_list": coord_y_list, "theta_list": theta_list,
                                       "axis_x": axis_x, "axis_y": axis_y, "ellipse_num": ellipse_num, "nx": nx, "ny": ny, "amp": amp, "noise_rate": noise_rate}
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
                                                axis_y=mode_params["axis_y"], amp=mode_params["amp"], noise_rate=mode_params["noise_rate"])
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
    """ミラー作成関数(trainとtestのデータを作ることに集中する関数)
    
    trainかtestかを分けるためだけの関数。
    これにより同じ処理を複数回書かずに済む。

    引数と戻り値は`_make_mirror_data`とほぼ同じなので省略。
    
    """

    # 訓練データの作成
    print("Make train data!\n")
    mirror_train_data = _make_mirror_data(mirror_params["shape"], is_train=True)

    # テストデータの作成
    print("Make test data!\n")
    mirror_test_data = _make_mirror_data(mirror_params["shape"], is_train=False)

    return mirror_train_data, mirror_test_data


def main_make_mirror_data():
    """ミラー作成の基幹となる場所。主にファイルの読み込み/書き込み。

    make_data.pyと連携して`make_detec_data.py`と一括で実行する際に通る道。

    Returns:
        result_folder (str): 作成したフォルダ名

    """

    start_folder = "./run_instruments/"
    name_json_mirror_params = "params_making_mirror_data.json"

    with open(start_folder + name_json_mirror_params, "r") as f:
        mirror_params = json.load(f)

    result_folder = make_dir(mirror_params["dir_name"], is_time=True, pre_folder="result/") + "/"

    with open(result_folder + name_json_mirror_params, "w") as f:
        json.dump(mirror_params, f, indent=2)

    mirror_train_data, mirror_test_data = make_mirror_data(mirror_params)

    with open(result_folder + mirror_params["pkl_mirror_train"], "wb") as f:
        pickle.dump(mirror_train_data, f)
        print("output: ", mirror_params["pkl_mirror_train"])
    with open(result_folder + mirror_params["pkl_mirror_test"], "wb") as f:
        pickle.dump(mirror_test_data, f)
        print("output: ", mirror_params["pkl_mirror_test"], "\n")

    print("Build {}\n".format(result_folder))

    return result_folder


if __name__ == "__main__":
    """ミラーのデータのみ作る際に使う(デバッグ用)"""

    start_folder = "./run_instruments/"
    name_json_mirror_params = "params_making_mirror_data.json"

    with open(start_folder + name_json_mirror_params, "r") as f:
        mirror_params = json.load(f)

    result_folder = make_dir(mirror_params["dir_name"], is_time=True, pre_folder="result/") + "/"

    with open(result_folder + name_json_mirror_params, "w") as f:
        json.dump(mirror_params, f, indent=2)

    mirror_train_data, mirror_test_data = make_mirror_data(mirror_params)

    with open(result_folder + mirror_params["pkl_mirror_train"], "wb") as f:
        pickle.dump(mirror_train_data, f)
        print("output: ", mirror_params["pkl_mirror_train"])
    with open(result_folder + mirror_params["pkl_mirror_test"], "wb") as f:
        pickle.dump(mirror_test_data, f)
        print("output: ", mirror_params["pkl_mirror_test"], "\n")

    print("Build {}\n".format(result_folder))
