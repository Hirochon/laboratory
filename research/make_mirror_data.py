import numpy as np
import pickle
import json
from tqdm import tqdm

from file_operation import make_dir


def make_elip_param_list(elip_params, dat_num):
    """ミラー毎の楕円のパラメータをランダム作成関数"""

    ellipse_nums = []
    param_list = []

    for dat_i in range(dat_num):

        ellipse_nums.append(np.random.randint(elip_params['elip_num_min'], elip_params['elip_num_max']))    # 楕円の数

        init_elip_len_x_list = np.random.randint(elip_params['elip_len_x_min'], elip_params['elip_len_x_max'], size=ellipse_nums[dat_i])    # 楕円の長さ(横)
        init_elip_len_y_list = np.random.randint(elip_params['elip_len_y_min'], elip_params['elip_len_y_max'], size=ellipse_nums[dat_i])    # 楕円の長さ(縦)
        init_coord_x_list = np.random.randint(elip_params['coord_x_min'], elip_params['coord_x_max'], size=ellipse_nums[dat_i])     # 楕円の中心からのズレ(横)
        init_coord_y_list = np.random.randint(elip_params['coord_y_min'], elip_params['coord_y_max'], size=ellipse_nums[dat_i])     # 楕円の中心からのズレ(縦)
        init_theta_list = np.pi * np.random.rand(ellipse_nums[dat_i]) * 2 / elip_params['theta_rate']

        param_list.append(np.array([init_elip_len_x_list, init_elip_len_y_list, init_coord_x_list, init_coord_y_list, init_theta_list]))
        
    return param_list, ellipse_nums, elip_params['axis_x'], elip_params['axis_y'], elip_params['nx'], elip_params['ny']


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

                x_formula = X**2 / elip_len_x_list[k]**2    # 楕円の方程式のx部分
                y_formula = Y**2 / elip_len_y_list[k]**2    # 楕円の方程式のy部分

                if x_formula + y_formula <= 1:
                    elip_spot_mirror[j, i] += 1

    xx = np.linspace(-0.5, 0.5, nx) * axis_x    # -0.5〜0.5間でnx個に分けて、axis_xでブロードキャスト
    yy = np.linspace(-0.5, 0.5, ny) * axis_y    # -0.5〜0.5間でny個に分けて、axis_yでブロードキャスト
    x, y = np.meshgrid(xx, yy, indexing="ij")   # 2次元配列としてxとyをそれぞれ用意
    
    third_dim_elip_spot_mirror = {'m': 1, 'n': 1, 'x': x, 'y': y, 'z': elip_spot_mirror}
    
    return third_dim_elip_spot_mirror


def _make_mirror_data(surface_params, is_train):
    if is_train:
        elip_dat_num = surface_params['elip']['train_num']
    else:
        elip_dat_num = surface_params['elip']['test_num']
    
    elip_param_list, ellipse_nums, axis_x, axis_y, nx, ny = make_elip_param_list(surface_params['elip'], elip_dat_num)

    mirror_data = []
    for [elip_len_x_list, elip_len_y_list, coord_x_list, coord_y_list, theta_list], ellipse_num in tqdm(zip(elip_param_list, ellipse_nums), total=elip_dat_num):
        mirror_data.append(make_elip_spot_mirror(elip_len_x_list, elip_len_y_list, coord_x_list, coord_y_list, theta_list, axis_x, axis_y, ellipse_num, nx, ny))

    return mirror_data


def make_mirror_data(surface_params):
    """ミラー作成関数(訓練データ/テストデータ)"""

    print()
    print("Make train data!")
    print("out=", surface_params["pkl_surface_train"])
    train_mirror_data = _make_mirror_data(surface_params["shape"], is_train=True)

    print()
    print("Make test data!")
    print("out=", surface_params["pkl_surface_test"])
    test_mirror_data = _make_mirror_data(surface_params["shape"], is_train=True)

    return train_mirror_data, test_mirror_data


if __name__ == "__main__":
    start_folder = "./input_params_data/"
    name_json_surface_params = "params_making_mirror_data.json"

    with open(start_folder + name_json_surface_params, "r") as f:
        surface_params = json.load(f)

    folder_name = make_dir(surface_params['dir_name'], is_time=True, pre_folder='result/')

    with open(folder_name + '/' + name_json_surface_params, "w") as f:
        json.dump(surface_params, f, indent=2)

    train_mirror_data, test_mirror_data = make_mirror_data(surface_params)

    with open(folder_name + '/' + surface_params["pkl_surface_train"], "wb") as f:
        pickle.dump(train_mirror_data, f)
    with open(folder_name + '/' + surface_params["pkl_surface_test"], "wb") as f:
        pickle.dump(test_mirror_data, f)
