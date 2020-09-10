import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from tqdm import tqdm


def make_param_list(mirror_params):
    """ミラー毎の楕円のパラメータをランダム作成関数"""

    ellipse_nums = []
    param_list = []

    for dat_i in range(mirror_params['dat_num']):

        ellipse_nums.append(np.random.randint(mirror_params['elip_num_min'], mirror_params['elip_num_max']))

        init_elip_len_x_list = np.random.randint(mirror_params['elip_len_x_min'], mirror_params['elip_len_x_max'], size=ellipse_nums[dat_i])
        init_elip_len_y_list = np.random.randint(mirror_params['elip_len_y_min'], mirror_params['elip_len_y_max'], size=ellipse_nums[dat_i])
        init_coord_x_list = np.random.randint(mirror_params['coord_x_min'], mirror_params['coord_x_max'], size=ellipse_nums[dat_i])
        init_coord_y_list = np.random.randint(mirror_params['coord_y_min'], mirror_params['coord_y_max'], size=ellipse_nums[dat_i])
        init_theta_list = np.pi * np.random.rand(ellipse_nums[dat_i]) * 2 / mirror_params['theta_rate']

        param_list.append(np.array([init_elip_len_x_list, init_elip_len_y_list, init_coord_x_list, init_coord_y_list, init_theta_list]))
        
    return param_list, ellipse_nums, mirror_params['axis_x'], mirror_params[
        'axis_y'], mirror_params['dat_num'], mirror_params['nx'], mirror_params['ny']


def make_elip_spot_mirror(elip_len_x_list, elip_len_y_list, coord_x_list, coord_y_list, theta_list, axis_x, axis_y, ellipse_num, nx, ny):
    """楕円毎に与えられたパラメータに従って楕円を描いていく"""
    
    elip_spot_mirror = np.zeros([axis_x, axis_y])
    
    for k in range(ellipse_num):
        rotate = np.array([[np.cos(theta_list[k]), np.sin(theta_list[k])], [
            -np.sin(theta_list[k]), np.cos(theta_list[k])]])
        
        for j in range(axis_y):
            for i in range(axis_x):
                y = j + 1
                x = i + 1
                
                [X, Y] = np.dot([x - coord_x_list[k], y - coord_y_list[k]], rotate)

                x_formula = X**2 / elip_len_x_list[k]**2
                y_formula = Y**2 / elip_len_y_list[k]**2

                if x_formula + y_formula <= 1:
                    elip_spot_mirror[j, i] += 1

    xx = np.linspace(-0.5, 0.5, nx) * axis_x
    yy = np.linspace(-0.5, 0.5, ny) * axis_y
    x, y = np.meshgrid(xx, yy, indexing="ij")
    
    # mとnはバグがおきない様に入れてるだけ。
    third_dim_elip_spot_mirror = {'m': 1, 'n': 1, 'x': x, 'y': y, 'z': elip_spot_mirror}
    
    return third_dim_elip_spot_mirror


def make_teacher_data(input_noda):
    """ミラー作成関数(訓練データ)"""

    dat = []
    print("make_teacher_data")
    print("out=", input_noda["pkl_surface_teacher"])

    param_list, ellipse_nums, axis_x, axis_y, dat_num, nx, ny = make_param_list(input_noda)

    for [elip_len_x_list, elip_len_y_list, coord_x_list, coord_y_list, theta_list], ellipse_num in tqdm(zip(param_list, ellipse_nums), total=dat_num):
        dat.append(make_elip_spot_mirror(elip_len_x_list, elip_len_y_list,
                                         coord_x_list, coord_y_list, theta_list,
                                         axis_x, axis_y, ellipse_num, nx, ny))

    with open(input_noda["pkl_surface_teacher"], 'wb') as f:
        pickle.dump(dat, f)


if __name__ == "__main__":
    start_folder = "./input_params_data/"
    name_json_surface_params = "params_making_mirror_data.json"

    with open(start_folder + name_json_surface_params, "r") as f:
        surface_params = json.load(f)

    folder_name = make_dir(surface_params['dir_name'], is_time=True, pre_folder='result/')

    with open(folder_name + '/' + name_json_surface_params, "w") as f:
        json.dump(surface_params, f, indent=2)

    with open('pkl_surf_teacher_noda.pkl', 'rb') as f:
        noda = pickle.load(f)

    # roop = ['x', 'y', 'z']
    # for j in range(len(noda)):
    #     fig = plt.figure(figsize=(15, 15))
    #     for i, v in enumerate(roop):
    #         fig.add_subplot(1, 3, i + 1)
    #         plt.imshow(noda[j][roop[i]])
    #     plt.show()
