import numpy as np
import pickle
import json
from tqdm import tqdm


def load_mirror_data(mirror_data):
    elip_mirror_data = mirror_data["elip"]
    mode_mirror_data = mirror_data["mode"]

    x = []
    y = []
    z = []
    shape = []

    if len(elip_mirror_data) > 0:
        for row in elip_mirror_data:
            x.append([row["x"]])
            y.append([row["y"]])
            z.append([row["z"]])
            shape.append([row["elip_len_x_list"], row["elip_len_y_list"], row["coord_x_list"], row["coord_y_list"],
                          row["theta_list"], row["axis_x"], row["axis_y"], row["ellipse_num"], row["nx"], row["ny"]])

    if len(mode_mirror_data) > 0:
        for row in mode_mirror_data:
            x.append([row["x"]])
            y.append([row["y"]])
            z.append([row["z"]])
            shape.append((row["m"], row["n"]))
        
    np_x = np.array(x)
    np_y = np.array(y)
    np_z = np.array(z)
    np_shape = np.array(shape)
    
    return np_x, np_y, np_z, np_shape


def _make_detec_data(detec_params, mirror_data):
    (x, y, z, shape) = load_mirror_data(mirror_data)

    for i in tqdm(range(x.shape)):
        print(i)

    exit()

    detec_data = []

    return detec_data


def make_detec_data(detec_params, train_mirror_data, test_mirror_data):
    print("Make train data!\n")
    train_detec_data = _make_detec_data(detec_params, train_mirror_data)

    print("Make test data!\n")
    test_detec_data = _make_detec_data(detec_params, test_mirror_data)

    return train_detec_data, test_detec_data


if __name__ == "__main__":
    start_folder = "./run_instruments/"
    name_json_mirror_params = "params_making_mirror_data.json"
    name_json_detec_params = "params_making_detec_data.json"

    # ハードコーディング警察(｡･_･｡)
    result_folder = "result/" + "2020_0917_155153_add_elip_func"

    # detectorのjsonパラメータをロード
    with open(start_folder + name_json_detec_params, "r") as f:
        detec_params = json.load(f)

    # detectorのjsonパラメータを結果にdump
    with open(result_folder + "/" + name_json_detec_params, "w") as f:
        json.dump(detec_params, f, indent=2)

    # mirrorのパラメーターjsonからmirrorのデータ名をロード
    with open(start_folder + name_json_mirror_params, "r") as f:
        mirror_params = json.load(f)

    # mirrorのデータをロード
    with open(result_folder + "/" + mirror_params["pkl_mirror_train"], "rb") as f:
        train_mirror_data = pickle.load(f)
    with open(result_folder + "/" + mirror_params["pkl_mirror_test"], "rb") as f:
        test_mirror_data = pickle.load(f)

    train_detec_data, test_detec_data = make_detec_data(detec_params, train_mirror_data, test_mirror_data)

    with open(result_folder + "/" + detec_params["pkl_detec_train"], "wb") as f:
        pickle.dump(train_detec_data, f)
        print("output: ", detec_params["pkl_detec_train"])

    with open(result_folder + "/" + detec_params["pkl_detec_test"], "wb") as f:
        pickle.dump(test_detec_data, f)
        print("output: ", detec_params["pkl_detec_test"])
