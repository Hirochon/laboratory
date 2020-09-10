import os
from datetime import datetime


def make_dir(folder_name, is_time, pre_folder=""):
    """
    入力したデータや出力データを格納するためのフォルダ作成関数
    第1引数: フォルダ名
    第2引数: フォルダ名の前に時間を入れるか否か
    第3引数: 作成するフォルダ上の補完
    返り値: 作成したフォルダ名
    """

    if is_time:
        strtime = datetime.now().strftime("%Y_%m%d_%H%M%S_")
    else:
        strtime = ""

    os.makedirs(pre_folder + strtime + folder_name)
    print()
    print("\"" + pre_folder + strtime + folder_name + "\" is created!")

    return pre_folder + strtime + folder_name


if __name__ == "__main__":
    make_dir("YesHoge", True)
    make_dir("NoHoge", False, "_No")
