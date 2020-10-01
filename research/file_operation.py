import os
from datetime import datetime


def make_dir(folder_name, is_time, pre_folder=""):
    """入力したデータや出力データを格納するためのフォルダ作成関数
    
    Args:
        folder_name (str): フォルダ名
        is_time (boolean): フォルダ名の前に時間を入れるか否か
        pre_folder (str): 作成するフォルダ上の補完
        
    Returns:
        pre_folder + strtime + folder_name (str): 作成したフォルダ名

    """

    if is_time:
        strtime = datetime.now().strftime("%Y_%m%d_%H%M%S_")
    else:
        strtime = ""

    os.makedirs(pre_folder + strtime + folder_name)
    print("\"" + pre_folder + strtime + folder_name + "\" folder is created!\n")

    return pre_folder + strtime + folder_name


if __name__ == "__main__":
    """デバッグ"""
    
    make_dir("YesHoge", True)
    make_dir("NoHoge", False, "_No")
