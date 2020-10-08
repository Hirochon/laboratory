"""
verinfo = ソースコードや入力、出力のバックアップ
ver1.1 2018.11.18 check for ITC27
ver1.0 2018.11.13 Developing codes
by H.Tsuchiya (NIFS)
"""

import subprocess as sb
import json
import time

#公開する関数を指定する
__all__ = ['COPY_FILES',\
           'MOVE_FILES',\
           'MAKE_DIR',\
           'READ_INPUT_DATA',\
           'SAVE_JSON_DATA'
           ];

def MAKE_DIR(folder):
    cmd = "mkdir "+folder
    sb.Popen(cmd,  shell=True, stdout=sb.DEVNULL)
    time.sleep(1)

def COPY_FILES(folder,extention):
    print("")
    for text in extention:
        cmd = "copy *."+text+" "+folder+"\\"
        print(cmd)
        sb.Popen(cmd,  shell=True, stdout=sb.DEVNULL)
        time.sleep(1)

def MOVE_FILES(folder,extention):
    print("")
    for text in extention:
        cmd = "move *."+text+" "+folder+"\\"
        print(cmd)
        sb.Popen(cmd,  shell=True, stdout=sb.DEVNULL)
        time.sleep(1)

#===============================================================================
def READ_INPUT_DATA(filename):
    f = open(filename, 'r')
    input_data = json.load(f) #JSON形式で読み込む
    return input_data

def SAVE_JSON_DATA(filename,out_data):
    f = open(filename, 'w')
    json.dump(out_data,f,indent=4)
    return
#===============================================================================

def test():
    s = r'{"C": "あ", "A": {"i": 1, "j": 2}, "B": [{"X": 1, "Y": 10}, {"X": 21, "Y": 20}]}'
    json_dict = json.loads(s)
    print(json_dict)
    file_json = "test_j.json"
    SAVE_JSON_DATA(file_json,json_dict)
    ss = READ_INPUT_DATA(file_json)
    print(ss)
    MAKE_DIR("test")
    COPY_FILES("test",["json","txt1","txt2"])
    MOVE_FILES("test",["json","txt2","txt3"])

if __name__ == "__main__":
    test()
