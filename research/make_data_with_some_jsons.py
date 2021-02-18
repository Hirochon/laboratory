import shutil
import make_data_mk


files = [   
            # "/Users/noda/kogaken/laboratory/research/run_instruments/params_making_mirror_data_20201021_001.json",
            # "/Users/noda/kogaken/laboratory/research/run_instruments/params_making_mirror_data_20201021_002.json",
            # "/Users/noda/kogaken/laboratory/research/run_instruments/params_making_mirror_data_20201127_001.json",
            # "/Users/noda/kogaken/laboratory/research/run_instruments/params_making_mirror_data_20201127_002.json",
            "/Users/noda/kogaken/laboratory/research/run_instruments/params_making_mirror_data_20201204_001.json"
        ]

for file in files:

    src  = file
    copy = "/Users/noda/kogaken/laboratory/research/run_instruments/params_making_mirror_data.json" 
    shutil.copy(src, copy)

    make_data_mk.main_make_data()