from make_mirror_data import main_make_mirror_data
from make_detec_data import main_make_detec_data

if __name__ == "__main__":
    print("===Start Create Mirror Data===")
    result_folder = main_make_mirror_data()
    print("===Start Create Detec Data===")
    main_make_detec_data(result_folder)
