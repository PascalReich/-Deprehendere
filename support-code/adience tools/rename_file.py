import os
import sys

def filter():

    with open("C:/Users/foggy/facialRecognition/adienceDB/fold_2_data.txt", "r") as ins:
        array = []
        for line in ins:
            array.append(line)

    del array[:1]

    split_array = []

    for x in array:
        split_array.append(x.split())

    num = 0
    to_be_deleted = []

    for y in split_array:
        del y[4]
        del y[3]
        del y[4:]

        """if split_array.index(y) != num:
            # sys.exit(1)
            print(split_array.index(y), num, type(split_array.index(y)), type(num))
        """

        if y[3].isnumeric() or y[3] == "u":
            # print("no gender:", y, y[3])
            to_be_deleted.append(num)

        num += 1


    # print(to_be_deleted, len(split_array))

    to_be_deleted = to_be_deleted[::-1]

    for z in to_be_deleted:
        # print(split_array[z])
        del split_array[z]

    # print(split_array)
    return split_array

def make_file_names(array):
    final_array = []
    for i in array:
        file_src = "coarse_tilt_aligned_face." + i[2] + "." + i[1]
        file_path = "C:/Users/foggy/facialRecognition/adienceDB/faces/" + i[0] + "/" + file_src
        if i[3] == "f":
            target_name = "female"
        elif i[3] == "m":
            target_name = "male"
        else:
            sys.exit(i[3] + "IS NOT STANDARD")

        target_name = "C:/Users/foggy/facialRecognition/adienceDB/faces/sorted/" + target_name + "/" + target_name
        int_array = [file_path, target_name]
        final_array.append(int_array)
    # print(final_array)
    return final_array


def rename(files):
    number = 30000
    for file in files:
        # print(os.path.isfile(file[0]))
        try:
            os.rename(file[0], file[1] + "_" + str(number) + ".jpg")
        except FileNotFoundError:
            print("file not found")
            continue
        number += 1


def main():
    info_array = filter()
    files = make_file_names(info_array)
    # print(files)
    rename(files)


if __name__ == '__main__':
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model-path', type=str, action='store', dest='model_path',
                        help='Path to model protobuf graph')
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir',
                        help='Input path of data to train on')
    
    args = parser.parse_args()
    """
    main()