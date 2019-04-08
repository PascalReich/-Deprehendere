import os
import sys

def main():
    file_array = []
    age_classes = [[0, 3], [4, 12], [12, 18], [18, 25], [26, 35], [36, 45], [46, 55], [56, 65], [66, 75], [76, 85], [86, 95], [96, False]]
    db_input = "C:/Users/foggy/facialRecognition/UKface/"
    # makedirs(db_input, age_classes)
    i = 0
    for folder in os.listdir(db_input):
        for file in os.listdir(os.path.join(db_input, folder)):
            full_file = os.path.join(db_input, folder, file)
            try:
                age = int(file.split('_')[0])
            except ValueError:
                age = False
                print(file)

            if age:
                for age_range in age_classes:
                    if isinrange(age, age_range[0], age_range[1]):
                        img_age_range = age_range
                        if img_age_range[1]:
                            name = str(img_age_range[0]) + "-" + str(img_age_range[1])
                        else:
                            name = str(img_age_range[0]) + "+"
                        path = db_input + name + "/" + name + "_" + str(i) + ".jpg"
                        if not os.path.exists(full_file):
                            print("error", full_file, path)
                        else:
                            os.rename(full_file, path)
                        i += 1


def makedirs(db_input, age_classes):
    for img_age_range in age_classes:
        if img_age_range[1]:
            name = str(img_age_range[0]) + "-" + str(img_age_range[1])
        else:
            name = str(img_age_range[0]) + "+"
        os.makedirs(db_input+name)
    sys.exit()


def isinrange(num, param1, param2):
    if param2:
        return param1 <= num <= param2
    else:
        return param1 <= num

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