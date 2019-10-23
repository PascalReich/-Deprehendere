import SE_code.main_prepro as prepro
import SE_code.evaluate_classifier as eval
import argparse
import time
import logging
import os
import shutil
import json
from random import randint
logger = logging.getLogger(__name__)


def main(input_directory, model_path, id_classifier_output_path, gen_classifier_output_path, age_classifier_output_path, batch_size, num_threads, is_yes, eval_only):
    start_time = time.time()
    if not eval_only:
        print("cropping images")
        random_int = str(randint(10000, 99999))
        random_int = "src/static/" + random_int
        cropped_img = prepro.main(input_directory, 180, random_int)
        input_directory = random_int
    print("starting eval")
    labels = eval.main(cropped_img, model_path, id_classifier_output_path, gen_classifier_output_path, age_classifier_output_path, batch_size,  num_threads, is_yes)
    """print("Final Results:")
    print(labels)"""
    #if not eval_only:
        #shutil.rmtree(input_directory)
    
    logger.info('Completed in {} seconds'.format(time.time() - start_time))
    return labels, input_directory


def diff(li1, li2):
    return list(set(li1) - set(li2))


def isin(array, thing):
    for i in array:
        if i == thing:
            return True
    return False


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    """parser.add_argument('--model-path', type=str, action='store', dest='model_path',
                        help='Path to model protobuf graph')"""
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir',
                        help='Input path of data to train on')
    """parser.add_argument('--batch-size', type=int, action='store', dest='batch_size',
                        help='Number of images are processed per iteration', default=128)
    parser.add_argument('--num-threads', type=int, action='store', dest='num_threads', default=16,
                        help='Number of threads to utilize for queue')
    parser.add_argument('--num-epochs', type=int, action='store', dest='num_epochs', default=3,
                        help='Path to output trained classifier model')
    parser.add_argument('--split-ratio', type=float, action='store', dest='split_ratio', default=0.7,
                        help='Ratio to split train/test dataset')
    parser.add_argument('--min-num-images-per-class', type=int, action='store', default=10,
                        dest='min_images_per_class', help='Minimum number of images per class')
    parser.add_argument('--training-path', type=str, action='store', dest='train_classifier_path',
                        help='Path to output trained classifier model')"""
    parser.add_argument('--id-classifier-path', type=str, action='store', dest='id_classifier_path',
                        help='Path to input trained classifier model of identity')
    parser.add_argument('--gender-classifier-path', type=str, action='store', dest='gen_classifier_path',
                        help='Path to input trained classifier model of gender')
    parser.add_argument('--age-classifier-path', type=str, action='store', dest='age_classifier_path',
                        help='Path to input trained classifier model of age')
    """parser.add_argument('--is-train', action='store_true', dest='is_train', default=False,
                        help='Flag to determine if train or evaluate')"""
    parser.add_argument('--override', action='store_true', dest='is_yes', default=False,
                        help='Flag to skip time verification')
    parser.add_argument('--eval-only', action='store_true', dest='eval', default=False,
                        help='Flag to skip time verification')

    full_args = parser.parse_args()

    """CallMain(full_args)

    watch(full_args)"""

    print(full_args.input_dir)

    base_dir = os.listdir(full_args.input_dir)

    print("Waiting for input")

    args = full_args

    while True:
        time.sleep(1)
        current_dir = os.listdir(full_args.input_dir)
        if base_dir != current_dir:
            diffs = diff(current_dir, base_dir)
            try:
                if diffs[0].split('_')[1] == 'closed': 
                    base_dir = current_dir;
                    continue;
            except IndexError:
                print('index error reeeeeeee')
            print(base_dir, current_dir, len(diffs))
            if len(diffs) is 1:
                diffs = diffs[0]
                if isin(current_dir, diffs):
                    folder = full_args.input_dir + "/" + diffs
                    print(os.path.isdir(folder), folder)
                    if os.path.isdir(folder):
                        a = 0
                        if '.DS_Store' in os.listdir(folder):
                            os.remove(folder + '/.DS_Store')
                        for sub_dir in os.listdir(folder):
                            for img in os.listdir(folder + "/" + sub_dir):
                                a += 1
                        print(os.listdir(folder + "/" + sub_dir))
                        labels, crop = main(input_directory=folder, model_path="/src/etc/20170511-185253/20170511-185253.pb",
                                      id_classifier_output_path=args.id_classifier_path,
                                      gen_classifier_output_path=args.gen_classifier_path,
                                      age_classifier_output_path=args.age_classifier_path,
                                      batch_size=128, num_threads=1, is_yes=args.is_yes, eval_only=args.eval)
                        out = folder + "/results.json"
                        crop = crop.split('/')[1]
                        labels.append(crop)
                        with open(out, 'w') as f:
                            json.dump(labels, f)
                            
                        nFold = folder.split('_')[0] + '_closed'
                        
                        print(nFold)

                        os.rename(folder, nFold)
                        
                        base_dir = current_dir;

            else:
                print("ignored because != 1 changes")
            base_dir = current_dir

    # main(full_args)

    # TODO implement eval only!, integrate file get into base script, test
