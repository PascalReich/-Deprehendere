import main_prepro as prepro
import evaluate_classifier as eval
import argparse
import logging
from random import randint
logger = logging.getLogger(__name__)


def main(input_directory, model_path, id_classifier_output_path, gen_classifier_output_path, batch_size, num_threads, is_yes, eval_only):
    if not eval_only:
        print("cropping images")
        random_int = str(randint(10000, 99999))
        random_int = "src/" + random_int
        prepro.main(input_directory, 180, random_int)
    labels = eval.main(input_directory, model_path, id_classifier_output_path, gen_classifier_output_path, batch_size,  num_threads, is_yes)
    print("Final Results:")
    print(labels)


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
    """parser.add_argument('--is-train', action='store_true', dest='is_train', default=False,
                        help='Flag to determine if train or evaluate')"""
    parser.add_argument('--override', action='store_true', dest='is_yes', default=False,
                        help='Flag to skip time verification')
    parser.add_argument('--eval-only', action='store_true', dest='eval', default=False,
                        help='Flag to skip time verification')

    args = parser.parse_args()

    main(input_directory=args.input_dir, model_path="/src/etc/20170511-185253/20170511-185253.pb",
         id_classifier_output_path=args.id_classifier_path,
         gen_classifier_output_path=args.gen_classifier_path,
         batch_size=128, num_threads=16, is_yes=args.is_yes, eval_only=args.eval)

    #TODO implelement eval only!, integrate file get into base script, test
