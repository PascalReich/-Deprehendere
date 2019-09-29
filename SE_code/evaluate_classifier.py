import argparse
import logging
import os
import decimal
try:
    import _pickle as pickle
    import gc
    print("using cPickle")
except:
    import pickle
    print("using pickle")
import sys
import time
import math

import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow.python.platform import gfile

from SE_code.lfw_input import filter_dataset, split_dataset, get_dataset
from SE_code import lfw_input
import json

logger = logging.getLogger(__name__)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main(images, model_path, id_classifier_output_path, gen_classifier_output_path, age_classifier_output_path, batch_size, num_threads, override):

    results_out_path = images
    tf.logging.set_verbosity(tf.logging.ERROR)


    """
    Loads images from :param input_dir, creates embeddings using a model defined at :param model_path, and trains
     a classifier outputted to :param output_path

    :param input_directory: Path to directory containing pre-processed images
    :param model_path: Path to protobuf graph file for facenet model
    :param id_classifier_output_path: Path to write pickled classifier
    :param batch_size: Batch size to create embeddings
    :param num_threads: Number of threads to utilize for queuing
    :param num_epochs: Number of epochs for each image
    :param min_images_per_labels: Minimum number of images per class
    :param split_ratio: Ratio to split train/test dataset
    :param is_train: bool denoting if training or evaluate
    """

    classifier_filenames = [None, None, None]

    """if is_train:
        if not train_path:
            sys.exit('no train classifier')
    else:
"""
    if id_classifier_output_path is not None:
        classifier_filenames[0] = id_classifier_output_path
    if gen_classifier_output_path is not None:
        classifier_filenames[1] = gen_classifier_output_path
    if age_classifier_output_path is not None:
        classifier_filenames[2] = age_classifier_output_path
    if not classifier_filenames:
        sys.exit('no classifier')

    start_time = time.time()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        test_set = images

        """, test_set = _get_test_and_train_set(input_directory, min_num_images_per_label=min_images_per_labels,
                                                      split_ratio=split_ratio)"""
        """if is_train:
            images, labels, class_names, image_paths = _load_images_and_labels(train_set, image_size=160, batch_size=batch_size,
                                                                               num_threads=num_threads, num_epochs=num_epochs,
                                                                               random_flip=True, random_brightness=True,
                                                                               random_contrast=True)
        else:"""
        images, labels, image_path = _load_images_and_labels(test_set, image_size=160, batch_size=batch_size,
                                                 num_threads=num_threads, num_epochs=1)

        _load_model(model_filepath=model_path)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        emb_array = _create_embeddings(embedding_layer, images, labels, images_placeholder,
                                       phase_train_placeholder, sess,
                                       batch_size, override, coord, threads)

        try:
            coord.request_stop()
            coord.join(threads=threads)
            sess.close()
        except:
            print("error in threds/sess closing")

        logger.info('Created {} embeddings'.format(len(emb_array)))

        """if is_train:
            _train_and_save_classifier(emb_array, label_array, class_names, train_path)

        else:"""

        returns = _evaluate_classifier(emb_array, classifier_filenames, results_out_path, image_path)

        time_dif = time.time() - start_time
        if time_dif > 3600:
            time_dif /= 3600
            time_dif = str(time_dif) + " hours"
        elif time_dif > 60:
            time_dif /= 60
            time_dif = str(time_dif) + " minutes"
        else:
            time_dif = str(time_dif) + " seconds"
    logger.info('Completed in {}'.format(time_dif))
    return returns


"""def _get_test_and_train_set(input_dir, min_num_images_per_label, split_ratio=0.7):
    
    Load train and test dataset. Classes with < :param min_num_images_per_label will be filtered out.
    :param input_dir:
    :param min_num_images_per_label:
    :param split_ratio:
    :return:
    
    dataset = get_dataset(input_dir)
    dataset = filter_dataset(dataset, min_images_per_label=min_num_images_per_label)
    train_set, test_set = split_dataset(dataset, split_ratio=split_ratio)

    return train_set, test_set
"""


def _load_images_and_labels(dataset, image_size, batch_size, num_threads, num_epochs, random_flip=False,
                            random_brightness=False, random_contrast=False):
    # class_names = [cls.name for cls in dataset]
    print(dataset)
    print("loading")
    dataset = get_dataset(dataset)
    dataset = filter_dataset(dataset, min_images_per_label=0)
    image_paths, labels = lfw_input.get_image_paths_and_labels(dataset)
    images, labels = lfw_input.read_data(image_paths, labels, image_size, batch_size, num_epochs, num_threads,
                                         shuffle=False, random_flip=random_flip, random_brightness=random_brightness,
                                         random_contrast=random_contrast)
    return images, labels, image_paths


def _load_model(model_filepath):
    """
    Load frozen protobuf graph
    :param model_filepath: Path to protobuf graph
    :type model_filepath: str
    """
    model_exp = os.path.expanduser(model_filepath)
    if os.path.isfile(model_exp):
        logging.info('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        logger.error('Missing model file. Exiting')
        sys.exit(-1)


def _create_embeddings(embedding_layer, images, labels, images_placeholder, phase_train_placeholder, sess,
                       batch_size, override, coord, threads):
    """
    Uses model to generate embeddings from :param images.
    :param embedding_layer:
    :param images:
    :param labels:
    :param images_placeholder:
    :param phase_train_placeholder:
    :param sess:
    :return: (tuple): image embeddings and labels
    """
    emb_array = None
    label_array = None

    """ok, it_total = Estimate.estimate(len(), batch_size, num_epochs, override, train)
    if not ok:
        coord.request_stop()
        coord.join(threads)
        sess.close()
        sys.exit()
    if os.path.exists("/src/embeddings.pkl"):
        with open("/src/embeddings.pkl", 'rb') as inF:
            print("using existing embeddings")
            emb_array = pickle.load(inF)
    else:

        str(math.floor((i/it_total)*100)) + """
    # batch_images, batch_labels = sess.run([images, labels])

    try:
        i = 0
        while True:
            batch_images, batch_labels = sess.run([images, labels])
            emb = sess.run(embedding_layer, feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})
            emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb
            logger.info('% - Processed iteration {} batch of size: {}'.format(i, len(batch_labels)))
            i += 1
    except tf.errors.OutOfRangeError:
        pass
        print(85294)
    """print("pickling embeddings")
    with open("/src/embeddings.pkl", 'wb') as outfile:
        gc.disable()
        pickle.dump(emb_array, outfile)
        gc.enable()
    print("pickled embeddings")"""
    return emb_array


"""def _train_and_save_classifier(emb_array, label_array, class_names, classifier_filename_exp):
    logger.info('Training Classifier at')
    model = SVC(kernel='linear', probability=True, verbose=False)
    model.fit(emb_array, label_array)

    with open(classifier_filename_exp, 'wb') as outfile:
        gc.disable()
        pickle.dump((model, class_names), outfile)
        gc.enable()
    logging.info('Saved classifier model to file "%s"' % classifier_filename_exp)"""


def _evaluate_classifier(emb_array, classifier_filename, images, paths):
    logger.info('Evaluating classifier on {} images'.format(len(emb_array)))
    results = []
    image_paths = []
    print(images)
    for i in range(len(emb_array)):
        image_paths.append("None")

    try:
        with open(classifier_filename[0], 'rb') as IDf:
            model, class_names = pickle.load(IDf)

            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            for i in range(len(best_class_indices)):
                results.append([i, paths[i], class_names[best_class_indices[i]], best_class_probabilities[i]])
                # print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                # print(image_paths[i])

                # print(str(i) + image_paths[i] + )
                # print(label_array)

            """
            right_wrong_array = np.equal(best_class_indices, label_array)
            exceptions = 0
            for x in range(0, len(right_wrong_array)):
                if int(right_wrong_array[x]) is 0:
                    #remove_paths = "src/output/intermediate/misc" + image_paths[x][24:]
                    #print(remove_paths)
                    #os.rename(image_paths[x], remove_paths)
                    exceptions += 1

            # accuracy = np.mean(right_wrong_array)
            # print('Accuracy: %.3f' % accuracy, "- " + str(exceptions) + " exceptions")"""

            print("ID eval complete")

    except TypeError:
        print("not evaluating ID")

    try:
        with open(classifier_filename[1], 'rb') as GENf:
            model, class_names = pickle.load(GENf)

            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            for i in range(len(best_class_indices)):
                try:
                    blablabla = results[0]
                    for y in results:
                        if y[0] == i:
                            y.append(class_names[best_class_indices[i]])
                            y.append(best_class_probabilities[i])
                except IndexError:
                    results.append([i, class_names[best_class_indices[i]], best_class_probabilities[i]])

            """print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                print(image_paths[i])
                # print(str(i) + image_paths[i] + )
                # print(label_array)"""

            """right_wrong_array = np.equal(best_class_indices, label_array)
            exceptions = 0
            for x in range(0, len(right_wrong_array)):
                if int(right_wrong_array[x]) is 0:
                    remove_paths = "src/output/intermediate/misc" + image_paths[x][24:]
                    print(remove_paths)
                    os.rename(image_paths[x], remove_paths)
                    exceptions += 1

            accuracy = np.mean(right_wrong_array)
            print('Accuracy: %.3f' % accuracy, "- " + str(exceptions) + " exceptions")"""

            print("GEN eval complete")

    except TypeError:
        print("Not evaluating Gender")

    try:
        with open(classifier_filename[2], 'rb') as AGEf:
            model, class_names = pickle.load(AGEf)

            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            for i in range(len(best_class_indices)):
                try:
                    blablabla = results[0]
                    for y in results:
                        if y[0] == i:
                            y.append(class_names[best_class_indices[i]])
                            y.append(best_class_probabilities[i])
                except IndexError:
                    results.append([i, class_names[best_class_indices[i]], best_class_probabilities[i]])

            """print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                print(image_paths[i])
                # print(str(i) + image_paths[i] + )
                # print(label_array)"""

            """right_wrong_array = np.equal(best_class_indices, label_array)
            exceptions = 0
            for x in range(0, len(right_wrong_array)):
                if int(right_wrong_array[x]) is 0:
                    remove_paths = "src/output/intermediate/misc" + image_paths[x][24:]
                    print(remove_paths)
                    os.rename(image_paths[x], remove_paths)
                    exceptions += 1

            accuracy = np.mean(right_wrong_array)
            print('Accuracy: %.3f' % accuracy, "- " + str(exceptions) + " exceptions")"""

            print("AGE eval complete")

    except TypeError:
        print("Not evaluating Age")

    results_dict_array = []

    for z in results:
        # string = str(z[0]) + "- [" + z[1] + "] recognized as " + z[2] + " (" + str(z[3]) + ") "
        # string += "and " + z[4] + " (" + str(z[5]) + ")"

        """z[3] *= 100
        z[5] *= 100
        z[3] = real_round(z[3], 2)
        z[5] = real_round(z[5], 2)"""
        z[3] = real_round(z[3]*100, 2) if z[3] is not None else None
        z[5] = real_round(z[5]*100, 2) if z[5] is not None else None
        z[7] = real_round(z[7]*100, 2) if z[7] is not None else None

        results_dict = {
            "#": z[0],
            "image_path": z[1],
            "ID": z[2],
            "ID-prob": z[3],
            "ID-case": decide_case(z[3]),
            "GEN": z[4],
            "GEN-prob": z[5],
            "GEN-case": decide_case(z[5]),
            "AGE": z[6],
            "AGE-prob": z[7],
            "AGE-case": decide_case_age(z[7])
        }
        results_dict_array.append(results_dict)
    # print(len(results), len(results_dict_array))

    """print(images)
    out = images + "/results.json"
    with open(out, 'w') as f:
        json.dump(results_dict_array, f)"""
    return results_dict_array


def real_round(num, place):
    return float(round(decimal.Decimal(num), place))


def decide_case(prob):
    if prob < 60:
        val = "Toss-Up"
    elif prob < 75:
        val = "Not Confident"
    else:
        val = "Confident"
    return val
    
def decide_case_age(prob):
    if prob < 40:
        val = "Toss-Up"
    elif prob < 55:
        val = "Not Confident"
    else:
        val = "Confident"
    return val



"""
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model-path', type=str, action='store', dest='model_path',
                        help='Path to model protobuf graph')
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir',
                        help='Input path of data to train on')
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size',
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
                        help='Path to output trained classifier model')
    parser.add_argument('--id-classifier-path', type=str, action='store', dest='id_classifier_path',
                        help='Path to input trained classifier model of identity')
    parser.add_argument('--gender-classifier-path', type=str, action='store', dest='gen_classifier_path',
                        help='Path to input trained classifier model of gender')
    parser.add_argument('--is-train', action='store_true', dest='is_train', default=False,
                        help='Flag to determine if train or evaluate')
    parser.add_argument('--override', action='store_true', dest='override', default=False,
                        help='Flag to skip time verification')
    parser.add_argument('--new-embeds', action='store_true', dest='embed', default=False,
                        help='Flag to skip time verification')  # TODO remove

    args = parser.parse_args()

    main(input_directory=args.input_dir, model_path=args.model_path, id_classifier_output_path=args.id_classifier_path,
         gen_classifier_output_path=args.gen_classifier_path,
         batch_size=args.batch_size, num_threads=args.num_threads, num_epochs=args.num_epochs,
         min_images_per_labels=args.min_images_per_class, split_ratio=args.split_ratio, is_train=args.is_train,
         override=args.override, embed=args.embed, train_path=args.train_classifier_path)"""

