import argparse
import glob
import logging
import multiprocessing as mp
import os
import time
import sys

import cv2

from SE_code.align_dlib import AlignDlib
# from code.align_dlib import AlignDlib

logger = logging.getLogger(__name__)
return_img = []

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))


def main(input_dir, crop_dim, random_int):
    start_time = time.time()
    pool = mp.Pool(processes=mp.cpu_count())

    logger.info("making directory: " + random_int)
    if not random_int:
        sys.exit("no random int")
    os.makedirs(random_int)

    for image_dir in os.listdir(input_dir):
        image_output_dir = os.path.join(random_int, os.path.basename(os.path.basename(image_dir)))
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

    image_paths = glob.glob(os.path.join(input_dir, '**/*.jpg'))
    # print(image_paths)

    image_path = image_paths[0]
    image_output_dir = os.path.join(random_int, os.path.basename(os.path.dirname(image_path)))
    path = image_path.split('/')[-1]
    output_path = os.path.join(image_output_dir, path)
    img = preprocess_image(image_path, output_path, crop_dim)
    print(img)
    return img

    
    logger.info('Completed in {} seconds'.format(time.time() - start_time))


def preprocess_image(input_path, output_path, crop_dim):
    
    """
    Detect face, align and crop :param input_path. Write output to :param output_path
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    """
    image = _process_image(input_path, crop_dim)
    if image is not None:
        cv2.imwrite(output_path, image)
        # print(image)
        logger.debug('Writing processed file: {}'.format(output_path))
    else:
        logger.debug("Skipping filename: {}".format(input_path))
    # logger.info("finished: " + output_path)
    return image


def _process_image(filename, crop_dim):
    image = None
    aligned_image = None

    image = _buffer_image(filename)

    if image is not None:
        aligned_image = _align_image(image, crop_dim)
    else:
        raise IOError('Error buffering image: {}'.format(filename))

    return aligned_image


def _buffer_image(filename):
    logger.debug('Reading image: {}'.format(filename))
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='data', dest='input_dir')
    parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
    parser.add_argument('--crop-dim', type=int, action='store', default=180, dest='crop_dim',
                        help='Size to crop images to')

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.crop_dim)
