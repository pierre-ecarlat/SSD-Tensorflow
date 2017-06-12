# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts Foodinc data to TFRecords file format with Example protos.

The raw Foodinc data set is expected to reside in PNG files located in the
directory 'Images'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotations' directory, in the TXT format

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing PNG encoded image in RGB colorspace
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always 'PNG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import re
import sys
import random
from PIL import Image

import numpy as np
import tensorflow as tf

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from datasets.foodinc_common import FOODINC_LABELS

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'Images/'
DIRECTORY_IMAGESETS = 'ImageSets/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200


def _parse_rec(filename, shape):
    """ Parse a Foodinc txt file """
    
    # Read annots
    with open(filename) as f:
        data = f.read()
    objs = re.findall('\d+[\s\-]+\d+[\s\-]+\d+[\s\-]+\d+[\s\-]+\d+', data)
    
    # Objects
    bboxes = []
    labels = []

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        coor = re.findall('\d+', obj)
        # Make pixel indexes 0-based
        cls = int(coor[0])
        x1 = float(coor[1])
        y1 = float(coor[2])
        x2 = float(coor[3])
        y2 = float(coor[4])

        labels.append (cls)
        bboxes.append ([y1 / shape[0], 
                        x1 / shape[1], 
                        y2 / shape[0], 
                        x2 / shape[1]])

    return labels, bboxes


def _process_image(directory, name):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.png'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, png encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    filename = directory + DIRECTORY_IMAGES + name + '.png'
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    # Image caracteristics
    size = Image.open(filename).size
    shape = [size[0], size[1], 3]

    # Read the TXT annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.txt')

    labels, bboxes = _parse_rec(filename, shape)

    return image_data, shape, bboxes, labels


def _convert_to_example(image_data, labels, bboxes, shape):
    """Build an Example proto for an image example.

    Args:
      image_data: string, PNG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'png'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels = _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, bboxes, shape)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='foodinc', set='trainval', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    set_list_path = os.path.join(dataset_dir, DIRECTORY_IMAGESETS, set + ".txt")
    if not os.path.isfile (set_list_path):
        print("Can't find the list", set_list_path)
        return
    set_list = [line.rstrip('\n') + ".txt" for line in open(set_list_path)]
    filenames = sorted(set_list)
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, '_'.join([name, set]), fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    #labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    #dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Foodinc dataset!')