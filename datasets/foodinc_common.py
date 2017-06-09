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
"""Provides data for the Foodinc Dataset (images + annotations).
"""
import os

import tensorflow as tf
from datasets import dataset_utils

slim = tf.contrib.slim

FOODINC_LABELS = {
    'none': (0, 'Background'),
    'white rice': (1, 'Rice'), 'brown rice': (2, 'Rice'), 'other rice': (3, 'Rice'),
    'bread': (4, 'Bread'), 'cooked bread': (5, 'Bread'), 'sweet bread': (6, 'Bread'), 'sandwich': (7, 'Bread'), 'other bread': (8, 'Bread'),
    'ramen': (9, 'Noodles'), 'udon': (10, 'Noodles'), 'soba': (11, 'Noodles'), 'other noodles': (12, 'Noodles'),
    'white fish': (13, 'Fish'), 'blue fish': (14, 'Fish'), 'shellfish': (15, 'Fish'), 'crustacean': (16, 'Fish'), 'other fish': (17, 'Fish'),
    'steak': (18, 'Meat'), 'beef': (19, 'Meat'), 'pork': (20, 'Meat'), 'chicken': (21, 'Meat'), 'ham - baccon': (22, 'Meat'), 'other meat': (23, 'Meat'),
    'tofu': (24, 'Soyfood'), 'soymilk': (25, 'Soyfood'), 'natto': (26, 'Soyfood'), 'beans': (27, 'Soyfood'), 'other soyfoods': (28, 'Soyfood'),
    'eggs': (29, 'Eggs'), 'egg dish': (30, 'Eggs'),
    'fruits': (31, 'Fruits'),
    'tomato': (32, 'Vegetables'), 'broccoli': (33, 'Vegetables'), 'root crops': (34, 'Vegetables'), 'green and yellow vegetables': (35, 'Vegetables'), 'mushrooms': (36, 'Vegetables'), 'other vegetables': (37, 'Vegetables'),
    'milk': (38, 'Dairy products'), 'yogurt': (39, 'Dairy products'), 'cheese': (40, 'Dairy products'), 'other dairy products': (41, 'Dairy products'),
    'nuts and seeds': (42, 'Nuts and Seeds'),
    'water': (43, 'Beverages'), 'juice': (44, 'Beverages'), 'vegetable juices': (45, 'Beverages'), 'coffe and tea': (46, 'Beverages'), 'alcohol': (47, 'Beverages'), 'other drinks': (48, 'Beverages'),
    'stir-fried food': (49, 'Recipes'), 'fried food': (50, 'Recipes'), 'steamed food': (51, 'Recipes'), 'grilled food': (52, 'Recipes'), 'simmered food': (53, 'Recipes'),
    'green salad': (54, 'Salad'), 'seaweed salad': (55, 'Salad'), 'potato - pumpkins salad': (56, 'Salad'), 'proteinized salad': (57, 'Salad'), 'vinegared salad': (58, 'Salad'),
    'soup stock': (59, 'Soup stock'),
    'pastry': (60, 'Pastries'), 'japanese pastry': (61, 'Pastries'),
    'curry': (62, 'Rice dishes'), 'rice ball - seaweed roll': (63, 'Rice dishes'), 'porridge': (64, 'Rice dishes'), 'other rice dishes': (65, 'Rice dishes'),
    'pot': (66, 'Others'), 'bento': (67, 'Others'),
}


def get_split(split_name, dataset_dir, file_pattern, reader,
              split_to_sizes, items_to_descriptions, num_classes):
    """Gets a dataset tuple with instructions for reading Foodinc dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    # Features in Foodinc TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    # else:
    #     labels_to_names = create_readable_names_for_imagenet_labels()
    #     dataset_utils.write_label_file(labels_to_names, dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)
