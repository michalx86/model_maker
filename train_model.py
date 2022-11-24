import numpy as np
import os
import random
import shutil

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

# Set this depending on model you train, processor used for training / evaluation: CPU / GPU, memory available (CPU RAM / GPU memory)
BATCH_SIZE = 2
EPOCHS = 750

# Your labels map as a dictionary (zero is reserved):
label_map = {1: 'GestUp', 2: 'GestDown', 3: 'GestLeft', 4: 'GestRight', 5: 'GestOk', 6: 'GestBack', 7: 'GestExit', 8: 'GestMenu', 9: 'GestPause', 10: 'Man', 11: 'Face'}




tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)



#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#  try:
#    for gpu in gpus:
#      tf.config.experimental.set_memory_growth(gpu, True)
#      print("Setting memory growth for GPU: {}".format(gpu))
#    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#    print("Setting memory limit for GPU")
#  except RuntimeError as e:
#    print(e)


def split_dataset(images_path, annotations_path, val_split, test_split, out_path):
  """Splits a directory of sorted images/annotations into training, validation, and test sets.

  Args:
    images_path: Path to the directory with your images (JPGs).
    annotations_path: Path to a directory with your VOC XML annotation files,
      with filenames corresponding to image filenames. This may be the same path
      used for images_path.
    val_split: Fraction of data to reserve for validation (float between 0 and 1).
    test_split: Fraction of data to reserve for test (float between 0 and 1).
  Returns:
    The paths for the split images/annotations (train_dir, val_dir, test_dir)
  """
  _, dirs, _ = next(os.walk(images_path))

  train_dir = os.path.join(out_path, 'train')
  val_dir = os.path.join(out_path, 'validation')
  test_dir = os.path.join(out_path, 'test')

  IMAGES_TRAIN_DIR = os.path.join(train_dir, 'images')
  IMAGES_VAL_DIR = os.path.join(val_dir, 'images')
  IMAGES_TEST_DIR = os.path.join(test_dir, 'images')
  os.makedirs(IMAGES_TRAIN_DIR, exist_ok=True)
  os.makedirs(IMAGES_VAL_DIR, exist_ok=True)
  os.makedirs(IMAGES_TEST_DIR, exist_ok=True)

  ANNOT_TRAIN_DIR = os.path.join(train_dir, 'annotations')
  ANNOT_VAL_DIR = os.path.join(val_dir, 'annotations')
  ANNOT_TEST_DIR = os.path.join(test_dir, 'annotations')
  os.makedirs(ANNOT_TRAIN_DIR, exist_ok=True)
  os.makedirs(ANNOT_VAL_DIR, exist_ok=True)
  os.makedirs(ANNOT_TEST_DIR, exist_ok=True)

  # Get all filenames for this dir, filtered by filetype
  filenames = os.listdir(os.path.join(images_path))
  filenames = [os.path.join(images_path, f) for f in filenames if (f.endswith('.jpg'))]
  # Shuffle the files, deterministically
  filenames.sort()
  random.seed(42)
  random.shuffle(filenames)
  # Get exact number of images for validation and test; the rest is for training
  val_count = int(len(filenames) * val_split)
  test_count = int(len(filenames) * test_split)
  for i, file in enumerate(filenames):
    source_dir, filename = os.path.split(file)
    annot_file = os.path.join(annotations_path, filename.replace("jpg", "xml"))
    if i < val_count:
      shutil.copy(file, IMAGES_VAL_DIR)
      shutil.copy(annot_file, ANNOT_VAL_DIR)
    elif i < val_count + test_count:
      shutil.copy(file, IMAGES_TEST_DIR)
      shutil.copy(annot_file, ANNOT_TEST_DIR)
    else:
      shutil.copy(file, IMAGES_TRAIN_DIR)
      shutil.copy(annot_file, ANNOT_TRAIN_DIR)
  return (train_dir, val_dir, test_dir)
  

SPLIT_DATASET_NAME='split-dataset'
dataset_is_split = True


if dataset_is_split:
    # If your dataset is already split, specify each path:
    train_images_dir = SPLIT_DATASET_NAME + '/train/images'
    train_annotations_dir = SPLIT_DATASET_NAME + '/train/annotations'
    val_images_dir = SPLIT_DATASET_NAME + '/validation/images'
    val_annotations_dir = SPLIT_DATASET_NAME + '/validation/annotations'
    test_images_dir = SPLIT_DATASET_NAME + '/test/images'
    test_annotations_dir = SPLIT_DATASET_NAME + '/test/annotations'
else:
    # If it's NOT split yet, specify the path to all images and annotations
    images_in = 'dataset/images'
    annotations_in = 'dataset/annotations'

# We need to instantiate a separate DataLoader for each split dataset
if dataset_is_split:
    train_data = object_detector.DataLoader.from_pascal_voc(
        train_images_dir, train_annotations_dir, label_map=label_map)
    validation_data = object_detector.DataLoader.from_pascal_voc(
        val_images_dir, val_annotations_dir, label_map=label_map)
    test_data = object_detector.DataLoader.from_pascal_voc(
        test_images_dir, test_annotations_dir, label_map=label_map)
else:
    train_dir, val_dir, test_dir = split_dataset(images_in, annotations_in,
                                                 val_split=0.2, test_split=0.2,
                                                 out_path='split-dataset')
    train_data = object_detector.DataLoader.from_pascal_voc(
        os.path.join(train_dir, 'images'),
        os.path.join(train_dir, 'annotations'), label_map=label_map)
    validation_data = object_detector.DataLoader.from_pascal_voc(
        os.path.join(val_dir, 'images'),
        os.path.join(val_dir, 'annotations'), label_map=label_map)
    test_data = object_detector.DataLoader.from_pascal_voc(
        os.path.join(test_dir, 'images'),
        os.path.join(test_dir, 'annotations'), label_map=label_map)

print(f'train count: {len(train_data)}')
print(f'validation count: {len(validation_data)}')
print(f'test count: {len(test_data)}')

spec = object_detector.EfficientDetLite0Spec()

model = object_detector.create(train_data=train_data, 
                               model_spec=spec, 
                               validation_data=validation_data, 
                               epochs=EPOCHS,
                               batch_size=BATCH_SIZE, 
                               train_whole_model=True)

TFLITE_FILENAME = 'efficientdet0-lite-gestures.tflite'
LABELS_FILENAME = 'labels.txt'

model.export(export_dir='.', tflite_filename=TFLITE_FILENAME, label_filename=LABELS_FILENAME,
             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])

if (test_data.size > 0):
    print("Using test data for final model evaluation.")
else:
    print("Warning: No test data. Final model evaluation will be done on data used during training.")
    test_data = validation_data
    

print("Evaluating base model: ")
print(model.evaluate(test_data, batch_size = BATCH_SIZE))
print("Evaluating TF Lite model: ")
print(model.evaluate_tflite(TFLITE_FILENAME, test_data))

