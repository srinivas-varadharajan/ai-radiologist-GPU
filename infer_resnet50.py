# This is just a quick and dirty inference, this doesn't use any pipelining mechanism or Optimized Runtime Inference Engine.  

import tensorflow
from sklearn.metrics import roc_auc_score
import numpy as np
from tensorflow.keras.applications import DenseNet121, resnet50
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import optimizers
import horovod.tensorflow.keras as hvd
import PIL.Image as pil
import PIL.ImageOps
from sys import argv
import pickle
import os
from time import time
import argparse

time1 = time()
np.random.seed(123)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--trained_model', type=str, default='/data/saved_model.h5',
    help='The directory where the model will be stored.')

FLAGS, _ = parser.parse_known_args()

def config_gpu():
    """"
    Setup the GPUs to support for running with more memory as opposed to pre allocated memory.
    Controls the percentage use of memory.
    """
    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    session = tensorflow.Session(config=config)


config_gpu()

with open('./training_labels_new.pkl', 'rb') as f:
    training_labels = pickle.load(f)

with open('./validation_labels_new.pkl', 'rb') as f:
    validation_labels = pickle.load(f)
validation_files = np.asarray(list(validation_labels.keys()))
labels = dict(training_labels.items())
labels.update(validation_labels.items())


def load_batch(batch_of_files, is_training=False):
    batch_images = []
    batch_labels = []
    for filename in batch_of_files:
        img = pil.open(os.path.join('/data/images', filename))
        img = img.convert('RGB')
        img = img.resize((256, 256), pil.NEAREST)
        if is_training and np.random.randint(2):
            img = PIL.ImageOps.mirror(img)
        batch_images.append(np.asarray(img))
        batch_labels.append(labels[filename])
    return preprocess_input(np.float32(np.asarray(batch_images))), np.asarray(batch_labels)


def val_generator(num_of_steps):
    while True:
        # np.random.shuffle(validation_files)
        batch_size = 128
        for i in range(num_of_steps):
            batch_of_files = validation_files[i * batch_size: i * batch_size + batch_size]
            batch_images, batch_labels = load_batch(batch_of_files, True)
            yield batch_images, batch_labels


val_steps = 8653 // 128



print("loading model weights from {}:".format(FLAGS.trained_model))
start = time()
model = tensorflow.keras.models.load_model(FLAGS.trained_model)
print("predicting using your model:....")
auc_labels = []
probs = []
for i in range(val_steps):
    batch_of_files = validation_files[i * 128: i * 128 + 128]
    batch_images, batch_labels = load_batch(batch_of_files)
    probs.extend(model.predict_on_batch(batch_images))
    auc_labels.extend(batch_labels)
end = time()
print("computing auc score")

avg_scores = []

for i in range(14):
    aucscore = roc_auc_score(np.asarray(auc_labels)[:, i], np.asarray(probs)[:, i])
    print(aucscore)
    avg_scores.append(aucscore)
print("AUC Avg: {}".format(np.array(avg_scores).mean()))
infer_time = end - start
print("Inference time:  {}s".format(infer_time))
print("Inference Throughput: {} images/second".format(8653/infer_time))
