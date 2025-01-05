from core.chameleon import P3MaskGeneration
from utils.common import load_extractor
from utils.constants import FR_COLLECTION, PROTECTED_ARTISTS
from utils.dataset import DataManager
import numpy as np
import pickle
import os
import tensorflow as tf

class PrivacyProtector(object):
    def __init__(self, team, batch_size, epsilon=16):
        self.feature_extractors_ls = [load_extractor(name) for name in team]
        self.protector = P3MaskGeneration(self.feature_extractors_ls, epsilon=epsilon, batch_size=batch_size)

    def run_protection(self, protectee_train_imgs, protectee_train_bbox, output_dir, l_threshold):
        os.makedirs(output_dir, exist_ok=True)
        return self.protector.compute(
            protectee_train_imgs, protectee_train_bbox, output_dir, l_threshold=l_threshold)


def run(artist, team, batch_size, epsilon, l_threshold):
    output_dir = './outputs/%s' % artist
    with open('./data/bboxes.pkl', 'rb') as f:
        bboxes = pickle.load(f)
    protectee_data, protectee_fnames = DataManager.getFacialImages(artist=artist,
                                                                   root_dir_faces='./data/faces',
                                                                   return_fnames=True)
    protectee_train_imgs = protectee_data['train']
    protectee_train_bbox = np.asarray([bboxes[artist][fname] for fname in protectee_fnames['train']])

    print('Protecting %s...' % artist)
    protector = PrivacyProtector(team=team, epsilon=epsilon, batch_size=batch_size)
    protector.run_protection(protectee_train_imgs=protectee_train_imgs,
                             protectee_train_bbox=protectee_train_bbox,
                             output_dir=output_dir,
                             l_threshold=l_threshold)
    del protector


if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("No GPU detected. Ensure that the correct drivers and libraries are installed.")
    else:
        print("GPU is detected and ready to use.")    
    epsilon = 16
    l_threshold = 0.030
    batch_size = 4
    team = (FR_COLLECTION[0], FR_COLLECTION[1])
    artist = 'artist1'
    run(artist, team, batch_size, epsilon, l_threshold)
