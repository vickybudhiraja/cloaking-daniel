import numpy as np
import pickle
import random
import os


class DataManager(object):
    @staticmethod
    def getSplits(artist, bbox_fname='./data/bboxes.pkl', probe_gallery_train=(0.1, 0.2, 0.7), seed=2023):
        with open(bbox_fname, 'rb') as f:
            data = pickle.load(f)[artist]
        fnames = list(data.keys())
        random.seed(seed)
        random.shuffle(fnames)
        splitter1 = int(len(fnames) * probe_gallery_train[0])
        splitter2 = int(len(fnames) * (probe_gallery_train[0] + probe_gallery_train[1]))
        return {'probe': fnames[:splitter1], 'gallery': fnames[splitter1:splitter2], 'train': fnames[splitter2:]}

    @staticmethod
    def getFacialImages(artist, root_dir_faces, return_fnames):
        splits = DataManager.getSplits(artist)
        with open(os.path.join(root_dir_faces, artist + '.pkl'), 'rb') as f:
            data = pickle.load(f)
        probe_images = np.asarray([data[fname] for fname in splits['probe'] if fname in data])
        gallery_images = np.asarray([data[fname] for fname in splits['gallery'] if fname in data])
        train_images = np.asarray([data[fname] for fname in splits['train'] if fname in data])
        out = {'probe': probe_images, 'gallery': gallery_images, 'train': train_images}
        if return_fnames:
            out = (out, splits)
        return out
