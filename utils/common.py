#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-17
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/


import errno
import glob
import gzip
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import tarfile
import zipfile
import PIL
import pkg_resources
import six
import cv2
import math
from keras.utils import Progbar
from six.moves.urllib.error import HTTPError, URLError
from onnx2keras import onnx_to_keras
import onnx

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras

sys.stderr = stderr
import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image, ExifTags, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from keras.layers import Dense, Activation
from keras.models import Model
from keras.preprocessing import image
from models import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, ArcFace

from utils.align_face import align, align_blazeface
from six.moves.urllib.request import urlopen

if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        def chunk_read(response, chunk_size=8192, reporthook=None):
            content_type = response.info().get('Content-Length')
            total_size = -1
            if content_type is not None:
                total_size = int(content_type.strip())
            count = 0
            while True:
                chunk = response.read(chunk_size)
                count += 1
                if reporthook is not None:
                    reporthook(count, chunk_size, total_size)
                if chunk:
                    yield chunk
                else:
                    break

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve


def clip_img(X, preprocessing='raw'):
    X = reverse_preprocess(X, preprocessing)
    X = np.clip(X, 0.0, 255.0)
    X = preprocess(X, preprocessing)
    return X


IMG_SIZE = 112
PREPROCESS = 'raw'


def load_image(path):
    try:
        img = Image.open(path)
    except PIL.UnidentifiedImageError:
        return None
    except IsADirectoryError:
        return None

    try:
        info = img._getexif()
    except AttributeError:
        return None

    if info is not None:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = dict(img._getexif().items())
        if orientation in exif.keys():
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
            else:
                pass

    img = img.convert('RGB')
    image_array = image.img_to_array(img)

    return image_array


def filter_image_paths(image_paths, verbose=1):
    if verbose:
        print("Identify {} files in the directory".format(len(image_paths)))
    new_image_paths = []
    new_images = []
    for p in image_paths:
        img = load_image(p)
        if img is None:
            print("{} is not an image file, skipped".format(p.split("/")[-1]))
            continue
        new_image_paths.append(p)
        new_images.append(img)
    if verbose:
        print("Identify {} images in the directory".format(len(new_image_paths)))
    return new_image_paths, new_images


def box_face(img, margin=30, need_resize=True, base_color=None):
    if base_color is None:
        base_color = np.mean(img, axis=(0, 1))
    long_size = max([img.shape[1], img.shape[0]]) + margin
    base = np.ones((long_size, long_size, 3)) * base_color
    start1, end1 = get_ends(long_size, img.shape[0])
    start2, end2 = get_ends(long_size, img.shape[1])
    base[start1:end1, start2:end2, :] = img
    return resize(base, (IMG_SIZE, IMG_SIZE)) if need_resize else base


class Faces(object):
    def __init__(self, image_paths, loaded_images, aligner, verbose=1, eval_local=False, preprocessing=True,
                 no_align=False):
        self.image_paths = image_paths
        self.verbose = verbose
        self.no_align = no_align
        self.aligner = aligner
        self.margin = 30
        self.org_faces = []
        self.cropped_faces = []
        self.cropped_faces_shape = []
        self.cropped_index = []
        self.start_end_ls = []
        self.callback_idx = []
        self.images_without_face = []
        for i in range(0, len(loaded_images)):
            cur_img = loaded_images[i]
            p = image_paths[i]
            self.org_faces.append(cur_img)

            if not no_align:
                align_img = align(cur_img, self.aligner)
                if align_img == ([], []):
                    if verbose:
                        print("Find 0 face(s) in {}".format(p.split("/")[-1]))
                    self.images_without_face.append(i)
                    continue

                cur_faces = align_img[0]
                if len(cur_faces) > 1:
                    cur_faces = [sorted(cur_faces, key=lambda x: -x.shape[0] * x.shape[1])[0]]
            else:
                cur_faces = [cur_img]

            cur_faces = [face for face in cur_faces if face.shape[0] != 0 and face.shape[1] != 0]
            cur_shapes = [f.shape[:-1] for f in cur_faces]

            cur_faces_square = []
            if verbose and not no_align:
                print("Find {} face(s) in {}".format(len(cur_faces), p.split("/")[-1]))
            if eval_local:
                cur_faces = cur_faces[:1]

            for img in cur_faces:
                if eval_local:
                    base = resize(img, (IMG_SIZE, IMG_SIZE))
                else:
                    long_size = max([img.shape[1], img.shape[0]]) + self.margin

                    base = np.ones((long_size, long_size, 3)) * np.mean(img, axis=(0, 1))

                    start1, end1 = get_ends(long_size, img.shape[0])
                    start2, end2 = get_ends(long_size, img.shape[1])

                    base[start1:end1, start2:end2, :] = img
                    cur_start_end = (start1, end1, start2, end2)
                    self.start_end_ls.append(cur_start_end)

                cur_faces_square.append(base)
            cur_faces_square = [resize(f, (IMG_SIZE, IMG_SIZE)) for f in cur_faces_square]
            self.cropped_faces.extend(cur_faces_square)

            if not self.no_align:
                cur_index = align_img[1]
                self.cropped_faces_shape.extend(cur_shapes)
                self.cropped_index.extend(cur_index[:len(cur_faces_square)])
                self.callback_idx.extend([i] * len(cur_faces_square))

        if len(self.cropped_faces) == 0:
            return

        self.cropped_faces = np.array(self.cropped_faces)

        if preprocessing:
            self.cropped_faces = preprocess(self.cropped_faces, PREPROCESS)

        self.cloaked_cropped_faces = None
        self.cloaked_faces = np.copy(self.org_faces)

    def get_faces(self):
        return self.cropped_faces

    def merge_faces(self, protected_images, original_images):
        if self.no_align:
            return np.clip(protected_images, 0.0, 255.0), self.images_without_face

        self.cloaked_faces = np.copy(self.org_faces)

        for i in range(len(self.cropped_faces)):
            cur_protected = protected_images[i]
            cur_original = original_images[i]

            org_shape = self.cropped_faces_shape[i]

            old_square_shape = max([org_shape[0], org_shape[1]]) + self.margin

            cur_protected = resize(cur_protected, (old_square_shape, old_square_shape))
            cur_original = resize(cur_original, (old_square_shape, old_square_shape))

            start1, end1, start2, end2 = self.start_end_ls[i]

            reshape_cloak = cur_protected - cur_original
            reshape_cloak = reshape_cloak[start1:end1, start2:end2, :]

            callback_id = self.callback_idx[i]
            bb = self.cropped_index[i]
            self.cloaked_faces[callback_id][bb[0]:bb[2], bb[1]:bb[3], :] += reshape_cloak

        for i in range(0, len(self.cloaked_faces)):
            self.cloaked_faces[i] = np.clip(self.cloaked_faces[i], 0.0, 255.0)
        return self.cloaked_faces, self.images_without_face


class Faces_MTCNN(object):
    def __init__(self, image_paths, loaded_images, aligner):
        self.aligner = aligner
        self.image_paths = image_paths
        self.margin = 30
        self.org_faces = []
        self.cropped_faces = []
        self.cropped_fnames = []
        self.cropped_faces_shape = []
        self.cropped_index = []
        self.start_end_ls = []
        self.callback_idx = []
        self.images_without_face = []
        for i in range(0, len(loaded_images)):
            cur_img = loaded_images[i]
            p = image_paths[i]
            self.org_faces.append(cur_img)

            align_img = align(cur_img, self.aligner)
            if align_img == ([], []):
                self.images_without_face.append(i)
                continue

            cur_faces = align_img[0]
            if len(cur_faces) > 1:
                continue

            cur_faces = [face for face in cur_faces if face.shape[0] != 0 and face.shape[1] != 0]
            cur_shapes = [f.shape[:-1] for f in cur_faces]

            cur_faces_square = []

            for img in cur_faces:
                long_size = max([img.shape[1], img.shape[0]]) + self.margin

                base = np.ones((long_size, long_size, 3)) * np.mean(img, axis=(0, 1))

                start1, end1 = get_ends(long_size, img.shape[0])
                start2, end2 = get_ends(long_size, img.shape[1])

                base[start1:end1, start2:end2, :] = img
                cur_start_end = (start1, end1, start2, end2)
                self.start_end_ls.append(cur_start_end)

                cur_faces_square.append(base)
            cur_faces_square = [resize(f, (IMG_SIZE, IMG_SIZE)) for f in cur_faces_square]
            self.cropped_faces.extend(cur_faces_square)
            self.cropped_fnames.extend([p] * len(cur_faces_square))

            cur_index = align_img[1]
            self.cropped_faces_shape.extend(cur_shapes)
            self.cropped_index.extend(cur_index[:len(cur_faces_square)])
            self.callback_idx.extend([i] * len(cur_faces_square))

        if len(self.cropped_faces) == 0:
            return

        self.cropped_faces = np.array(self.cropped_faces)
        self.cropped_faces = preprocess(self.cropped_faces, PREPROCESS)
        self.cloaked_cropped_faces = None
        self.cloaked_faces = np.copy(self.org_faces)


class Faces_BlazeFace(object):
    def __init__(self, image_paths, loaded_images):
        self.image_paths = image_paths
        self.margin = 30
        self.org_faces = []
        self.cropped_faces = []
        self.cropped_fnames = []
        self.cropped_faces_shape = []
        self.cropped_index = []
        self.start_end_ls = []
        self.callback_idx = []
        self.images_without_face = []
        for i in range(0, len(loaded_images)):
            cur_img = loaded_images[i]
            p = image_paths[i]
            self.org_faces.append(cur_img)

            align_img = align_blazeface(cur_img)
            if align_img == ([], []):
                self.images_without_face.append(i)
                continue

            cur_faces = align_img[0]
            if len(cur_faces) > 1:
                continue

            cur_faces = [face for face in cur_faces if face.shape[0] != 0 and face.shape[1] != 0]
            cur_shapes = [f.shape[:-1] for f in cur_faces]

            cur_faces_square = []

            for img in cur_faces:
                long_size = max([img.shape[1], img.shape[0]]) + self.margin

                base = np.ones((long_size, long_size, 3)) * np.mean(img, axis=(0, 1))

                start1, end1 = get_ends(long_size, img.shape[0])
                start2, end2 = get_ends(long_size, img.shape[1])

                base[start1:end1, start2:end2, :] = img
                cur_start_end = (start1, end1, start2, end2)
                self.start_end_ls.append(cur_start_end)

                cur_faces_square.append(base)
            cur_faces_square = [resize(f, (IMG_SIZE, IMG_SIZE)) for f in cur_faces_square]
            self.cropped_faces.extend(cur_faces_square)
            self.cropped_fnames.extend([p] * len(cur_faces_square))

            cur_index = align_img[1]
            self.cropped_faces_shape.extend(cur_shapes)
            self.cropped_index.extend(cur_index[:len(cur_faces_square)])
            self.callback_idx.extend([i] * len(cur_faces_square))

        if len(self.cropped_faces) == 0:
            return

        self.cropped_faces = np.array(self.cropped_faces)
        self.cropped_faces = preprocess(self.cropped_faces, PREPROCESS)
        self.cloaked_cropped_faces = None
        self.cloaked_faces = np.copy(self.org_faces)


def get_ends(longsize, window):
    start = (longsize - window) // 2
    end = start + window
    return start, end


def dump_dictionary_as_json(dict, outfile):
    j = json.dumps(dict)
    with open(outfile, "wb") as f:
        f.write(j.encode())


def load_victim_model(number_classes, teacher_model=None, end2end=False):
    for l in teacher_model.layers:
        l.trainable = end2end
    x = teacher_model.layers[-1].output

    x = Dense(number_classes)(x)
    x = Activation('softmax', name="act")(x)
    model = Model(teacher_model.input, x)
    opt = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def resize(img, sz):
    assert np.min(img) >= 0 and np.max(img) <= 255.0, '%s | %s' % (str(img.shape), str(np.max(img)))
    im_data = image.array_to_img(img).resize((sz[1], sz[0]))
    im_data = image.img_to_array(im_data)
    return im_data


def init_gpu(gpu):
    ''' code to initialize gpu in tf2'''
    if isinstance(gpu, list):
        gpu_num = ','.join([str(i) for i in gpu])
    else:
        gpu_num = str(gpu)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print('GPU already initiated')
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)


def fix_gpu_memory(mem_fraction=1):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf_config = None
    if tf.test.is_gpu_available():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        tf_config.gpu_options.allow_growth = True
        tf_config.log_device_placement = False
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)
    return sess


def preprocess(X, method):
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method == 'raw':
        pass
    elif method == 'imagenet':
        X = imagenet_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def reverse_preprocess(X, method):
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method == 'raw':
        pass
    elif method == 'imagenet':
        X = imagenet_reverse_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def imagenet_preprocessing(x, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    x = np.array(x)
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]

    mean = [103.939, 116.779, 123.68]
    std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]

    return x


def imagenet_reverse_preprocessing(x, data_format=None):
    import keras.backend as K
    x = np.array(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    if data_format == 'channels_first':
        if x.ndim == 3:
            # Zero-center by mean pixel
            x[0, :, :] += 103.939
            x[1, :, :] += 116.779
            x[2, :, :] += 123.68
            # 'BGR'->'RGB'
            x = x[::-1, :, :]
        else:
            x[:, 0, :, :] += 103.939
            x[:, 1, :, :] += 116.779
            x[:, 2, :, :] += 123.68
            x = x[:, ::-1, :, :]
    else:
        # Zero-center by mean pixel
        x[..., 0] += 103.939
        x[..., 1] += 116.779
        x[..., 2] += 123.68
        # 'BGR'->'RGB'
        x = x[..., ::-1]
    return x


def reverse_process_cloaked(x, preprocess='raw'):
    # x = clip_img(x, preprocess)
    return reverse_preprocess(x, preprocess)


def build_bottleneck_model(model, cut_off):
    bottleneck_model = Model(model.input, model.get_layer(cut_off).output)
    bottleneck_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])
    return bottleneck_model


def load_extractor(name, image_shape=(112, 112, 3)):
    # Input Layer -> Backbone -> 512D Feature Vector
    fpath_weights = "weights/{}.h5".format(name)
    if name in ['ArcFace', 'DeepFace', 'DeepID', 'Facenet', 'Facenet512', 'OpenFace', 'VGG-Face']:
        models = {
            'VGG-Face': VGGFace.loadModel,
            'OpenFace': OpenFace.loadModel,
            'Facenet': Facenet.loadModel,
            'Facenet512': Facenet512.loadModel,
            'DeepFace': FbDeepFace.loadModel,
            'DeepID': DeepID.loadModel,
            'ArcFace': ArcFace.loadModel,
        }
        model = models[name](fpath_weights)
    elif 'InsightFace' in name:
        fpath_weights = "weights/{}.onnx".format(name)
        onnx_model = onnx.load(fpath_weights)
        model = onnx_to_keras(onnx_model, [onnx_model.graph.input[0].name], name_policy='renumerate', verbose=False)
    else:
        model = keras.models.load_model(fpath_weights, compile=False)

    model_channel_first = model.layers[0].input_shape[0][1] == 3
    model_input_res = model.layers[0].input_shape[0][2:4] if model_channel_first else model.layers[0].input_shape[0][1:3]
    pipeline_input_res = image_shape[:2]
    if model_input_res != pipeline_input_res:
        assert model_channel_first is False, 'model_channel_first is True and resizing layer is needed.'
        inputs = keras.Input(shape=(3, *pipeline_input_res) if model_channel_first else (*pipeline_input_res, 3))
        x = inputs
        x = keras.layers.experimental.preprocessing.Resizing(*model_input_res)(x)
        outputs = model(x)
        model = keras.Model(inputs, outputs)
    model = Extractor(name, model)
    return model


class Extractor(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.input_shape = self.find_input_shape()

    def predict(self, imgs):
        if self.name == 'Celeb1M-EfficientNet':
            imgs = imgs / 255.0
        elif self.name == 'Celeb1M-ResNet50':
            imgs = imgs / 255.0
        elif self.name == 'WebFace':
            mean = tf.constant(np.repeat([[[[103.939, 116.779, 123.68]]]], imgs.shape[0], axis=0), dtype=tf.float32)
            imgs = (imgs[..., ::-1] - mean)
        elif self.name == 'ArcFace':
            imgs = imgs[..., ::-1]
            imgs = imgs / 255.0
        elif self.name == 'DeepFace':
            imgs = imgs[..., ::-1]
            imgs = imgs / 255.0
        elif self.name == 'DeepID':
            imgs = imgs[..., ::-1]
            imgs = imgs / 255.0
        elif self.name == 'Facenet':
            imgs = imgs[..., ::-1]
            imgs = imgs / 255.0
        elif self.name == 'Facenet512':
            imgs = imgs[..., ::-1]
            imgs = imgs / 255.0
        elif self.name == 'OpenFace':
            imgs = imgs[..., ::-1]
            imgs = imgs / 255.0
        elif self.name == 'VGG-Face':
            imgs = imgs[..., ::-1]
            imgs = imgs / 255.0
        elif 'InsightFace' in self.name:
            imgs = (imgs / 255.0 - 0.50) / 0.50
            imgs = tf.transpose(imgs, perm=(0, 3, 1, 2))
        else:
            raise NotImplementedError
        embeds = l2_norm(self.model(imgs))
        return embeds

    def __call__(self, x):
        return self.predict(x)

    def find_input_shape(self):
        input_shape = self.model.layers[0].input_shape
        if type(input_shape) == list:
            if input_shape[0][1] == 3:
                input_shape = input_shape[0][2:4]
            else:
                input_shape = input_shape[0][1:3]
        else:
            if input_shape[1] == 3:
                input_shape = input_shape[2:4]
            else:
                input_shape = input_shape[1:3]
        if type(input_shape) == list:  # issue 197: some people got array here instead of tuple
            input_shape = tuple(input_shape)
        return input_shape


def get_dataset_path(dataset):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise Exception("Please config the datasets before running protection code. See more in README and config.py.")

    config = json.load(open(os.path.join(model_dir, "config.json"), 'r'))
    if dataset not in config:
        raise Exception(
            "Dataset {} does not exist, please download to data/ and add the path to this function... Abort".format(
                dataset))
    return config[dataset]['train_dir'], config[dataset]['test_dir'], config[dataset]['num_classes'], config[dataset][
        'num_images']


def dump_image(x, filename, format="png", scale=False):
    img = image.array_to_img(x, scale=scale)
    img.save(filename, format)
    return


def load_embeddings(feature_extractors_names):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    for extractor_name in feature_extractors_names:
        fp = gzip.open(os.path.join(model_dir, "{}_emb.p.gz".format(extractor_name)), 'rb')
        path2emb = pickle.load(fp)
        fp.close()

    return path2emb


def extractor_ls_predict(feature_extractors_ls, X):
    feature_ls = []
    for extractor in feature_extractors_ls:
        cur_features = extractor.predict(X)
        feature_ls.append(cur_features)
    concated_feature_ls = np.concatenate(feature_ls, axis=1)
    return concated_feature_ls


def pairwise_l2_distance(A, B):
    BT = B.transpose()
    vecProd = np.dot(A, BT)
    SqA = A ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def select_target_label(imgs, feature_extractors_ls, feature_extractors_names, metric='l2'):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')

    original_feature_x = extractor_ls_predict(feature_extractors_ls, imgs)

    path2emb = load_embeddings(feature_extractors_names)

    items = list([(k, v) for k, v in path2emb.items()])
    paths = [p[0] for p in items]
    embs = [p[1] for p in items]
    embs = np.array(embs)

    pair_dist = pairwise_l2_distance(original_feature_x, embs)
    pair_dist = np.array(pair_dist)

    max_sum = np.min(pair_dist, axis=0)
    max_id_ls = np.argsort(max_sum)[::-1]

    max_id = random.choice(max_id_ls[:20])

    target_data_id = paths[int(max_id)]
    print("target ID: {}".format(target_data_id))

    image_dir = os.path.join(model_dir, "target_data/{}".format(target_data_id))

    os.makedirs(os.path.join(model_dir, "target_data"), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    for i in range(10):
        if os.path.exists(os.path.join(model_dir, "target_data/{}/{}.jpg".format(target_data_id, i))):
            continue
        try:
            get_file("{}.jpg".format(i),
                     "http://mirror.cs.uchicago.edu/fawkes/files/target_data/{}/{}.jpg".format(target_data_id, i),
                     cache_dir=model_dir, cache_subdir='target_data/{}/'.format(target_data_id))
        except Exception:
            pass

    image_paths = glob.glob(image_dir + "/*.jpg")

    target_images = [image.img_to_array(image.load_img(cur_path)) for cur_path in
                     image_paths]

    target_images = np.array([resize(x, (IMG_SIZE, IMG_SIZE)) for x in target_images])
    target_images = preprocess(target_images, PREPROCESS)

    target_images = list(target_images)
    while len(target_images) < len(imgs):
        target_images += target_images

    target_images = random.sample(target_images, len(imgs))
    return np.array(target_images)


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = tf.norm(x, axis=axis, keepdims=True)
    output = x / norm
    return output


""" TensorFlow implementation get_file
https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/utils/data_utils.py#L168-L297
"""


def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, cache_subdir)
    _makedirs_exist_ok(datadir)

    # fname = path_to_string(fname)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of ' + file_hash +
                      ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


def _extract_archive(file_path, path='.', archive_format='auto'):
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def _makedirs_exist_ok(datadir):
    if six.PY2:
        # Python 2 doesn't have the exist_ok arg, so we try-except here.
        try:
            os.makedirs(datadir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    else:
        os.makedirs(datadir, exist_ok=True)  # pylint: disable=unexpected-keyword-arg


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.
    Arguments:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    Returns:
        Whether the file is valid
    """
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(file_hash) == 64):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """Calculates a file sha256 or md5 hash.
    Example:
    ```python
    _hash_file('/path/to/file.zip')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```
    Arguments:
        fpath: path to the file being validated
        algorithm: hash algorithm, one of `'auto'`, `'sha256'`, or `'md5'`.
            The default `'auto'` detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    Returns:
        The file hash
    """
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def apply_upmask(imgs, upmask):
    protected_imgs = np.clip(imgs + upmask, a_min=0., a_max=255.)
    return protected_imgs


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
