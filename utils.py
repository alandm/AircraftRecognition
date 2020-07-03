import pickle
import os
import cv2 
import urllib
import numpy as np


def pickle_save(object, path):
    """save object as pickle

    Args:
        object (model): object
        path (string): file path 

    Returns:
        [pkl]: pickle file
    """
    try:
        print('save data to {}'.format(path))
        with open(path, "wb") as f:
            return pickle.dump(object, f)
    except:
        print('save data to {} failed'.format(path))


def http_get_img(url, rst=64, gray=False):
    """ get http image 

    Args:
        url (string): url
        rst (int, optional): [description]. Defaults to 64.
        gray (bool, optional): [description]. Defaults to False.

    Returns:
        [img]:image
    """
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img = cv2.resize(img, (rst, rst))
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img.reshape((1, rst, rst, -1)) - 127.5) / 127.5

def save_weights(model, dir):
    """store model weights from h5 to pickle

    Args:
        model ([type]): [description]
        dir ([type]): path model
    """
    name = model.name + '.h5'
    weights = [l.get_weights() for l in model.layers]
    pickle_save(weights, os.path.join(dir, name))

def set_weights(model, dir):
    """set weight models

    Args:
        model ([type]): [description]
        dir ([type]): [description]
    """
    name = model.name + '.h5'
    weights = pickle_load(os.path.join(dir, name))
    for layer, w in zip(model.layers, weights):
        layer.set_weights(w)


def pickle_load(path):
    """set pickle loads

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    try:
        print('load data from {} successfully'.format(path))
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(str(e))
        return None

def add_padding(img):
    """
    Add black padding
    """
    w, h, _ = img.shape
    size = abs(w - h) // 2
    value= [0, 0, 0]
    if w < h:
        return cv2.copyMakeBorder(img, size, size, 0, 0,
                                    cv2.BORDER_CONSTANT,
                                    value=value)
    return cv2.copyMakeBorder(img, 0, 0, size, size,
                                    cv2.BORDER_CONSTANT,
                                    value=value)

def save_ds(imgs, rst, opt):
    """save figures to pickle

    Args:
        imgs ([type]): [description]
        rst ([type]): [description]
        opt ([type]): [description]
    """
    path = '{}/imgs_{}_{}.pkl'.format(DS_SAVE_DIR, opt, rst)
    pickle_save(imgs, path)

def load_ds(rst, opt):
    """[summary]

    Args:
        rst ([type]): [description]
        opt ([type]): [description]

    Returns:
        [type]: [description]
    """
    path = '{}/imgs_{}_{}.pkl'.format(DS_SAVE_DIR, opt, rst)
    return pickle_load(path)

def get_img(path, rst):
    """[summary]

    Args:
        path ([type]): [description]
        rst ([type]): [description]

    Returns:
        [type]: [description]
    """
    img = cv2.imread(path)
    img = add_padding(img)
    img = cv2.resize(img, (rst, rst))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.expand_dims(img, axis=0)
    return img.tolist()