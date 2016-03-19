# coding : utf-8

import os
import numpy as np
import theano
from PIL import Image

def load_data( data_dir, shape=None, box=None, restrict=-1 ):
    img_files = []
    for root,dirs,files in os.walk( data_dir ):
        if len(files) == 0:
            continue
        img_files.extend( map( lambda f : os.path.join(root,f), files ) )

    arrs = []
    if restrict < 0:
        data_size = len(img_files)
    else:
        data_size = restrict

    for f_name in img_files[:data_size]:
        img = Image.open(f_name)
        if not box == None:
            img = img.crop(box)
        if not shape == None:
            img = img.resize( shape )
        arr = np.asarray( img ).astype( theano.config.floatX )
        arr = preprocess( arr )
        arrs.append( arr )

    arrs = np.asarray( arrs ).astype( theano.config.floatX )

    return arrs

def preprocess( arr ):
    arr = np.array( arr ) # copy
    arr /= 127.5
    arr -= 1.
    arr = arr.transpose(2,0,1)
    return arr

def postprocess( arr ):
    arr = np.array( arr ) # copy
    arr = arr.transpose(1,2,0)
    arr += 1.
    arr *= 127.5
    return arr

