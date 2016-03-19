# coding : utf-8

import sys
import os
module_path = os.path.abspath( u'..'  )
sys.path.append( module_path )

import json
import cPickle
import numpy as np
import theano
from vaegan.data import load_data, postprocess
from vaegan import Operate, image_grid
from vaegan import VAEGAN

def load_json( infile_name ):
    fin = open( infile_name, u'r' )
    conf = json.load( fin )
    fin.close()
    return conf

conf = load_json( sys.argv[1] )

param_path = conf[u'param_path'] 
data_dir   = conf[u'test_data_dir'] 
output_dir = conf[u'output_dir'] 
box        = conf[u'box'] # cropped region
shape = (64, 64)         

n_samples = 5

seed = 1234
np.random.seed( seed )

# Build VAEGAN
vaegan = VAEGAN( seed )

try:
    fin = open( param_path, u'rb' )
    params = cPickle.load( fin )
    params = [ p.astype( theano.config.floatX ) for p in params ]
    fin.close()
except:
    print >>sys.stderr, u'Cannot open file:', param_path
    sys.exit(-1)

vaegan.load_params( params )

data = load_data( data_dir, shape, box )

indices = np.arange( len(data) )
np.random.shuffle( indices )
sources = data[ indices[:n_samples] ]
np.random.shuffle( indices )
data1 = data[ indices[:n_samples] ]
np.random.shuffle( indices )
data2 = data[ indices[:n_samples] ]

operate = Operate( vaegan, sources, output_dir )
operate.plus(data1)
operate.minus(data2)
operate.equal()

source_arrs = np.asarray( map( postprocess, sources ) )
source_imgs = image_grid( source_arrs, (n_samples,1) )
source_imgs.save( u'%s/sources.jpeg' % output_dir )

data1_arrs  = np.asarray( map( postprocess, data1 ) )
data1_imgs  = image_grid( data1_arrs, (n_samples,1) )
data1_imgs.save( u'%s/data1.jpeg' % output_dir )

data2_arrs  = np.asarray( map( postprocess, data2 ) )
data2_imgs  = image_grid( data2_arrs, (n_samples,1) )
data2_imgs.save( u'%s/data2.jpeg' % output_dir )


