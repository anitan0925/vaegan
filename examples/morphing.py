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
from vaegan import Morph
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

n_samples = 10
n_steps = 10

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
sinks = data[ indices[:n_samples] ]

form_type = 1
morph = Morph( vaegan, form_type, sources, sinks, output_dir, n_steps=10 )
morph()
