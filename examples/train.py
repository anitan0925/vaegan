# coding : utf-8

import sys
import os
module_path = os.path.abspath( u'..'  )
sys.path.append( module_path )

import json
import numpy as np
import theano
from vaegan.data import load_data
from vaegan import Morph, Reconstruct
from vaegan import VAEGAN
from vaegan.optimizers import Adam

def load_json( infile_name ):
    fin = open( infile_name, u'r' )
    conf = json.load( fin )
    fin.close()
    return conf

conf = load_json( sys.argv[1] )

output_params_dir = conf[u'output_params_dir'] 
output_dir        = conf[u'output_dir'] 
train_data_dir    = conf[u'train_data_dir']
test_data_dir     = conf[u'test_data_dir']
box               = conf[u'box'] # cropped region
shape = (64, 64)         

pretrain = True
seed = 1234
np.random.seed( seed )

# Build VAEGAN
vaegan = VAEGAN( seed )

train_data = load_data( train_data_dir, shape, box )
test_data  = load_data( test_data_dir,  shape, box )

# Set transformation
indices = np.arange( len(train_data) )
grid_shape = (10,10)
np.random.shuffle( indices )
indices = indices[:grid_shape[0]*grid_shape[1]]
form_type = 0 # Form of output images.
morph = Morph( vaegan, form_type, train_data[indices], 
               output_dir=output_dir, shape=grid_shape )

# Train
if pretrain:
    print u'#Early phase'
    solver = Adam( vaegan, eta=1e-3, beta1=0.9, minibatch_size=64 )
    solver.run( train_data, T=20, params_dir=output_params_dir, callback=morph,
                Xt=test_data )

print u'#Final phase'
vaegan.set_phase(1)
solver = Adam( vaegan, eta=1e-3, beta1=0.9, minibatch_size=64 )
solver.run( train_data, T=200, params_dir=output_params_dir, callback=morph,
            Xt=test_data )
