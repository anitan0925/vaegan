# coding : utf-8

"""
< Optimizing Procedure >
"""

import time
import numpy as np
import theano
import theano.tensor as tensor
from ..utils import minibatches
from ..utils import Progress
import cPickle
import os
import sys 

NUM    = 0
FLUSH_BUF = True
SAVING_INTERVAL = 5
MONITORING_INTERVAL = 5

def set_num( n ):
    global NUM
    NUM = n

def save_params( params, params_dir, param_name=None ):
    global NUM
    if param_name == None:
        fout = open( u'%s/param_%d.model' % (params_dir,NUM), u'wb' )
    else:
        fout = open( u'%s/%s.model' % (params_dir,param_name), u'wb' )
    cPickle.dump( [ p.get_value() for p in params ], fout )
    fout.close()
    NUM += 1

def opt_proc( optimizer, X, T, params_dir, callback=None, Xt=None ):
    """
    Run algorigthm for T epochs on training data X.
    
    Arguments
    ---------
    optimizer          : instance of optimizer class.
    X                  : numpy array. 
    Data.
    T                  : integer or float..
    callback           : function.
    Xt                 : numpy array. 
                         Test data.
    """

    if not os.path.isdir( params_dir ):
        os.makedirs( params_dir )

    acc_time = 0.
    acc_itr  = 0
    prog = Progress( len( list( 
        minibatches( optimizer.minibatch_size, X ) ) ), 50 )
    for epoch in range(T):
        itr = 0
        stime = time.time()
        for Xb in minibatches( optimizer.minibatch_size, X, shuffle_f=True ):
            if FLUSH_BUF:
                prog.prog()
            eb = np.asarray( np.random.randn( Xb.shape[0], optimizer.model.n_hidden ),
                             dtype = theano.config.floatX )
            zb = np.asarray( np.random.randn( Xb.shape[0], optimizer.model.n_hidden ),
                             dtype = theano.config.floatX )

            if optimizer.model.phase == 0:
                optimizer.update_funcs[0]( Xb, eb )
                optimizer.update_funcs[1]( Xb, zb )
            else:
                gen_update = True
                gan_update = True
                real_cost = optimizer.model.real_cost_func( Xb )
                fake_cost = optimizer.model.fake_cost_func( Xb, eb, zb )
                equilibrium = 0.68
                margin = 0.4

                if real_cost < equilibrium - margin or \
                   fake_cost < equilibrium - margin:
                    gan_update = False
                if real_cost > equilibrium + margin or \
                   fake_cost > equilibrium + margin:
                    gen_update = False
                if not (gen_update or gan_update):
                    gen_update = True
                    gan_update = True

                if gen_update:
                    optimizer.update_funcs[0]( Xb, eb, zb )
                if gan_update:
                    optimizer.update_funcs[1]( Xb, eb, zb )

            if hasattr( optimizer, 'updates' ):
                optimizer.updates()

            itr += 1

        prog.end() 
        etime = time.time()
        acc_time += etime - stime
        acc_itr  += itr
        print u'Epoch: %d, Iterations: %d, Time: %f' % \
            ( epoch+1, acc_itr, acc_time )

        if (epoch+1) % MONITORING_INTERVAL == 0:
            print u'train:', optimizer.model.cost( X )
            if Xt != None:
                print u'test:', optimizer.model.cost( Xt )

        if (epoch+1) % SAVING_INTERVAL == 0 or epoch == T-1:
            if epoch == T-1:
                param_name = u'param'
            else:
                param_name = None
            save_params( optimizer.model.gen_params + optimizer.model.gan_params, 
                         params_dir, param_name )
            callback()

