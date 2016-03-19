# coding : utf-8

"""
< ADAM >
"""

from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as tensor
from opt_proc import opt_proc

class Adam( object ):
    """
    ADAM
    """

    def __init__( self, model, eta=1e-3, beta1=0.9, beta2=0.999, 
                  epsilon=1e-8, minibatch_size=10 ):

        """
        Initialize ADAM.

        Arguments
        ---------
        model          : model instance should equip params, grad(), [and updates].
        eta            : float.
                         Learning rate.
        beta1, beta2   : float.
        epsilon        : float.
                         Constant for numerical stability.
        minibatch_size : integer.
                         Minibatch size to calcurate stochastic gradient.        
        """
        self.model            = model
        self.__eta            = eta
        self.__beta1          = beta1  
        self.__beta2          = beta2 
        self.__eps            = epsilon
        self.minibatch_size  = minibatch_size

        self.__compile()

    def __compile( self ):
        t = theano.shared( np.asarray( 0.,dtype=theano.config.floatX ) )
        new_t = t + 1

        a_t = self.__eta * tensor.sqrt( 1 - self.__beta2 ** new_t ) \
              / ( 1 - self.__beta1 ** new_t )

        # Shared variables for ms and ves.
        self.update_funcs = []
        for params, inputs, cost in self.model.get_opt_infos():
            ms = [ theano.shared( 
                np.zeros( p.get_value().shape, dtype=theano.config.floatX ) )
                       for p in params ]
            ves = [ theano.shared( 
                np.zeros( p.get_value().shape, dtype=theano.config.floatX ) ) 
                        for p in params ]

            sgrad = tensor.grad( cost, params )

            new_ms = [ self.__beta1 * m + ( 1 - self.__beta1 ) * sg 
                       for (m, sg)  in zip( ms, sgrad ) ]
            new_ves = [ self.__beta2 * ve + ( 1 - self.__beta2 ) * (sg ** 2) 
                        for (ve, sg)  in zip( ves, sgrad ) ]

            steps = [ a_t * new_m / ( tensor.sqrt(new_ve) + self.__eps ) 
                      for (new_m, new_ve) in zip( new_ms, new_ves ) ]

            updates = OrderedDict()
            updates.update( zip( ms, new_ms ) )
            updates.update( zip( ves, new_ves ) )
            updates.update( [ (p, p - step ) for (p, step) 
                             in zip( params, steps ) ] )

            self.update_funcs.append( theano.function( inputs  = inputs,
                                                       updates = updates ) )

        self.updates = theano.function( [], updates=[ (t,new_t) ] )

    def run( self, X, T, params_dir=u'./tmp_params', callback=None, Xt=None ):
        """
        Run algorigthm for T epochs on training data X.

        Arguments
        ---------
        X                  : numpy array. 
                             Data.
        T                  : integer or float..
        params_dir         : str.
                             Path to directory in which params will be saved.
        callback           : function.
        Xt                 : numpy array. 
                             Test data.
        """
        opt_proc( self, X, T, params_dir, callback, Xt )
