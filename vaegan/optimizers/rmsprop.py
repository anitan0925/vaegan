# coding : utf-8
"""
< RMSPROP >
"""

from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as tensor
from opt_proc import opt_proc

class RMSProp( object ):
    """
    RMSProp
    """

    def __init__( self, model, eta=1e-2, rho=0.9, epsilon=1e-6, minibatch_size=10 ):
        """
        Initialize RMSPROP.

        Arguments
        ---------
        model          : model instance should equip params, grad(), [and updates].
        eta            : float.
                         Learning rate.
        rho            : float.
        epsilon        : float.
                         Constant for numerical stability.
        minibatch_size : integer.
                         Minibatch size to calcurate stochastic gradient.        
        """
        self.model            = model
        self.__eta            = eta
        self.__rho            = rho  
        self.__eps            = epsilon
        self.minibatch_size = minibatch_size

        self.__compile()

    def __compile( self ):
        self.update_funcs = []
        for params, inputs, cost in self.model.get_opt_infos():
            # Shared variables for acc.
            accs = [ theano.shared( 
                np.zeros( p.get_value().shape, dtype=theano.config.floatX ) ) 
                     for p in params ]

            sgrad = tensor.grad( cost, params )

            new_accs = [ self.__rho * acc + (1 - self.__rho) * sg ** 2 
                         for (acc, sg) in zip( accs, sgrad ) ]

            updates = OrderedDict()
            updates.update( zip( accs, new_accs ) )
            updates.update( 
                [ (p, p - ( self.__eta * sg / tensor.sqrt( acc_new + self.__eps ) ) ) 
                  for (p, sg, acc_new) 
                  in zip( params, sgrad, new_accs ) ] )

            self.update_funcs.append( theano.function( inputs  = inputs,
                                                       updates = updates ) )

    def run( self, X, T, params_dir=u'./tmp_params', callback=None, Xt=None  ):
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
