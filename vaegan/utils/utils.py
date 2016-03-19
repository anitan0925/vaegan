# coding : utf-8

import numpy as np

def minibatches( minibatch_size, X, Y=None, shuffle_f=False ):
    indices = np.arange( len(X) )
    if shuffle_f:
        np.random.shuffle( indices )

    for start in range( 0, len(X), minibatch_size ):
        minibatch_indices = indices[ start : start+minibatch_size ]
        if Y:
            yield X[minibatch_indices], Y[minibatch_indices] 
        else:
            yield X[minibatch_indices]
