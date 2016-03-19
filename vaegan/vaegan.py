# coding : utf-8
"""
< Variational auto-encoder > 
"""

import numpy as np
import theano
import theano.tensor as tensor
from theano.tensor.shared_randomstreams import RandomStreams
from functions import *
import utils

def uniform_param( shape, name, scale=0.1 ):
    return theano.shared( 
        np.random.uniform( size=shape, low=-scale, high=scale ).astype( 
        dtype=theano.config.floatX ), name=name )

def zeros_param( shape, name ):
    return theano.shared( np.zeros( shape, dtype=theano.config.floatX ), name=name )

class VAEGAN( object ):
    def __init__( self, image_size, phase=0, seed=1234 ):
        self.phase = 0
        self.alpha = 0.5

        # Random 
        np.random.seed( seed )
        self.numpy_rng  = np.random.RandomState( seed )
        self.theano_rng = RandomStreams( self.numpy_rng.randint( 2**30 ) )

        self.x = tensor.tensor4()
        self.e = tensor.matrix()
        self.z_in = tensor.matrix()
        self.n_hidden = 512
        self.image_size = 64

        # Build network
        ## VAE
        mu, log_sigma, self.z = self._build_encoding_layer( self.x, self.e )
        self.y = self._build_decoding_layer( self.z )

        self.gen_params = self.encoding_params + self.decoding_params

        ## GAN
        p_org,  v_org = self._build_gan_layer( self.x )
        p_org2, v_gen = self._build_gan_layer( self.y )
        p_gen = 1 - p_org2

        self.y_out = self._build_decoding_layer( self.z_in )
        p_org3, v_gen2 = self._build_gan_layer( self.y_out )
        p_gen2 = 1-p_org3

        # Build cost
        minibatch_size = tensor.cast( self.x.shape[0], theano.config.floatX )

        ## Prior (maximize)
        self.prior_cost = 0.5 * tensor.sum( 1 + 2*log_sigma - mu**2 
                                            - tensor.exp(2*log_sigma), 
                                            axis=1 ).mean()

        ## Early phase (train VAE using pixel-wise error + discriminator saparately)
        ## Reconstruct (minimize w.r.t. generative params )
        self.reconstruct_cost_vis = 0.5 * tensor.sum( tensor.sqr(self.x - self.y),
                                                      axis=(1,2,3) ).mean() 

        ## GAN (maximize w.r.t discriminator params )
        self.gan_logprob = tensor.mean( tensor.log( p_org ) + tensor.log( p_gen2 ) ) 

        ## Overall cost w.r.t generator in early pahse
        ## inputs: [ self.x, self.e ]
        self.early_cost_gen = self.reconstruct_cost_vis - self.prior_cost \

        ## Overall cost w.r.t discriminator in early pahse
        ## inputs: [ self.z_in ]
        self.early_cost_dis = - self.gan_logprob

        ## Monitoring cost in early phase
        self.early_monitoring_cost = [ self.reconstruct_cost_vis - self.prior_cost,
                                       self.gan_logprob ]

        self.early_opt_infos = [ 
            [ self.gen_params, [self.x, self.e], self.early_cost_gen ],
            [ self.gan_params, [self.x, self.z_in], self.early_cost_dis ] ]

        ## Final phase
        ## Reconstruct (minimize w.r.t. generative params )
        self.reconstruct_cost_hid = 0.5 * tensor.sum( tensor.sqr(v_org - v_gen), 
                                                      axis=(1,2,3) ).mean()

        ## GAN (maximize w.r.t discriminator params )
        self.real_cost = - tensor.mean( tensor.log( p_org ) )
        self.fake_cost = - 0.5 * ( tensor.mean( tensor.log( p_gen  ) ) \
                         + tensor.mean( tensor.log( p_gen2 ) ) )
        self.gan_logprob_plus = - self.real_cost - self.fake_cost

        ## GAN (minimize w.r.t generative params )
        self.gan_logprob_gen = 0.5 * tensor.mean( tensor.log( p_gen ) 
                                                  + tensor.log( p_gen2 ) ) 
        # To balance the progresses, we uses the following proxy objective.
        # self.gan_logprob_gen = - tensor.mean( tensor.log( 1 - p_gen )
        #                                       - tensor.log( 1 - p_gen2 ) ) 

        ## Oveall cost w.r.t generator in final phase
        ## inputs: [ self.x, self.e. self.z_in ] 
        self.final_cost_gen = self.alpha * ( self.reconstruct_cost_hid \
                                             - self.prior_cost ) \
            + self.gan_logprob_gen

        ## Oveall cost w.r.t discriminator in final phase
        ## inputs: [ self.x, self.e, self.z_in ]
        self.final_cost_dis = - self.gan_logprob_plus

        ## monitoring cost in final phase
        self.final_monitoring_cost = self.alpha * ( self.reconstruct_cost_hid \
                                                    - self.prior_cost ) \
            + self.gan_logprob_plus

        self.final_opt_infos = [ 
            [ self.gen_params, [self.x, self.e, self.z_in], self.final_cost_gen ],
            [ self.gan_params, [self.x, self.e, self.z_in], self.final_cost_dis ] ]

        # Compile functions
        self.reconstruct_func = theano.function( [self.x,self.e], self.y )
        self.encode_func = theano.function( [self.x,self.e], self.z )
        self.decode_func = theano.function( [self.z_in], self.y_out )
        self.early_cost_func = theano.function( [self.x,self.e,self.z_in],
                                               self.early_monitoring_cost )
        self.final_cost_func = theano.function( [self.x,self.e,self.z_in], 
                                                self.final_monitoring_cost )
        self.real_cost_func = theano.function( [self.x],self.real_cost )
        self.fake_cost_func = theano.function( [self.x,self.e,self.z_in],
                                               self.fake_cost )

    def load_params( self, params ):
        for (p_,p) in zip( params, self.gen_params + self.gan_params ):
            p.set_value(p_)

    def set_phase( self, phase ):
        self.phase = phase

    def get_opt_infos( self ):
        if self.phase == 0:
            return self.early_opt_infos
        else:
            return self.final_opt_infos

    def _build_encoding_layer( self, x, e ):
        down_size  = self.image_size // 8

        if not hasattr( self, 'encoding_params' ):
            we1  = uniform_param( (64, 3, 5, 5), u'we1' )
            bwe1 = uniform_param( (64), u'bwe1' )
            bbe1 = zeros_param( (64), u'bbe1' )
            we2  = uniform_param( (128, 64, 5, 5), u'we2' ) 
            bwe2 = uniform_param( (128), u'bwe2' )
            bbe2 = zeros_param( (128), u'bbe2' )
            we3  = uniform_param( (256, 128, 5, 5), u'we3' ) 
            bwe3 = uniform_param( (256), u'bwe3' )
            bbe3 = zeros_param( (256), u'bbe3' )
            we4  = uniform_param( (256*(down_size**2), 2048), u'we4' )
            be4  = zeros_param( (2048), u'be4' ) 
            bwe4 = uniform_param( (2048), u'bwe4' )
            bbe4 = zeros_param( (2048), u'bbe4' )
            wmu  = uniform_param( (2048, self.n_hidden), u'wmu' )
            bmu  = zeros_param( (self.n_hidden), u'bmu' )
            wsigma = uniform_param( (2048, self.n_hidden), u'wsigma' )
            bsigma = zeros_param( (self.n_hidden), u'bsigma' )

            self.encoding_params = [ we1, bwe1, bbe1, we2, bwe2, bbe2, 
                                     we3, bwe3, bbe3,
                                     we4, be4, bwe4, bbe4,
                                     wmu, bmu, wsigma, bsigma ]

        [ we1, bwe1, bbe1, we2, bwe2, bbe2, we3, bwe3, bbe3, 
          we4, be4, bwe4, bbe4,
          wmu, bmu, wsigma, bsigma ] = self.encoding_params

        h1  = rectify( batchnorm( max_pool( conv( x,  we1 ), (2,2) ), bwe1, bbe1 ) )
        h2  = rectify( batchnorm( max_pool( conv( h1, we2 ), (2,2) ), bwe2, bbe2 ) )
        h3  = rectify( batchnorm( max_pool( conv( h2, we3 ), (2,2) ), bwe3, bbe3 ) )
        h3_ = h3.reshape( (-1, 256*(down_size**2)) )
        h4  = rectify( batchnorm( full_conn( h3_, we4, be4 ), bwe4, bbe4 ) )

        mu = theano.dot( h4, wmu ) + bmu
        log_sigma = 0.5 * ( tensor.dot( h4, wsigma) + bsigma )
        # The number of e should be the same as x.
        z = mu + tensor.exp( log_sigma ) * e 

        return mu, log_sigma, z

    def _build_decoding_layer( self, z ):
        down_size = self.image_size // 8

        if not hasattr( self, 'decoding_params' ):
            wd4  = uniform_param( (self.n_hidden, 256*(down_size**2) ), u'wd4' )
            bd4  = zeros_param( (256*(down_size**2)), u'bd4' )
            bwd4 = uniform_param( (256*(down_size**2)), u'bwd4' )
            bbd4 = zeros_param( (256*(down_size**2)), u'bbd4' )
            wd3  = uniform_param( (256,256,5,5), u'wd3' )
            bwd3 = uniform_param( (256), u'bwd3' )
            bbd3 = zeros_param( (256), u'bbd3' )
            wd2  = uniform_param( (128,256,5,5), u'wd2' )
            bwd2 = uniform_param( (128), u'bwd2' )
            bbd2 = zeros_param( (128), u'bbd2' )
            wd1  = uniform_param( (32,128,5,5), u'wd1' )
            bwd1 = uniform_param( (32), u'bwd1' )
            bbd1 = zeros_param( (32), u'bbd1' )
            wd0  = uniform_param( (3,32,5,5), u'wd0' ) 

            self.decoding_params = [ wd4, bd4, bwd4, bbd4, 
                                     wd3, bwd3, bbd3, 
                                     wd2, bwd2, bbd2, wd1, bwd1, bbd1, wd0 ]

        [ wd4, bd4, bwd4, bbd4, wd3, bwd3, bbd3, 
          wd2, bwd2, bbd2, wd1, bwd1, bbd1, wd0 ] \
            = self.decoding_params
        
        h1  = rectify( batchnorm( full_conn( z,  wd4, bd4 ), bwd4, bbd4 ) )
        h1_ = h1.reshape( (-1,256, down_size, down_size) )
        h2  = rectify( batchnorm( conv( depool( h1_, factor=2 ), wd3 ), bwd3, bbd3 ) )
        h3  = rectify( batchnorm( conv( depool( h2 , factor=2 ), wd2 ), bwd2, bbd2 ) )
        h4  = rectify( batchnorm( conv( depool( h3,  factor=2 ), wd1 ), bwd1, bbd1 ) )
        y   = tanh( conv( h4, wd0 ) )
                      
        return y
                    
    def _build_gan_layer( self, x ):
        down_size  = self.image_size // 8

        if not hasattr( self, 'gan_params' ):
            wg1 = uniform_param( (32, 3, 5, 5), u'wg1' )
            wg2 = uniform_param( (128, 32, 5, 5), u'wg2' ) 
            wg3 = uniform_param( (256, 128, 5, 5), u'wg3' ) 
            wg4 = uniform_param( (256, 256, 5, 5), u'wg4' ) 
            wg5 = uniform_param( (256*(down_size**2), 512), u'wg5' )
            bg5 = zeros_param( (512), u'bg5' )
            wg6 = uniform_param( (512, 1), u'wg6' )
            bg6 = zeros_param( (1), u'bgd6' )
            self.gan_params = [ wg1, wg2, wg3, wg4, wg5, bg5, wg6, bg6 ]

        [ wg1, wg2, wg3, wg4, wg5, bg5, wg6, bg6 ] = self.gan_params

        h1 = rectify( conv( x,  wg1 ) )
        h2 = rectify( max_pool( conv( h1, wg2 ), (2,2) ) )
        h3 = rectify( max_pool( conv( h2, wg3 ), (2,2) ) )
        h4 = rectify( max_pool( conv( h3, wg4 ), (2,2) ) )
        h4_ = h4.reshape( (-1,256*(down_size**2)) )
        h5 = rectify( full_conn( h4_, wg5, bg5 ) )

        image_feature = h3

        return sigmoid( full_conn( h5, wg6, bg6 ) ), image_feature

    def reconstruct( self, x ):
        eb = np.asarray( np.ones( (x.shape[0], self.n_hidden) ),
                         dtype = theano.config.floatX )
        return self.reconstruct_func(x,eb)

    def encode( self, x ):
        eb = np.asarray( np.ones( (x.shape[0], self.n_hidden) ),
                         dtype = theano.config.floatX )

        return self.encode_func(x,eb)

    def decode( self, z ):
        return self.decode_func(z)

    def cost( self, X, minibatch_size = 20 ):
        if self.phase == 0:
            val = [0,0]
        else:
            val = 0

        data_size = X.shape[0]
        for Xb in utils.minibatches( minibatch_size, X, shuffle_f=False ):        
            eb = np.asarray( np.random.randn( Xb.shape[0], self.n_hidden ),
                             dtype = theano.config.floatX )
            zb = np.asarray( np.random.randn( Xb.shape[0], self.n_hidden ),
                             dtype = theano.config.floatX )
            if self.phase == 0:
                c = self.early_cost_func( Xb, eb, zb )
                val[0] += c[0] * float(Xb.shape[0]) \
                          / float(data_size)
                val[1] += c[1] * float(Xb.shape[1]) \
                          / float(data_size)
            else:
                val += self.final_cost_func( Xb, eb, zb ) * float(Xb.shape[0]) \
                       / float(data_size)

        return val

