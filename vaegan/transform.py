# coding : utf-8

import os
import numpy as np
import theano
from data import preprocess, postprocess
from PIL import Image

def image_grid( arrs, grid_shape ):
    row_arrs = []
    for i in range( grid_shape[0] ):
        img_l = range( i*grid_shape[1], i*grid_shape[1]+grid_shape[1] )
        row_arrs.append( np.hstack( [ arrs[t] for t in img_l ] ) )

    arr = np.vstack( row_arrs )
    img = Image.fromarray( np.uint8(arr) )
    
    return img

class Morph( object ):
    def __init__( self, model, form_type, sources, sinks=None, output_dir=u'./tmp', 
                  shape=(10,10), n_steps=10 ):
        self.model = model
        self.form_type = form_type
        self.sources = sources
        self.shape = shape
        self.n_steps = n_steps
        self.output_dir = output_dir
        self.g_id = 0

        if sinks == None:
            self.sink_z = np.asarray( np.random.randn( len(self.sources), 
                                                       model.n_hidden ),
                                      dtype=theano.config.floatX )
            self.sinks = None
        else:
            self.sinks = sinks

        if not os.path.isdir( self.output_dir ):
            os.makedirs( self.output_dir )

    def __call__( self ):
        sample_z = self.model.encode( self.sources )
        if self.sinks != None:
            sink_z = self.model.encode( self.sinks )
        else:
            sink_z = self.sink_z

        self.paths = [ (1.-r)*sample_z + r*sink_z 
                       for r in np.linspace(0.,1.,self.n_steps ) ]

        if self.form_type == 0:
            for i,p in enumerate(self.paths):
                decoded = self.model.decode(p)
                decoded = np.asarray( map( postprocess, decoded ) )
                grid_img = image_grid( decoded, self.shape )
                grid_img.save( u'%s/morphing_%d_%d.jpeg' 
                               % (self.output_dir, self.g_id, i) )
        elif self.form_type == 1:
            cor_arrs = [ np.vstack( map( postprocess, self.sources ) ) ]
            for i,p in enumerate(self.paths):
                decoded = self.model.decode(p)
                decoded = np.asarray( map( postprocess, decoded ) )
                cor_arrs.append( np.vstack( decoded ) )

            if self.sinks != None:
                cor_arrs.append( np.vstack( map( postprocess, self.sinks ) ) )

            arr = np.hstack( cor_arrs )
            path_img = Image.fromarray( np.uint8(arr) )
            path_img.save( u'%s/morphing_%d.jpeg' % (self.output_dir, self.g_id) )
        else:
            raise Exception( u'form_type:%s is not supported.' % self.form_type )

        self.g_id += 1

class Reconstruct( object ):
    def __init__( self, model, samples, output_dir=u'./tmp', shape=(10,10) ):
        self.model = model
        self.samples = samples
        self.shape = shape
        self.output_dir = output_dir
        self.g_id = 0

        if not os.path.isdir( self.output_dir ):
            os.makedirs( self.output_dir )

    def __call__( self ):
        reconstructed = self.model.reconstruct( self.samples )
        reconstructed = np.asarray( map( postprocess, reconstructed ) )
        grid_img = image_grid( reconstructed, self.shape )
        grid_img.save( u'%s/reconstruct_%d.jpeg' 
                       % (self.output_dir, self.g_id) )
        self.g_id += 1

class Operate( object ):
    def __init__( self, model, sources, output_dir=u'./tmp' ):
        self.model = model
        self.zs = self.model.encode( sources )
        self.output_dir = output_dir
        self.g_id = 0

        if not os.path.isdir( self.output_dir ):
            os.makedirs( self.output_dir )

    def plus( self, arrs ):
        self.zs += self.model.encode( arrs )

    def minus( self, arrs ):
        self.zs -= self.model.encode( arrs )

    def equal( self ):
        decoded = self.model.decode( self.zs )
        results = np.asarray( map( postprocess, decoded ) )

        grid_img = image_grid( results, (len(results),1) )
        grid_img.save( u'%s/operate_%d.jpeg' 
                       % (self.output_dir, self.g_id) )

        self.g_id += 1

