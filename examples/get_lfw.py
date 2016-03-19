# coding : utf-8

import sys
import os
import glob
import random
import shutil

source    = u'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
filename  = u'lfw-deepfunneled.tgz'
org_file  = u'lfw-deepfunneled'
train_dir = u'train_lfw'
test_dir  = u'test_lfw'
train_ratio = 0.9

if not os.path.exists( filename ):
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    urlretrieve( source, filename )

if not os.path.exists( org_file ):
    import tarfile
    tar = tarfile.open( filename )
    tar.extractall()
    tar.close()

file_list  = [ os.path.relpath(x, org_file) for x in glob.glob( u'%s/*' % org_file ) ]
data_size  = len(file_list)
train_size = int( train_ratio * data_size )
test_size  = data_size - train_size
print u'n_train, n_test = %d, %d' % (train_size, test_size)
random.shuffle( file_list )

train_list = file_list[:train_size]
test_list  = file_list[train_size:]

if not os.path.isdir( train_dir ):
    os.makedirs( train_dir )

if not os.path.isdir( test_dir ):
    os.makedirs( test_dir )

map( lambda p : shutil.copytree( os.path.join( org_file, p ), 
                                 os.path.join( train_dir, p ) ), train_list )

map( lambda p : shutil.copytree( os.path.join( org_file, p ), 
                                 os.path.join( test_dir, p ) ), test_list )

