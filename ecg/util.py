import os
import pickle  # cPickle is merged into pickle in Python 3

def load(dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'rb') as fid:  # Use 'rb' for reading binary files
        preproc = pickle.load(fid)
    return preproc

def save(preproc, dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'wb') as fid:  # Use 'wb' for writing binary files
        pickle.dump(preproc, fid)
