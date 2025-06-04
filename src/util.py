#!/usr/bin/env python3

from models import *

def get_list_of_models():
    print('List of all models:')
    print('====================')
    for x in globals():
        if type(eval(x)) == Model:
            print(x)

def get_max_3D(zxy):
    """
    input is a 3D data tuple like (Z,X,Y)
    find max Z, and corresponding x,y values
    """
    Z, X, Y = zxy
    ind = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
    print("max Z is:", Z[ind]," located at qx=",X[ind]," qy=",Y[ind])
    return Z[ind], (X[ind], Y[ind])
