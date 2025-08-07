#!/usr/bin/env python3

import numpy as np
#from models import *

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

def rotate_meshgrid(X,Y):
    # Flatten and stack into (x, y) pairs
    points = np.column_stack((X.ravel(), Y.ravel()))

    # Define rotation function
    def rotate_45_deg(points):
        theta = np.pi / 4  # 45 degrees in radians
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        return points @ rotation_matrix.T

    # Apply rotation
    rotated_points = rotate_45_deg(points)

    # Reshape to match original grid shape
    X_rot = rotated_points[:, 0].reshape(X.shape)
    Y_rot = rotated_points[:, 1].reshape(Y.shape)
    return X_rot,Y_rot
