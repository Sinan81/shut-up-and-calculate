#!/usr/bin/env python3

from models import *

def get_list_of_models():
    print('List of all models:')
    print('====================')
    for x in globals():
        if type(eval(x)) == Model:
            print(x)
