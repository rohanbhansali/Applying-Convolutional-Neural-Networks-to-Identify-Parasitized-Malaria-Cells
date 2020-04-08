#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:16:40 2020

@author: rohan
"""

from numpy import loadtxt
from keras.models import load_model
 
# load model
model = load_model('SimpNet-7_weights.h5')