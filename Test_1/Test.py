from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import RFclass
import Preprocessdata 
#import Preprocessdata1 as p
import MySVM as mysvc
import TAlogistic as tl
import cPickle, theano
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
import TAmlp as mlp


import TAdbn as dbn
import TAsda as sda
from scipy.interpolate import spline


import tensorflow as tf

hello = tf.constant('hello, TensorFlow')
sess = tf.Session()
print sess.run(hello)
a = tf.constant(10)
b = tf.constant(10)
print sess.run(a+b)