# import libaries
from __future__ import print_function
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import csv as csv
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# import data and preprocessing the data
dataset_path = 'data'
train_datafile_features = os.path.join(dataset_path, 'test.csv')
train_data_features = pd.read_csv(train_datafile_features)
train_data_features['salary'] = int(train_data_features['salary'])
