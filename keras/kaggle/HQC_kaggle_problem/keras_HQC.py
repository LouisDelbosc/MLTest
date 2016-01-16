import numpy as np
import pandas as pd
from keras.regularizers import l2, activity_l2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, AutoEncoder
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adagrad, SGD, Adadelta
from keras.callbacks import EarlyStopping
from keras.layers import containers
from keras.constraints import maxnorm
from keras.utils import np_utils, generic_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(1778) # for reproducibility
need_normalise = True
need_validation = True
need_categorical = False
save_categorical = False
save_categorical_file = False
nb_epoch = 1 #400
golden_feature = [("CoverageField1B", "PropertyField21B"),
                  ("GeographicField6A", "GeographicField8A"),
                  ("GeographicField6A", "GeographicField13A"),
                  ("GeographicField8A", "GeographicField13A"),
                  ("GeographicField11A", "GeographicField13A"),
                  ("GeographicField8A", "GeographicField11A")]

def save2model(submission, file_name, y_pre):
    assert len(y_pre) == len(submission)
    submission['QuoteConversion_Flag'] = y_pre
    submission.to_csv(file_name, index=False)
    print ("saved files %s" % file_name)

def getDummy(df, col):
    category_values = df[col].unique()
    data = [[0 for i in range(len(category_values))] for i in range(len(df))]
    dic_category = dict()
    for i, val in enumerate(list(category_values)):
        dic_category[str(val)] = i
    for i in range(len(df)):
        data[i][dic_category[str(df[col][i])]] = 1
    data = np.array(data)
    for i, val in enumerate(list(category_values)):
        df.loc[:, "_".join([col, str(val)])] = data[:, i]
