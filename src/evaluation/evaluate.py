# Import Libraries
import os
import argparse

import pandas as pd
import numpy as np

import pickle
import json
from scipy import sparse
from sklearn.metrics import mean_squared_error

import lightgbm as lgb


# Define Arguments for argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/models/baseline_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/baseline_model',
                    help="Directory containing the dataset")

# Define functions
def mean_absolute_percentage_error(y_true,y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == '__main__':
    args = parser.parse_args()

    # Read Data
    os.chdir(args.data_dir)

    # Read Training Set
    X_train = sparse.load_npz('train/X_train.npz')
    y_train = pd.read_hdf('train/y_train.hdf', key='train')

    #Read Validation Set
    X_valid = sparse.load_npz('valid/X_valid.npz')
    y_valid = pd.read_hdf('valid/y_valid.hdf', key='valid')


    # Read Test Set
    X_test = sparse.load_npz('test/X_test.npz')
    y_test = pd.read_hdf('test/y_test.hdf', key='test')


    # Read Column Names
    columns = pd.read_csv('columns.csv')
    columns = columns.T.rename(columns=columns.T.iloc[0])
    columns = np.array(columns)

    # Read Pickle File
    os.chdir(args.model_dir)

    # Load Model from pickle file
    gbm = pickle.load(open('model.sav', 'rb'))

    # Get predictions on test set
    y_pred_test = gbm.predict(X_test)

    print('RMSE',np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print('MAPE',mean_absolute_percentage_error(y_test, y_pred_test))

