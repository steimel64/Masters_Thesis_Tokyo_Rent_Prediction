#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import os
import argparse

import pandas as pd
import numpy as np

import pickle
import json
from scipy import sparse
import lightgbm as lgb


# Define Arguments for argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/models/baseline_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/baseline_model',
                    help="Directory containing the dataset")


# Customized Eval Function - RMSLE
def lgb_rmsle(preds, train_data):
    labels = train_data.get_label()
    return 'rmsle', np.sqrt(np.mean((np.log1p(labels) - np.log1p(preds))**2)), False


# Main
if __name__ == '__main__':

    args = parser.parse_args()
    # Json Path
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    # read json
    with open(json_path, 'r') as json_file:
        params = json.load(json_file)

    # Change to data directory
    os.chdir(args.data_dir)

    # Read Training Set
    X_train = sparse.load_npz('train/X_train.npz')
    X_valid = sparse.load_npz('valid/X_valid.npz')

    # Read Validation Set
    y_train = pd.read_hdf('train/y_train.hdf', key='train')
    y_valid = pd.read_hdf('valid/y_valid.hdf', key='valid')

    # Read Column Names
    columns = pd.read_csv('columns.csv')
    columns = columns.T.rename(columns=columns.T.iloc[0])
    columns = np.array(columns)
    print(columns[0].shape)

    # Create Data for Light GBM
    train_data = lgb.Dataset(X_train, label=y_train)
    eval_data = lgb.Dataset(X_valid, label=y_valid)

    # Train LGBM Model
    gbm = lgb.train(params, train_data, feval=lgb_rmsle, num_boost_round=1000000, valid_sets=[train_data, eval_data],valid_names=['train','valid'], early_stopping_rounds=1000, verbose_eval=100,)

    # Save Model as Pickle Format
    os.chdir(args.model_dir)
    filename = 'model.sav'
    pickle.dump(gbm, open(filename, 'wb'))

