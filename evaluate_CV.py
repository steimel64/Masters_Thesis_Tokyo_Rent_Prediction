# Import Libraries
import os
import argparse

import pandas as pd
import numpy as np

import json
from scipy import sparse
from scipy.sparse import vstack
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

    # Import Data

    # Read Training Set
    X_train = sparse.load_npz('train/X_train.npz')
    X_valid = sparse.load_npz('valid/X_valid.npz')

    # Read Validation Set
    y_train = pd.read_hdf('train/y_train.hdf', key='train')
    y_valid = pd.read_hdf('valid/y_valid.hdf', key='valid')

    # Combine X_train/XValid
    X_train_CV = vstack([X_train, X_valid])

    # combine y_train/y_valid
    y_train_CV = np.concatenate([y_train, y_valid])

    train_data = lgb.Dataset(X_train_CV, label=y_train_CV)

    ## This code calculates CV Scores for RMSE, RMSLE, MAPE for 5 seeds and appends their scores to lists
    seeds = [36, 41, 89, 54, 13]
    rmse = []
    rmsle = []
    mape = []

    for seed in seeds:
        cv_scores = lgb.cv(params, train_data, num_boost_round=1000000, metrics=['mape', 'rmse'], feval=lgb_rmsle,
                           stratified=False, early_stopping_rounds=1000, verbose_eval=100, seed=seed)
        rmse.append(np.array(cv_scores['rmse-mean']).min())
        rmsle.append(np.array(cv_scores['rmsle-mean']).min())
        mape.append(np.array(cv_scores['mape-mean']).min())
        print('Score for seed-', seed, 'rmse-', rmse, 'rmsle-', rmsle, 'mape-', mape)

    cvresults = pd.DataFrame({'rmse': rmse, 'rmsle': rmsle, 'mape': mape})

    print(cvresults.mean())

    # Change to data directory
    os.chdir(args.model_dir)

    # Print Results to CV
    cvresults.to_csv('CV_Results.csv', index=False)

