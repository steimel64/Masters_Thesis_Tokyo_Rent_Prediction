# Import Libraries
import os
import argparse

import pandas as pd
import numpy as np

import json
from scipy import sparse
from scipy.sparse import vstack
import lightgbm as lgb
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.5)

# Define Arguments for argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',
                    default='/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/models/baseline_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir',
                    default='/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/baseline_model',
                    help="Directory containing the dataset")


# Customized Eval Function - RMSLE
def lgb_rmsle(preds, train_data):
    labels = train_data.get_label()
    return 'rmsle', np.sqrt(np.mean((np.log1p(labels) - np.log1p(preds)) ** 2)), False


# Main
if __name__ == '__main__':
    args = parser.parse_args()

    # Change to data directory
    os.chdir(args.data_dir)

    # Read Test Set
    X_test = sparse.load_npz('test/X_test.npz')
    y_test = pd.read_hdf('test/y_test.hdf', key='test')
    columns = pd.read_csv('columns.csv')

    # Change to data directory
    os.chdir(args.model_dir)
    gbm = pickle.load(open('model.sav', 'rb'))

    # Generate Predictions
    print('Generating Predictions')
    y_pred_test = gbm.predict(X_test)

    ## Feature Importance Graph 1
    print('Generate Feature Importance Graph 1')
    feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importance(), columns[0])),
                               columns=['Value', 'Feature'])

    plt.figure(figsize=(6, 6))

    barplot = sns.barplot(x="Value", y="Feature", palette=("Blues_d"), alpha=0.85,
                          data=feature_imp.sort_values(by="Value",
                                                       ascending=False)[0:15])

    plt.tight_layout()
    plt.show()
    plt.savefig('lgbm_importances.png', bbox_inches="tight")